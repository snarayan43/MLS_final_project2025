import tensorflow as tf
import os
import numpy as np
from tensorflow.keras.utils import Sequence
from sklearn.metrics import confusion_matrix

from models import build_deepfake_detector_control, build_deepfake_detector_attention

# ==========================================
# GPU SETUP & SAFETY
# ==========================================
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# ==========================================
# EXPERIMENT CONFIGURATION
# to test: control-full; attention-full; attention-eyes; attention-mouth
# ==========================================

MODEL_NAME = 'attention'    # 'control' OR 'attention'
REGION_TO_CROP = 'full'    # 'full', 'eyes', OR 'mouth'

# Constants (matching preprocess.py)
CROP_SIZE = 224
NUM_FRAMES = 32
FRAME_HEIGHT = CROP_SIZE
FRAME_WIDTH = CROP_SIZE
BATCH_SIZE = 2   # gpu can't handle more     

# Dataset Subset
NUM_TRAIN_REAL = 800
NUM_VAL_REAL = 200
NUM_TRAIN_FAKE_PER_METHOD = 800
NUM_VAL_FAKE_PER_METHOD = 200

# Paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(CURRENT_DIR) 
BASE_DATA_DIR = os.path.join(PROJECT_DIR, 'FaceForensics++_C23')
ALL_FAKE_DIRS = ['Face2Face'] 

# ==========================================
# SMART GENERATOR - extracts video array from preprocessed folders
# ==========================================
class VideoDataGenerator(Sequence):
    def __init__(self, file_paths, labels, batch_size, num_frames, 
                 frame_height, frame_width, region, shuffle=True):
        
        self.file_paths = []
        target_folder = f'Processed_Data_{region.capitalize()}_HighRes'
        
        for p in file_paths:
            new_path = p.replace('FaceForensics++_C23', target_folder)
            new_path = new_path.replace('.mp4', '.npy').replace('.avi', '.npy')
            self.file_paths.append(new_path)

        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(self.file_paths))
        if self.shuffle: np.random.shuffle(self.indices)

    def __len__(self):
        return int(np.ceil(len(self.file_paths) / self.batch_size))

    def __getitem__(self, index):
        batch_indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        batch_x, batch_y = [], []

        for idx in batch_indices:
            path = self.file_paths[idx]
            label = self.labels[idx]
            try:
                data = np.load(path)
                batch_x.append(data.astype('float32')) 
                batch_y.append(label)
            except Exception:
                batch_x.append(np.zeros((NUM_FRAMES, FRAME_HEIGHT, FRAME_WIDTH, 3), dtype='float32'))
                batch_y.append(label)

        return np.array(batch_x), np.array(batch_y)

    def on_epoch_end(self):
        if self.shuffle: np.random.shuffle(self.indices)

# ==========================================
# MAIN EXECUTION
# ==========================================
def get_clean_video_list(directory):
    try:
        return sorted([f for f in os.listdir(directory) if not f.startswith('.') and f.endswith(('.mp4', '.avi'))])
    except FileNotFoundError: return []

# --- Collect Data ---
TRAIN_FILE_PATHS, VAL_FILE_PATHS = [], []
TRAIN_LABELS, VAL_LABELS = [], []

print("--- Collecting Data ---")
for method in ['original'] + ALL_FAKE_DIRS:
    source_dir = os.path.join(BASE_DATA_DIR, method)
    video_list = get_clean_video_list(source_dir)
    
    is_real = 1 if method == 'original' else 0
    train_count = NUM_TRAIN_REAL if is_real else NUM_TRAIN_FAKE_PER_METHOD
    val_count = NUM_VAL_REAL if is_real else NUM_VAL_FAKE_PER_METHOD
    val_start = train_count
        
    train_samples = [os.path.join(source_dir, f) for f in video_list[0 : train_count]]
    TRAIN_FILE_PATHS.extend(train_samples)
    TRAIN_LABELS.extend([is_real] * len(train_samples))

    val_samples = [os.path.join(source_dir, f) for f in video_list[val_start : val_start + val_count]]
    VAL_FILE_PATHS.extend(val_samples)
    VAL_LABELS.extend([is_real] * len(val_samples))

# --- Generators ---
train_gen = VideoDataGenerator(TRAIN_FILE_PATHS, TRAIN_LABELS, BATCH_SIZE, NUM_FRAMES, 
                               FRAME_HEIGHT, FRAME_WIDTH, region=REGION_TO_CROP, shuffle=True)
val_gen = VideoDataGenerator(VAL_FILE_PATHS, VAL_LABELS, BATCH_SIZE, NUM_FRAMES, 
                             FRAME_HEIGHT, FRAME_WIDTH, region=REGION_TO_CROP, shuffle=False)

# --- Build Model based on model_name and region given ---
if MODEL_NAME == 'control':
    model = build_deepfake_detector_control(NUM_FRAMES, FRAME_HEIGHT, FRAME_WIDTH)
else:
    model = build_deepfake_detector_attention(NUM_FRAMES, FRAME_HEIGHT, FRAME_WIDTH)

print(f"\n--- Starting Experiment: Model={MODEL_NAME} | Region={REGION_TO_CROP} ---")

# --- Resume Logic if model exists ---
checkpoint_filename = f'best_model_{MODEL_NAME}_{REGION_TO_CROP}.h5'

if os.path.exists(checkpoint_filename):
    print(f"Found existing checkpoint: {checkpoint_filename}")
    print("Loading weights and resuming training...")
    try:
        model.load_weights(checkpoint_filename)
        print("Weights loaded successfully! Training will resume.")
    except Exception as e:
        print(f"Error loading weights: {e}")
        print("Starting from scratch.")
else:
    print("No existing checkpoint found. Starting from scratch.")

# --- Training ---
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        checkpoint_filename,
        save_best_only=True, monitor='val_loss'
    ),
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
]

history = model.fit(train_gen, validation_data=val_gen, epochs=30, callbacks=callbacks)

# --- Evaluation with confusion matrix ---
print("--- Evaluation ---")
val_steps = len(val_gen)
if val_steps > 0:
    y_pred = (model.predict(val_gen, steps=val_steps, verbose=1) > 0.5).astype(int).flatten()
    y_true = []
    val_gen.on_epoch_end()
    for i in range(val_steps):
        _, batch_y = val_gen[i]
        y_true.extend(batch_y.flatten())
    y_true = np.array(y_true)[:len(y_pred)]

    cm = confusion_matrix(y_true, y_pred)
    if cm.size == 4:
        TN, FP, FN, TP = cm.ravel()
        print(f"Accuracy: **{(TP+TN)/len(y_true):.4f}**")
        print(cm)