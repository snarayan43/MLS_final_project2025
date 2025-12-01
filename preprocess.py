import os
import numpy as np
import cv2
from tqdm import tqdm

# ==========================================
# CONFIGURATION
# ==========================================
SOURCE_DATA_DIR = '../FaceForensics++_C23'
NUM_FRAMES = 32
CROP_SIZE = 224

# Dataset Balancing (1000 Real vs 1000 Fake)
# Using original and 1 deepfake dataset to focus on learnings and improvement due to attention layer
ALL_DIRS = ['original', 'Face2Face']
COUNTS = {
    'original': 1000,
    'Face2Face': 1000, 
}

# Output Folders
OUTPUT_BASES = {
    'full': '../Processed_Data_Full_HighRes',
    'eyes': '../Processed_Data_Eyes_HighRes',
    'mouth': '../Processed_Data_Mouth_HighRes'
}

# ==========================================

# Setup Face Detection
FACE_CASCADE_PATH = 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH) if os.path.exists(FACE_CASCADE_PATH) else None

def get_crop_center(x, y, w, h, region):
    """ Returns (cx, cy) based on the region of interest """
    if region == 'mouth': 
        return x + w // 2, y + int(h * 0.70) # Lower face
    elif region == 'eyes': 
        return x + w // 2, y + int(h * 0.35) # Upper face
    else: 
        return x + w // 2, y + h // 2        # Center

def process_video_single_pass(video_path):
    """
    Reads video once. Returns a dict containing 3 high-res arrays (Full, Eyes, Mouth).
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Initialize containers
    data = {'full': [], 'eyes': [], 'mouth': []}
    
    # Handle empty/corrupt videos
    if total_frames <= 0:
        empty_arr = np.zeros((NUM_FRAMES, CROP_SIZE, CROP_SIZE, 3), dtype=np.uint8)
        return {k: empty_arr for k in data}

    # Sample 32 frames evenly from ~10 second video
    frame_indices = np.linspace(0, total_frames - 1, NUM_FRAMES, dtype=int)
    
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret: continue
        
        # Convert to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect Face (Once per frame)
        faces = []
        if face_cascade:
            faces = face_cascade.detectMultiScale(frame_gray, 1.1, 5, minSize=(50, 50))
        
        # Process all 3 regions for this single frame
        for region in ['full', 'eyes', 'mouth']:
            cropped_frame = None
            
            # 1. Try Smart Crop (if face detected)
            if len(faces) > 0:
                # Pick largest face
                (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])
                cx, cy = get_crop_center(x, y, w, h, region)
                
                half = CROP_SIZE // 2
                x1, y1 = max(0, cx-half), max(0, cy-half)
                x2, y2 = min(frame_rgb.shape[1], cx+half), min(frame_rgb.shape[0], cy+half)
                
                crop = frame_rgb[y1:y2, x1:x2]
                if crop.size > 0:
                    # Use INTER_AREA for high-quality shrinking (critical for texture)
                    cropped_frame = cv2.resize(crop, (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_AREA)
            
            # 2. Fallback Crop (Center of image)
            if cropped_frame is None:
                h, w, _ = frame_rgb.shape
                min_dim = min(h, w)
                sx = (w - min_dim) // 2
                sy = (h - min_dim) // 2
                crop = frame_rgb[sy:sy+min_dim, sx:sx+min_dim]
                cropped_frame = cv2.resize(crop, (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_AREA)
            
            data[region].append(cropped_frame)

    cap.release()
    
    # Pad to exactly 32 frames if video was short
    final_output = {}
    for region, frames in data.items():
        while len(frames) < NUM_FRAMES:
            frames.append(np.zeros((CROP_SIZE, CROP_SIZE, 3), dtype=np.uint8) if not frames else frames[-1])
        final_output[region] = np.array(frames, dtype=np.uint8)
        
    return final_output

# --- MAIN LOOP ---
if __name__ == "__main__":
    print(f"--- Starting High-Fidelity Pre-processing ---")
    print(f"Config: {NUM_FRAMES} Frames | {CROP_SIZE}px Resolution")
    print(f"Source: {os.path.abspath(SOURCE_DATA_DIR)}")
    
    for method in ALL_DIRS:
        input_path = os.path.join(SOURCE_DATA_DIR, method)
        if not os.path.exists(input_path):
            print(f"Skipping {method} (Path not found)")
            continue
            
        # Create output directories
        for base in OUTPUT_BASES.values():
            os.makedirs(os.path.join(base, method), exist_ok=True)
            
        # Sort and limit to 1000
        limit = COUNTS.get(method, 0)
        videos = sorted([f for f in os.listdir(input_path) if f.endswith(('.mp4', '.avi'))])[:limit]
        
        print(f"\nProcessing {method} ({len(videos)} videos)...")
        
        for vid in tqdm(videos):
            vid_path = os.path.join(input_path, vid)
            save_name = vid.replace('.mp4', '.npy').replace('.avi', '.npy')
            
            # Skip if ALL 3 already exist
            all_exist = True
            for base in OUTPUT_BASES.values():
                if not os.path.exists(os.path.join(base, method, save_name)):
                    all_exist = False
                    break
            if all_exist: continue 
            
            # processing video - refer to method
            results = process_video_single_pass(vid_path)
            
            # Save it
            for region, data_array in results.items():
                save_path = os.path.join(OUTPUT_BASES[region], method, save_name)
                np.save(save_path, data_array)
            
    print("\n High-Res Pre-processing Complete!")