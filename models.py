import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Dense, LSTM, Bidirectional, TimeDistributed,
    Dropout, BatchNormalization, GlobalAveragePooling1D,
    Softmax, Multiply, Lambda
)
from tensorflow.keras.models import Model
from tensorflow.keras.applications import EfficientNetB1

def build_deepfake_detector_control(num_frames, frame_height, frame_width):
    """ 
    Control Model: CNN + BiLSTM + Global Average Pooling (No Attention).
    Baseline model for comparison.
    """
    input_shape = (num_frames, frame_height, frame_width, 3)
    video_input = Input(shape=input_shape, name='control_input')

    # Unfreeze EfficientNet for fine-tuning
    base_cnn = EfficientNetB1(
        weights='imagenet', include_top=False,
        input_shape=(frame_height, frame_width, 3), pooling='avg'
    )
    base_cnn.trainable = True 

    frame_features = TimeDistributed(base_cnn)(video_input)
    bilstm = Bidirectional(LSTM(128, return_sequences=True))(frame_features)
    
    # Simple Global Average Pooling
    context_vector = GlobalAveragePooling1D()(bilstm)

    x = Dense(64, activation='relu')(context_vector)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    output = Dense(1, activation='sigmoid', dtype='float32')(x)

    model = Model(inputs=video_input, outputs=output, name="Control_Model")
    # Lower learning rate (1e-5) because we unfroze the CNN
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_deepfake_detector_attention(num_frames, frame_height, frame_width):
    """ 
    Attention Model: CNN + BiLSTM + Attention Mechanism.
    Focuses on specific frames (jitters) rather than averaging.
    """
    input_shape = (num_frames, frame_height, frame_width, 3)
    video_input = Input(shape=input_shape, name='attention_input')

    # Unfreeze EfficientNet for fine-tuning
    base_cnn = EfficientNetB1(
        weights='imagenet', include_top=False,
        input_shape=(frame_height, frame_width, 3), pooling='avg'
    )
    base_cnn.trainable = True 

    frame_features = TimeDistributed(base_cnn)(video_input)
    bilstm = Bidirectional(LSTM(128, return_sequences=True))(frame_features)

    # Attention Mechanism
    attention_scores = Dense(1, activation='tanh')(bilstm)
    attention_weights = Softmax(axis=1)(attention_scores)
    context_vector = Multiply()([bilstm, attention_weights])
    context_vector = Lambda(lambda x: tf.reduce_sum(x, axis=1))(context_vector)

    x = Dense(64, activation='relu')(context_vector)
    x = Dropout(0.5)(x)
    output = Dense(1, activation='sigmoid', dtype='float32')(x)

    model = Model(inputs=video_input, outputs=output, name="Attention_Model")
    # Lower learning rate (1e-5) because we unfroze the CNN
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss='binary_crossentropy', metrics=['accuracy'])
    return model