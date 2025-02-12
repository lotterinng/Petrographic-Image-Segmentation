"""
Reinforcement Learning and Transformer-based Segmentation Models

This module defines two new models:
  - rl_segmentation_model: A U-Net–like architecture that includes a reinforcement-learning–inspired 
    policy module at the bottleneck to modulate features.
  - transformer_segmentation_model: A segmentation model that inserts a Transformer encoder block
    at the bottleneck to enhance global feature relationships.
"""

import tensorflow as tf
import segmentation_models as sm
sm.set_framework('tf.keras')

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, 
                                     Conv2DTranspose, Dropout, GlobalAveragePooling2D, Dense, 
                                     Multiply, Reshape, LayerNormalization, MultiHeadAttention, Add)
from tensorflow.keras import backend as K

def rl_segmentation_model(n_classes=4, IMG_HEIGHT=256, IMG_WIDTH=256, IMG_CHANNELS=1):
    """
    RL-Inspired Segmentation Model:
    Uses a U-Net encoder–decoder structure. At the bottleneck, a policy vector is learned (via
    global average pooling and a Dense softmax layer) that is used to modulate the bottleneck features.
    """
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    
    # Encoder (similar to U-Net)
    c1 = Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    c1 = Dropout(0.2)(c1)
    c1 = Conv2D(16, (3, 3), activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)
    
    c2 = Conv2D(32, (3, 3), activation='relu', padding='same')(p1)
    c2 = Dropout(0.2)(c2)
    c2 = Conv2D(32, (3, 3), activation='relu', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)
    
    c3 = Conv2D(64, (3, 3), activation='relu', padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(64, (3, 3), activation='relu', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)
    
    # Bottleneck
    c4 = Conv2D(128, (3, 3), activation='relu', padding='same')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(128, (3, 3), activation='relu', padding='same')(c4)
    
    # Reinforcement Learning Inspired Policy Module
    policy = GlobalAveragePooling2D()(c4)
    policy = Dense(128, activation='softmax', name='policy_vector')(policy)  # Policy vector of size 128
    policy_reshaped = Reshape((1, 1, 128))(policy)
    c4_modulated = Multiply()([c4, policy_reshaped])
    
    # Decoder
    u5 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c4_modulated)
    u5 = concatenate([u5, c3])
    c5 = Conv2D(64, (3, 3), activation='relu', padding='same')(u5)
    c5 = Dropout(0.2)(c5)
    c5 = Conv2D(64, (3, 3), activation='relu', padding='same')(c5)
    
    u6 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c2])
    c6 = Conv2D(32, (3, 3), activation='relu', padding='same')(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(32, (3, 3), activation='relu', padding='same')(c6)
    
    u7 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c1])
    c7 = Conv2D(16, (3, 3), activation='relu', padding='same')(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(16, (3, 3), activation='relu', padding='same')(c7)
    
    outputs = Conv2D(n_classes, (1, 1), activation='softmax')(c7)
    
    model = Model(inputs=[inputs], outputs=[outputs])
    return model

def transformer_segmentation_model(n_classes=4, IMG_HEIGHT=256, IMG_WIDTH=256, IMG_CHANNELS=1,
                                   num_heads=4, transformer_units=64):
    """
    Transformer-Based Segmentation Model:
    Uses a U-Net encoder–decoder structure and injects a Transformer encoder block at the bottleneck.
    """
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    
    # Encoder (similar to U-Net)
    c1 = Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    c1 = Dropout(0.2)(c1)
    c1 = Conv2D(16, (3, 3), activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)
    
    c2 = Conv2D(32, (3, 3), activation='relu', padding='same')(p1)
    c2 = Dropout(0.2)(c2)
    c2 = Conv2D(32, (3, 3), activation='relu', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)
    
    c3 = Conv2D(64, (3, 3), activation='relu', padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(64, (3, 3), activation='relu', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)
    
    # Bottleneck convolution
    bottleneck = Conv2D(128, (3, 3), activation='relu', padding='same')(p3)
    bottleneck = Dropout(0.2)(bottleneck)
    bottleneck = Conv2D(128, (3, 3), activation='relu', padding='same')(bottleneck)
    
    # Prepare transformer input: flatten spatial dimensions
    # Here we use static shapes; ensure that your IMG_HEIGHT and IMG_WIDTH are divisible by 8.
    H = bottleneck.shape[1]
    W = bottleneck.shape[2]
    C = bottleneck.shape[3]
    num_patches = H * W
    x = Reshape((num_patches, C))(bottleneck)
    
    # Add learnable positional embeddings
    pos_embed = tf.Variable(tf.random.normal([num_patches, C]), name="pos_embed")
    x = x + pos_embed
    
    # Transformer encoder block
    attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=C, dropout=0.1)(x, x)
    x = Add()([x, attn_output])
    x = LayerNormalization(epsilon=1e-6)(x)
    
    ff = tf.keras.layers.Dense(transformer_units, activation='relu')(x)
    ff = tf.keras.layers.Dense(C)(ff)
    x = Add()([x, ff])
    x = LayerNormalization(epsilon=1e-6)(x)
    
    # Reshape back to spatial dimensions
    x = Reshape((H, W, C))(x)
    
    # Decoder (similar to U-Net)
    u1 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(x)
    u1 = concatenate([u1, c3])
    d1 = Conv2D(64, (3, 3), activation='relu', padding='same')(u1)
    d1 = Dropout(0.2)(d1)
    d1 = Conv2D(64, (3, 3), activation='relu', padding='same')(d1)
    
    u2 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(d1)
    u2 = concatenate([u2, c2])
    d2 = Conv2D(32, (3, 3), activation='relu', padding='same')(u2)
    d2 = Dropout(0.2)(d2)
    d2 = Conv2D(32, (3, 3), activation='relu', padding='same')(d2)
    
    u3 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(d2)
    u3 = concatenate([u3, c1])
    d3 = Conv2D(16, (3, 3), activation='relu', padding='same')(u3)
    d3 = Dropout(0.2)(d3)
    d3 = Conv2D(16, (3, 3), activation='relu', padding='same')(d3)
    
    outputs = Conv2D(n_classes, (1, 1), activation='softmax')(d3)
    model = Model(inputs=[inputs], outputs=[outputs])
    return model
