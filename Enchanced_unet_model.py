"""
Enhanced U-Net with Photometric Augmentation, Attention Gates, and a Transformer Block at the Bottleneck.
This model is designed for advanced segmentation research and retains the original U-Net structure,
with the following modifications:
  - Photometric augmentations (random brightness, contrast, and saturation) applied via a Lambda layer.
  - A recommendation for patch selection (oversampling rare-class patches) to be handled in the data pipeline.
  - Attention gates integrated into skip connections (Attention U-Net style) to focus on salient features.
  - A transformer block inserted at the bottleneck to capture long-range dependencies.
"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, concatenate, 
                                     Conv2DTranspose, Dropout, Lambda, Activation, Add, Multiply, UpSampling2D)
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras import backend as K

# --- Define the custom Jaccard coefficient metric ---
def jacard_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1.0)

##############################################
# 1. Photometric Augmentation Function
##############################################
def random_photometric_augmentation(x):
    """
    Randomly adjust brightness, contrast, and saturation.
    If the input has 6 channels (e.g. from stacking two 3-channel images),
    split the tensor into two 3-channel tensors, apply augmentation independently,
    and then concatenate the results.
    (In practice, you may combine this with CLAHE-based histogram equalization externally.)
    """
    channels = x.shape[-1]
    if channels == 3:
        x = tf.image.random_brightness(x, max_delta=0.1)
        x = tf.image.random_contrast(x, lower=0.9, upper=1.1)
        x = tf.image.random_saturation(x, lower=0.9, upper=1.1)
    elif channels == 6:
        # Split the tensor into two parts along the channel axis
        x1, x2 = tf.split(x, num_or_size_splits=2, axis=-1)
        x1 = tf.image.random_brightness(x1, max_delta=0.1)
        x1 = tf.image.random_contrast(x1, lower=0.9, upper=1.1)
        x1 = tf.image.random_saturation(x1, lower=0.9, upper=1.1)
        x2 = tf.image.random_brightness(x2, max_delta=0.1)
        x2 = tf.image.random_contrast(x2, lower=0.9, upper=1.1)
        x2 = tf.image.random_saturation(x2, lower=0.9, upper=1.1)
        x = tf.concat([x1, x2], axis=-1)
    else:
        # If the number of channels is not 3 or 6, you can choose to skip augmentation or handle accordingly.
        pass
    return x

##############################################
# 2. Attention Gate for Skip Connections
##############################################
def attention_gate(x, g, inter_channels):
    """
    Compute an attention coefficient for the skip connection 'x' using the gating signal 'g'.
    If the spatial dimensions of x and g are equal, no downsampling is applied.
    Otherwise, x is downsampled to match g.
    
    Parameters:
      x: Tensor from the encoder (skip connection).
      g: Gating tensor from the decoder.
      inter_channels: Number of filters for intermediate computations (typically half of x's channels).
      
    Returns:
      y: The gated output with the same shape as x.
    """
    # Check if spatial dimensions match
    if x.shape[1] == g.shape[1] and x.shape[2] == g.shape[2]:
        # No need to downsample x since sizes already match
        theta_x = Conv2D(inter_channels, (1, 1), padding='same')(x)
        phi_g = Conv2D(inter_channels, (1, 1), padding='same')(g)
        add_xg = Add()([theta_x, phi_g])
        act_xg = Activation('relu')(add_xg)
        psi = Conv2D(1, (1, 1), padding='same')(act_xg)
        sigmoid_xg = Activation('sigmoid')(psi)
        y = Multiply()([x, sigmoid_xg])
    else:
        # Downsample x so that it matches g's spatial dimensions
        theta_x = Conv2D(inter_channels, (2, 2), strides=(2, 2), padding='same')(x)
        phi_g = Conv2D(inter_channels, (1, 1), padding='same')(g)
        add_xg = Add()([theta_x, phi_g])
        act_xg = Activation('relu')(add_xg)
        psi = Conv2D(1, (1, 1), padding='same')(act_xg)
        sigmoid_xg = Activation('sigmoid')(psi)
        upsampled_psi = UpSampling2D(size=(2, 2), interpolation='bilinear')(sigmoid_xg)
        y = Multiply()([x, upsampled_psi])
    return y

##############################################
# 3. Custom Transformer Block Layer (defined above)
##############################################
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, num_heads=4, key_dim=64, ff_dim=256, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.ff_dim = ff_dim
        self.attn = tf.keras.layers.MultiHeadAttention(num_heads=self.num_heads, key_dim=self.key_dim, dropout=0.1)
        self.add1 = Add()
        self.dense_ffn1 = tf.keras.layers.Dense(self.ff_dim, activation='relu')
        self.add2 = Add()
    
    def build(self, input_shape):
        C = input_shape[-1]
        self.dense_ffn2 = tf.keras.layers.Dense(C)
        super(TransformerBlock, self).build(input_shape)
    
    def call(self, x):
        # Use dynamic shape for spatial dimensions
        shape = tf.shape(x)
        H = shape[1]
        W = shape[2]
        C = x.shape[-1]  # static if available
        num_tokens = H * W
        x_reshaped = tf.reshape(x, (-1, num_tokens, C))
        attn_output = self.attn(x_reshaped, x_reshaped)
        x_res = self.add1([x_reshaped, attn_output])
        ffn_output = self.dense_ffn1(x_res)
        ffn_output = self.dense_ffn2(ffn_output)
        x_trans = self.add2([x_res, ffn_output])
        x_trans = tf.reshape(x_trans, (-1, H, W, C))
        return x_trans

##############################################
# 4. Enhanced U-Net Model Definition
##############################################
def multi_unet_model(n_classes=4, IMG_HEIGHT=256, IMG_WIDTH=256, IMG_CHANNELS=1):
    # Input layer
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    
    # Photometric augmentation applied at the model level (for training).
    # In practice, additional CLAHE and other histogram equalization may be applied externally.
    augmented = Lambda(random_photometric_augmentation)(inputs)
    s = augmented  # Use augmented images as input
    
    # Note: Patch selection (oversampling patches containing underrepresented classes)
    # should be handled in the data preprocessing pipeline.
    
    # Contraction path (Encoder)
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
    c1 = Dropout(0.2)(c1)
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = Dropout(0.2)(c2)
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)

    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
    
    # Apply a transformer block at the bottleneck for long-range dependencies using the custom layer
    c5 = TransformerBlock(num_heads=4, key_dim=64, ff_dim=256)(c5)

    # Expansive path (Decoder) with Attention Gates
    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    attn_c4 = attention_gate(c4, u6, inter_channels=64)
    u6 = concatenate([u6, attn_c4])
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    attn_c3 = attention_gate(c3, u7, inter_channels=32)
    u7 = concatenate([u7, attn_c3])
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    attn_c2 = attention_gate(c2, u8, inter_channels=16)
    u8 = concatenate([u8, attn_c2])
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = Dropout(0.2)(c8)
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    attn_c1 = attention_gate(c1, u9, inter_channels=8)
    u9 = concatenate([u9, attn_c1], axis=3)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = Dropout(0.2)(c9)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

    outputs = Conv2D(n_classes, (1, 1), activation='softmax')(c9)

    model = Model(inputs=[inputs], outputs=[outputs])
    return model
