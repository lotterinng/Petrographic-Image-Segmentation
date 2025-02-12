import numpy as np
from matplotlib import pyplot as plt
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.saving import register_keras_serializable
from sklearn.preprocessing import MinMaxScaler
from smooth_tiled_predictions import predict_img_with_smooth_windowing
from petrographic_image_utils import rgb_to_2d_label

# Define and register the custom function
@register_keras_serializable()
def random_photometric_augmentation(x):
    # Dummy implementation: simply return the input.
    # Replace this with your actual augmentation logic if needed.
    return x

# Alternatively, if you do not wish to decorate,
# you can define the function and then pass it via custom_objects.
# def random_photometric_augmentation(x):
#     return x

scaler = MinMaxScaler()
patch_size = 400
n_classes = 5  # Number of classes for segmentation

# Define image and mask paths
image_p_path = 'petrographic_image_dataset/paralelos/8a_paralelos.jpg'
image_c_path = 'petrographic_image_dataset/cruzados/8a_cruzados.jpg'
mask_path = 'petrographic_image_dataset/masks/8a_mascara.ome.tiff'

# Read images and mask
image_p = cv2.imread(image_p_path, 1)           # Read parallel nicols image as BGR
image_c = cv2.imread(image_c_path, 1)             # Read crossed nicols image as BGR
mask = cv2.imread(mask_path, 1)                   # Read mask
mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
mask = rgb_to_2d_label(mask)

# Load model using raw string for the Windows path.
model_path = r"C:\Users\Lottering\OneDrive\Documents\Petrographic-Image-Segmentation\models\Transformer-Segmentation.keras"
# Pass the custom function via custom_objects so that Keras can locate it.
model = load_model(model_path, custom_objects={'random_photometric_augmentation': random_photometric_augmentation}, compile=False)

###################################################################################
# Predict using smooth blending
input_img_p = scaler.fit_transform(image_p.reshape(-1, image_p.shape[-1])).reshape(image_p.shape)
input_img_c = scaler.fit_transform(image_c.reshape(-1, image_c.shape[-1])).reshape(image_c.shape)
input_img = np.dstack((input_img_p, input_img_c))

predictions_smooth = predict_img_with_smooth_windowing(
    input_img,
    window_size=patch_size,
    subdivisions=2,  # Minimal amount of overlap for windowing. Must be an even number.
    nb_classes=n_classes,
    pred_func=lambda img_batch_subdiv: model.predict(img_batch_subdiv)
)

final_prediction = np.argmax(predictions_smooth, axis=2)

########################
# Plot and save results
plt.figure(figsize=(12, 12))
plt.subplot(221)
plt.title('Image Plain Polarized Light')
plt.imshow(image_p)
plt.subplot(222)
plt.title('Image Cross Polarized Light')
plt.imshow(image_c)
plt.subplot(223)
plt.title('Testing Label')
plt.imshow(mask)
plt.subplot(224)
plt.title('Transformer Prediction')
plt.imshow(final_prediction)
plt.show()
#############################
