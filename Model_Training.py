import os
import cv2
import random
import numpy as np
from matplotlib import pyplot as plt
from patchify import patchify
from petrographic_image_utils import image_crop, process_img_patch, rgb_to_2d_label
import segmentation_models as sm
sm.set_framework('tf.keras')
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import compute_class_weight
from sklearn.model_selection import train_test_split
from skimage import exposure
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import CSVLogger, ReduceLROnPlateau
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam, SGD, RMSprop

# Import the original U-Net and its custom metric
from Enchanced_unet_model import multi_unet_model, jacard_coef
# Import new models
from reinforce_transformer_model import rl_segmentation_model, transformer_segmentation_model

##############################################
# 1. Data Preparation
##############################################
scaler = MinMaxScaler()
patch_size = 400
n_classes = 5  # Initial number of classes; will be redefined after label extraction
root_directory = 'petrographic_image_dataset/'
image_dataset = []

# Process parallel and crossed images
for path, subdirs, files in os.walk(root_directory):
    dirname = os.path.basename(path)
    if dirname == 'paralelos':
        images = os.listdir(path)
        for image_name in images:
            if image_name.endswith(".jpg"):
                path_paralelos = os.path.join(path, image_name)
                path_cruzados = os.path.join(root_directory, "cruzados", image_name.split('_')[0] + "_cruzados.jpg")
                image_p = cv2.imread(path_paralelos, 1)
                image_c = cv2.imread(path_cruzados, 1)
                image_p = exposure.equalize_hist(image_p)
                image_c = exposure.equalize_hist(image_c)
                print("Now patchifying image:", path_paralelos)
                patches_p = patchify(image_p, (patch_size, patch_size, 3), step=patch_size)
                print("Now patchifying image:", path_cruzados)
                patches_c = patchify(image_c, (patch_size, patch_size, 3), step=patch_size)
                for j in range(patches_p.shape[0]):
                    for k in range(patches_p.shape[1]):
                        single_patch_p = process_img_patch(patches_p[j, k, :, :], scaler)
                        single_patch_c = process_img_patch(patches_c[j, k, :, :], scaler)
                        stacked = np.dstack((single_patch_p, single_patch_c))
                        image_dataset.append(stacked)

# Process masks
mask_dataset = []
for path, subdirs, files in os.walk(root_directory):
    dirname = os.path.basename(path)
    if dirname == 'masks':
        masks = os.listdir(path)
        for mask_name in masks:
            if mask_name.endswith(".tiff"):
                mask = cv2.imread(os.path.join(path, mask_name), 1)
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
                print("Now patchifying mask:", os.path.join(path, mask_name))
                patches_mask = patchify(mask, (patch_size, patch_size, 3), step=patch_size)
                for j in range(patches_mask.shape[0]):
                    for k in range(patches_mask.shape[1]):
                        single_patch_mask = patches_mask[j, k, :, :][0]  # Remove extra dimension
                        mask_dataset.append(single_patch_mask)

image_dataset = np.array(image_dataset)
mask_dataset = np.array(mask_dataset)

# Convert masks to label maps
labels = []
for i in range(mask_dataset.shape[0]):
    label = rgb_to_2d_label(mask_dataset[i])
    labels.append(label)
labels = np.array(labels)
print("Unique labels in label dataset are: ", np.unique(labels))

# Sanity check: visualize a sample image and its mask
sample_idx = random.randint(0, len(image_dataset) - 1)
plt.figure(figsize=(12, 4))
plt.subplot(131)
plt.imshow(image_dataset[sample_idx, :, :, 0:3])
plt.title("Parallel Polarized Light (PPL)")
plt.subplot(132)
plt.imshow(image_dataset[sample_idx, :, :, 3:7])
plt.title("Crossed Polarized Light (XPL)")
plt.subplot(133)
plt.imshow(labels[sample_idx])
plt.title("Mask Labels")
plt.show()

##############################################
# 2. Data Splitting and Preprocessing
##############################################
n_classes = len(np.unique(labels))
labels_cat = to_categorical(labels, num_classes=n_classes)
X_train, X_test, y_train, y_test = train_test_split(image_dataset, labels_cat, test_size=0.20, random_state=42)

##############################################
# 3. Loss, Metrics, and Optimizer Setup
##############################################
weights = compute_class_weight(class_weight='balanced',
                               classes=np.unique(np.ravel(labels, order='C')),
                               y=np.ravel(labels, order='C'))
dice_loss = sm.losses.DiceLoss(class_weights=weights)
focal_loss = sm.losses.CategoricalFocalLoss()
total_loss = (dice_loss * 0.5) + (focal_loss * 0.5)
metrics = ['accuracy', jacard_coef]

IMG_HEIGHT = X_train.shape[1]
IMG_WIDTH = X_train.shape[2]
IMG_CHANNELS = X_train.shape[3]

model_folder = 'models/'
if not os.path.exists(model_folder):
    os.makedirs(model_folder)
batch_size = 16
learning_rate = 0.001
epochs = 100
optimizer = Adam(learning_rate=learning_rate)

print("=== MODEL INFORMATION ===")
print("Loss Weights: ", weights)
print("Batch Size: ", batch_size)
print("Epochs: ", epochs)
print("Learning Rate: ", learning_rate)
print("Image Dimensions (HxW): ", IMG_HEIGHT, 'x', IMG_WIDTH)
print("Channels: ", IMG_CHANNELS)
print("=========================")

##############################################
# 4. Model Comparison: Training, Evaluation, and Visualization
##############################################
model_constructors = {
    'U-Net': lambda: multi_unet_model(n_classes=n_classes, IMG_HEIGHT=IMG_HEIGHT, IMG_WIDTH=IMG_WIDTH, IMG_CHANNELS=IMG_CHANNELS),
    'RL-Segmentation': lambda: rl_segmentation_model(n_classes=n_classes, IMG_HEIGHT=IMG_HEIGHT, IMG_WIDTH=IMG_WIDTH, IMG_CHANNELS=IMG_CHANNELS),
    'Transformer-Segmentation': lambda: transformer_segmentation_model(n_classes=n_classes, IMG_HEIGHT=IMG_HEIGHT, IMG_WIDTH=IMG_WIDTH, IMG_CHANNELS=IMG_CHANNELS)
}

model_results = {}   # To store evaluation metrics for each model
histories = {}       # To store training histories for plotting
trained_models = {}  # To store trained model instances

for model_name, constructor in model_constructors.items():
    print("===================================")
    print("Training model:", model_name)
    print("===================================")
    current_optimizer = Adam(learning_rate=learning_rate)
    
    model = constructor()
    model.compile(optimizer=current_optimizer, loss=total_loss, metrics=metrics)
    model.summary()
    
    csv_logger = CSVLogger(os.path.join(model_folder, model_name + '.log'), append=True)
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        verbose=1,
        min_lr=1e-6
    )


    history = model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        validation_data=(X_test, y_test),
        shuffle=True,
        callbacks=[csv_logger, reduce_lr]
    )
    histories[model_name] = history
    trained_models[model_name] = model
    
    evaluation = model.evaluate(X_test, y_test, verbose=0)
    model_results[model_name] = dict(zip(model.metrics_names, evaluation))
    
    # Save the trained model in the native Keras format
    model_path = os.path.join(model_folder, model_name + '.keras')
    if os.path.exists(model_path):
        os.remove(model_path)
    model.save(model_path)
    
    sample_idx = random.randint(0, X_test.shape[0] - 1)
    sample_input = np.expand_dims(X_test[sample_idx], axis=0)
    prediction = model.predict(sample_input)
    
    plt.figure(figsize=(12, 4))
    plt.subplot(131)
    plt.imshow(X_test[sample_idx, :, :, 0:3])
    plt.title("Input (Parallel)")
    plt.subplot(132)
    plt.imshow(np.argmax(y_test[sample_idx], axis=-1))
    plt.title("Ground Truth")
    plt.subplot(133)
    plt.imshow(np.argmax(prediction[0], axis=-1))
    plt.title(model_name + " Prediction")
    plt.show()

##############################################
# Additional Evaluation Functions: Mean IoU and Dice Score
##############################################
def compute_iou(y_true, y_pred, num_classes):
    ious = []
    for cls in range(num_classes):
        true_cls = (y_true == cls)
        pred_cls = (y_pred == cls)
        intersection = np.logical_and(true_cls, pred_cls).sum()
        union = np.logical_or(true_cls, pred_cls).sum()
        if union == 0:
            iou = float('nan')
        else:
            iou = intersection / union
        ious.append(iou)
    return np.nanmean(ious), ious

def compute_dice(y_true, y_pred, num_classes):
    dices = []
    for cls in range(num_classes):
        true_cls = (y_true == cls)
        pred_cls = (y_pred == cls)
        intersection = np.logical_and(true_cls, pred_cls).sum()
        dice = (2 * intersection) / (true_cls.sum() + pred_cls.sum() + 1e-6)
        dices.append(dice)
    return np.mean(dices), dices

##############################################
# 5. Loop Through Each Model to Display Detailed Outputs
##############################################
class_names = [f'Class{i}' for i in range(n_classes)]

for model_name, model in trained_models.items():
    print(f"\n==== Detailed Evaluation for {model_name} ====")
    
    # Plot a sample prediction
    sample_idx = random.randint(0, X_test.shape[0] - 1)
    sample_input = np.expand_dims(X_test[sample_idx], axis=0)
    prediction = model.predict(sample_input)
    
    plt.figure(figsize=(12, 4))
    plt.subplot(131)
    plt.imshow(X_test[sample_idx, :, :, 0:3])
    plt.title("Input (Parallel)")
    plt.subplot(132)
    plt.imshow(np.argmax(y_test[sample_idx], axis=-1))
    plt.title("Ground Truth")
    plt.subplot(133)
    plt.imshow(np.argmax(prediction[0], axis=-1))
    plt.title(model_name + " Prediction")
    plt.show()
    
    from sklearn.metrics import confusion_matrix, classification_report
    def evaluate_segmentation_performance(y_true, y_pred, class_names):
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
        cm = confusion_matrix(y_true_flat, y_pred_flat, labels=range(len(class_names)))
        report = classification_report(y_true_flat, y_pred_flat, target_names=class_names)
        return cm, report

    y_test_labels = np.argmax(y_test, axis=-1)
    y_pred_labels = np.argmax(model.predict(X_test), axis=-1)
    cm, report = evaluate_segmentation_performance(y_test_labels, y_pred_labels, class_names)
    print(f"Confusion Matrix for {model_name}:\n", cm)
    print(f"Classification Report for {model_name}:\n", report)
    
    mean_iou, per_class_iou = compute_iou(y_test_labels, y_pred_labels, n_classes)
    mean_dice, per_class_dice = compute_dice(y_test_labels, y_pred_labels, n_classes)
    print(f"Mean IoU for {model_name}: {mean_iou}")
    print(f"Per-Class IoU for {model_name}: {per_class_iou}")
    print(f"Mean Dice Score for {model_name}: {mean_dice}")
    print(f"Per-Class Dice Score for {model_name}: {per_class_dice}")
    
    history = histories[model_name]
    
    plt.figure(figsize=(8,6))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f"{model_name} Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
    
    plt.figure(figsize=(8,6))
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f"{model_name} Accuracy over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()
