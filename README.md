**Petrographic-Image-Segmentation**
This project is aimed at segmenting petrographic images from rock thin sections in order to classify rock types. The initial objective is to classify 4 different minerals essential for sandstone classification in the Folk chart, plus the background:

Class 0: Background (Bkg)

Class 1: Feldspar (Fk)

Class 2: Lithic fragments (Lf)

Class 3: Plagioclase (Plg)

Class 4: Quartz (Qz)

To accomplish this, three main segmentation models are compared:

Enhanced U-Net (or “U-Net” in some scripts)
RL-Segmentation (U-Net variant with a reinforcement‐learning–inspired policy module at the bottleneck)

Transformer-Segmentation (U-Net variant that includes a Transformer block for long-range dependencies)

Previously, the project included DeepLab v3 comparisons, but the current code focuses on advanced U-Net–like approaches that utilize additional modules to handle the 
complexities of petrographic images under parallel and crossed polarized light.

**1. Data Overview**
Each rock thin section is imaged under:

Parallel Polarized Light (PPL) and Crossed Polarized Light (XPL)

These two images provide complementary information about the minerals. In this project, the PPL and XPL images are combined into a 6-channel input (3 channels each) so that the segmentation network can exploit both images at once.

Mask Annotation
The annotated mask for each sample uses integer labels (0–4) to represent the classes listed above. The code expects dummy-coded mask images (e.g., (0,0,0) for background, (1,1,1) for Feldspar, etc.) or otherwise an adapted rgb_to_2d_label function that maps the actual colors in the mask to integers 0–4.

**2. Patchification and Data Preparation**

Patch Extraction:

Each large thin-section image is “patchified” into smaller patches using the patchify library, typically at size 400×400 patches.
The same process is applied to the two input images (PPL, XPL) and the mask.
Combining Channels:

Once patches are extracted, the script creates a 6-channel image by concatenating the scaled PPL patch and scaled XPL patch along the channel dimension.
Mask Conversion:

A function rgb_to_2d_label maps the RGB-coded mask patches into integer label arrays for training.
Data Splitting:

The data is split into X_train, X_test, y_train, y_test using train_test_split (with test_size=0.20).
Each label patch is then converted to a one-hot encoding via to_categorical.

**3. Model Architectures**

**A. Enhanced U-Net**
A U-Net–like architecture that can handle 6 input channels. Additional augmentations often include attention gates, a transformer block, and other modifications in certain variants. The code references an “Enchanced_unet_model.py” file, which defines functions like multi_unet_model(...).

**B. RL-Segmentation**
Similar to U-Net, but includes a “policy module” at the bottleneck.
This module learns a policy vector (via global average pooling and a Dense “softmax” layer) that multiplicatively modulates the bottleneck feature maps.

**C. Transformer-Segmentatio**n
Inserts a Transformer encoder block at the bottleneck.
Flattens spatial dimensions, applies multi-head self-attention, and re-injects learned positional embeddings before reshaping back to the original 2D feature map.

**4. Training Script and Hyperparameter Tuning**
The main training script (which can be named something like Model_Training.py) demonstrates:

Data Loading and Patchification

Loss, Metrics, and Model Setup

Uses a combination of Dice Loss and Categorical Focal Loss to handle class imbalance.

Computes class weights via compute_class_weight.

Tracks metrics including overall accuracy and a custom Jaccard coefficient (jacard_coef).

Minimal Hyperparameter Search


The code includes loops over candidate batch sizes, learning rates, and epoch counts.

Each combination is tried for each of the three model types (U-Net, RL-Segmentation, Transformer-Segmentation).

The script keeps track of the best accuracy across runs.
Model weights and logs are saved in a models/ folder.

Visualizations
For each trained model, the script picks a random test sample, shows:
Parallel Light patch (first 3 channels),
Ground Truth segmentation,
Predicted segmentation.
The code also plots training/validation loss and training/validation accuracy over epochs.
Saving Figures

If you run on a headless environment, the recommended approach is to use plt.savefig(...) to store the plots on disk, then optionally call plt.show() if you have a GUI or notebook interface.

**5. Example Results**

Class Imbalance:
Many images show a low occurrence of Feldspar (Fk) and Plagioclase (Plg), leading to poor recall for these classes. The combination of Focal + Dice Loss and class weighting partially mitigates this issue.
Transformer Blocks can help capture global context. However, more tuning (learning rate, patch size, data augmentation) may be required to see improvements.
RL-Segmentation might help the network highlight subtle regions based on a learned policy vector, though it can require more specialized tuning of the reinforcement-inspired module.

**6. Future Work**

Additional Data Augmentation:
Consider advanced augmentations (random rotations, elastic deformations, etc.) to handle class imbalance or subtle morphological differences.

Further Architectural Tweaks:
You can try other attention mechanisms (e.g., Squeeze-and-Excitation blocks) or deeper transformer blocks to see if it improves segmentation.

Extend Beyond Sandstones:
Once the four minerals are segmented reliably, you can add more classes (e.g., porosity, cements, other diagenetic features).

**7. References**
U-Net Paper: Ronneberger, O. et al. (2015). “U-Net: Convolutional Networks for Biomedical Image Segmentation.” arXiv:1505.04597

Attention Is All You Need: Vaswani, A. et al. (2017). “Attention Is All You Need.” arXiv:1706.03762

DeepLab: Chen, L.-C. et al. (2017). “Rethinking Atrous Convolution for Semantic Image Segmentation.” arXiv:1706.05587
