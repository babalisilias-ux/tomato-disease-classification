# ============================================================
# Plant Disease Classification with Transfer Learning
# ============================================================
# Author: Ilias Bampalis
#
# Goal: Classify tomato leaf images as healthy or as having a
#       specific disease, using a pre-trained MobileNetV2
#       (transfer learning).
#
# This is written to run on Google Colab (free GPU).
# To use it:
#   1. Go to https://colab.research.google.com
#   2. New notebook -> paste each "CELL" below into its own cell
#   3. Runtime -> Change runtime type -> Hardware accelerator: GPU
#   4. Run cells top to bottom.
# ============================================================


# ============================================================
# CELL 1 — Install & import libraries
# ============================================================
# kagglehub downloads the dataset straight from Kaggle.
# TensorFlow/Keras is the deep-learning framework we use.

!pip install kagglehub --quiet

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

print("TensorFlow version:", tf.__version__)
print("GPU available:", tf.config.list_physical_devices('GPU'))
# If GPU shows an empty list, go to Runtime -> Change runtime type -> GPU.


# ============================================================
# CELL 2 — Download the dataset
# ============================================================
# The PlantVillage dataset has 38 classes across many plants.
# To keep this project small and fast, we will use ONLY tomato
# classes (still ~10 disease/healthy categories — plenty to show skill).

import kagglehub

# Downloads to a local cache folder and returns the path
dataset_path = kagglehub.dataset_download("abdallahalidev/plantvillage-dataset")
print("Dataset downloaded to:", dataset_path)

# Explore the folder structure to find the image directories
for root, dirs, files in os.walk(dataset_path):
    # only print the first couple of levels so we don't flood the output
    depth = root.replace(dataset_path, "").count(os.sep)
    if depth <= 2:
        print(root, "->", len(dirs), "subfolders,", len(files), "files")


# ============================================================
# CELL 3 — Point to the colored images & keep only tomato classes
# ============================================================
# The dataset has 'color', 'grayscale', and 'segmented' versions.
# We use the 'color' images (most realistic).

# NOTE: after running CELL 2, look at the printed folder structure
# and adjust this path if needed. It is usually:
#   <dataset_path>/plantvillage dataset/color
base_dir = os.path.join(dataset_path, "plantvillage dataset", "color")

# List all class folders
all_classes = sorted(os.listdir(base_dir))
print("Total classes in dataset:", len(all_classes))

# Keep only tomato classes
tomato_classes = [c for c in all_classes if c.lower().startswith("tomato")]
print("\nTomato classes we will use:")
for c in tomato_classes:
    n_images = len(os.listdir(os.path.join(base_dir, c)))
    print(f"  {c}: {n_images} images")


# ============================================================
# CELL 4 — Build train/validation datasets (tomato only)
# ============================================================
# We create a temporary folder with symlinks to only the tomato
# folders, then let Keras load images straight from disk.

import shutil

work_dir = "/content/tomato_data"
if os.path.exists(work_dir):
    shutil.rmtree(work_dir)
os.makedirs(work_dir)

for c in tomato_classes:
    src = os.path.join(base_dir, c)
    dst = os.path.join(work_dir, c)
    os.symlink(src, dst)

IMG_SIZE = (224, 224)   # MobileNetV2 expects 224x224 images
BATCH_SIZE = 32

# 80% training, 20% validation, split automatically
train_ds = tf.keras.utils.image_dataset_from_directory(
    work_dir,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    work_dir,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
)

class_names = train_ds.class_names
print("\nClass names:", class_names)
NUM_CLASSES = len(class_names)


# ============================================================
# CELL 5 — Look at a few example images
# ============================================================
# Always eyeball your data before training — it catches problems early.

plt.figure(figsize=(12, 8))
for images, labels in train_ds.take(1):
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]], fontsize=8)
        plt.axis("off")
plt.tight_layout()
plt.savefig("sample_images.png", dpi=120)
plt.show()


# ============================================================
# CELL 6 — Performance setup (speeds up training)
# ============================================================
# .cache() keeps images in memory; .prefetch() loads the next
# batch while the current one trains. Standard Keras best practice.

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


# ============================================================
# CELL 7 — Build the model (transfer learning)
# ============================================================
# We load MobileNetV2 pre-trained on ImageNet (1.4M images),
# WITHOUT its top classification layer (include_top=False).
# We freeze it (it already knows how to "see") and add our own
# small classifier on top for the tomato classes.

# Data augmentation: randomly flip/rotate images during training
# so the model generalizes better and doesn't just memorize.
data_augmentation = models.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

# Load the pre-trained base
base_model = MobileNetV2(
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights="imagenet",
)
base_model.trainable = False   # freeze — we don't retrain these layers

# Assemble the full model
inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
x = data_augmentation(inputs)
x = preprocess_input(x)                 # scale pixels the way MobileNet expects
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)  # turn feature maps into a vector
x = layers.Dropout(0.2)(x)              # dropout reduces overfitting
outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

model = models.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()


# ============================================================
# CELL 8 — Train the model
# ============================================================
# 5 epochs is enough for a quick, strong result with transfer learning.
# Each epoch = one full pass over the training data.

EPOCHS = 5
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
)


# ============================================================
# CELL 9 — Plot training curves
# ============================================================
# These plots show whether the model is learning and whether it
# overfits (train accuracy >> validation accuracy).

acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]
epochs_range = range(EPOCHS)

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label="Train")
plt.plot(epochs_range, val_acc, label="Validation")
plt.legend()
plt.title("Accuracy")
plt.xlabel("Epoch")

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label="Train")
plt.plot(epochs_range, val_loss, label="Validation")
plt.legend()
plt.title("Loss")
plt.xlabel("Epoch")
plt.tight_layout()
plt.savefig("training_curves.png", dpi=120)
plt.show()


# ============================================================
# CELL 10 — Evaluate: confusion matrix & classification report
# ============================================================
# This is the same kind of evaluation you did in your thesis
# (confusion matrix + per-class precision/recall/accuracy).

y_true = []
y_pred = []
for images, labels in val_ds:
    preds = model.predict(images, verbose=0)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(preds, axis=1))

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix — Tomato Disease Classification")
plt.xticks(rotation=45, ha="right", fontsize=7)
plt.yticks(fontsize=7)
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=120)
plt.show()

# Classification report (precision / recall / f1 per class)
print(classification_report(y_true, y_pred, target_names=class_names))

overall_acc = (y_true == y_pred).mean()
print(f"\nOverall validation accuracy: {overall_acc:.4f}")


# ============================================================
# CELL 11 — Save the trained model
# ============================================================
# Saves the model so you can reuse it or build a demo app later.

model.save("tomato_disease_model.keras")
print("Model saved as tomato_disease_model.keras")

# To download it to your computer (Colab only):
# from google.colab import files
# files.download("tomato_disease_model.keras")
