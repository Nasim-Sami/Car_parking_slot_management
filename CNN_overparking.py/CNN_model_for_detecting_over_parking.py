"""
Parking classification model trainer (grayscale)
------------------------------------------------
1. Splits dataset into train/test with randomness.
2. Trains a CNN model on grayscale images.
3. Saves model as parking_model.h5
"""

import os
import shutil
import random
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

# ---------------------
# Config
# ---------------------
BASE_DIR = "/Users/sami/Documents/HECKATHON/Car_PArking_project_Academic/Wrong_parking_Detection_by_perspective_correction/Dataset_parking"  # contains good_parking/ and over_parking/
OUTPUT_MODEL = "parking_model.h5"
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 25
TEST_SIZE = 0.2  # 80% train, 20% test
SEED = None      # random every run

# ---------------------
# Prepare dataset
# ---------------------
classes = ["good_parking", "over_parking"]

# Temporary folders
train_dir = os.path.join(BASE_DIR, "train")
test_dir = os.path.join(BASE_DIR, "test")

# Clean old splits
if os.path.exists(train_dir):
    shutil.rmtree(train_dir)
if os.path.exists(test_dir):
    shutil.rmtree(test_dir)

for split in ["train", "test"]:
    for cls in classes:
        os.makedirs(os.path.join(BASE_DIR, split, cls), exist_ok=True)

# Split & copy images
for cls in classes:
    src_folder = os.path.join(BASE_DIR, cls)
    images = [f for f in os.listdir(src_folder) if f.lower().endswith((".jpg", ".png", ".jpeg"))]

    train_files, test_files = train_test_split(
        images, test_size=TEST_SIZE, random_state=SEED
    )

    for fname in train_files:
        shutil.copy(os.path.join(src_folder, fname), os.path.join(train_dir, cls, fname))
    for fname in test_files:
        shutil.copy(os.path.join(src_folder, fname), os.path.join(test_dir, cls, fname))

print("✅ Dataset split done!")

# ---------------------
# Load datasets (grayscale)
# ---------------------
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="grayscale",   # <--- Force grayscale
    label_mode="int"
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="grayscale",   # <--- Force grayscale
    label_mode="int"
)

# Normalize
normalization_layer = layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))

# Prefetch
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(100).prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

# ---------------------
# CNN model (grayscale input)
# ---------------------
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation="relu", input_shape=(128, 128, 1)),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(len(classes), activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ---------------------
# Train
# ---------------------
history = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=EPOCHS
)

# ---------------------
# Save
# ---------------------
model.save(OUTPUT_MODEL)
print(f"✅ Model saved as {OUTPUT_MODEL}")
