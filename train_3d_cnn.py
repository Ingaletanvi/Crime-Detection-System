import os
import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# === STEP 1: Load video clips as 3D tensors ===
def load_clips_from_directory(directory, label, clip_len=16, max_clips=50):
    X = []
    y = []
    count = 0
    for clip_folder in os.listdir(directory):
        if count >= max_clips:
            break
        clip_path = os.path.join(directory, clip_folder)
        frames = sorted(os.listdir(clip_path))
        if len(frames) < clip_len:
            continue
        clip = []
        for i in range(clip_len):
            img_path = os.path.join(clip_path, frames[i])
            img = cv2.imread(img_path)
            img = cv2.resize(img, (112, 112))  # Downscale for speed
            img = img.astype(np.float32) / 255.0
            clip.append(img)
        X.append(np.array(clip))
        y.append(label)
        count += 1
    return X, y

# === STEP 2: Define dataset paths ===
crime_dir = r"C:\Users\itanv\Desktop\CRIME youtube dataset - Copy\crime_detection_system\data\crime"
non_crime_dir = r"C:\Users\itanv\Desktop\CRIME youtube dataset - Copy\crime_detection_system\data\non_crime"

X_crime, y_crime = load_clips_from_directory(crime_dir, label=1, max_clips=70)
X_normal, y_normal = load_clips_from_directory(non_crime_dir, label=0, max_clips=70)

# === STEP 3: Preprocess ===
X = np.array(X_crime + X_normal)
y = np.array(y_crime + y_normal)
X = X.reshape(-1, 16, 112, 112, 3)
y = to_categorical(y, num_classes=2)

# === STEP 4: Train/test split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === STEP 5: Define simplified 3D CNN ===
model = Sequential([
    Conv3D(16, kernel_size=(3, 3, 3), activation='relu', input_shape=(16, 112, 112, 3)),
    MaxPooling3D(pool_size=(2, 2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(2, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# === STEP 6: Train ===
model.fit(X_train, y_train, epochs=5, batch_size=2, validation_data=(X_test, y_test))

# === STEP 7: Save model ===
model_path = r"C:\Users\itanv\Desktop\CRIME youtube dataset - Copy\crime_detection_system\models\3d_cnn_model.h5"
model.save(model_path)

print(f"✅ Training complete. Model saved to {model_path}")
