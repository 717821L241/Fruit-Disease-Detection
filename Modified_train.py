import os
import pandas as pd
import cv2
import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from sklearn.neighbors import KNeighborsClassifier
import joblib

# Parameters
data_dir ="data"  # Update with your data directory
input_shape = (50, 50, 3)  # Image size
test_size = 0.1  

# Read the data with a limit of 100 images per category
subfolders = os.listdir(data_dir)
data = []
for cls in subfolders:
    cls_dir = os.path.join(data_dir, cls)
    images = os.listdir(cls_dir)[:100]  # Limit to 100 images per category
    for img_name in images:
        img_path = os.path.join(cls_dir, img_name)
        data.append((img_path, cls))

# Split into train and test sets
train_data, test_data = train_test_split(data, test_size=test_size, random_state=42)
train_df = pd.DataFrame(train_data, columns=["image_path", "class"])
test_df = pd.DataFrame(test_data, columns=["image_path", "class"])

# Image preprocessing function
def preprocess_image(img_path):
    try:
        img_array = cv2.imread(img_path, 1)
        img_array = cv2.medianBlur(img_array, 1)
        new_array = cv2.resize(img_array, input_shape[:2])
        return new_array
    except Exception as e:
        return None

# Create training data
training_data = []
for index, row in train_df.iterrows():
    img_path = row["image_path"]
    class_num = subfolders.index(row["class"])
    img_data = preprocess_image(img_path)
    if img_data is not None:
        training_data.append([img_data, class_num])

random.shuffle(training_data)

X = []  # Features
y = []  # Labels
for features, label in training_data:
    X.append(features)
    y.append(label)

X = np.array(X*2) / 255.0  # Normalize the data
y = np.array(y*2)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)

# CNN model for feature extraction
cnn_model = Sequential([
    Conv2D(32, (3, 3), input_shape=X.shape[1:]),
    Activation("relu"),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(64, (3, 3)),
    Activation("relu"),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(64, (3, 3)),
    Activation("relu"),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Flatten(),
    Dense(128, activation="relu"),
    Dense(128, activation="relu"),
    Dense(22, activation="softmax")  # Assuming 4 classes
])

cnn_model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train the CNN model
cnn_model.fit(X_train, y_train, batch_size=32, epochs=10, validation_split=0.2)
cnn_model.save('cnn_model.h5')

# Use CNN as a feature extractor
cnn_feature_extractor = Model(inputs=cnn_model.input, outputs=cnn_model.layers[-5].output)
X_train_cnn_features = cnn_feature_extractor.predict(X_train)
X_test_cnn_features = cnn_feature_extractor.predict(X_test)

# Flatten features for KNN input
X_train_cnn_features = X_train_cnn_features.reshape(X_train_cnn_features.shape[0], -1)
X_test_cnn_features = X_test_cnn_features.reshape(X_test_cnn_features.shape[0], -1)

# Train KNN on extracted features
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train_cnn_features, y_train)

# Save the KNN model
joblib.dump(knn_model, 'knn_model_cnn_features.pkl')

# Evaluate KNN model
y_pred = knn_model.predict(X_test_cnn_features)
accuracy = metrics.accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Classification report
print('\nClassification Report\n')
print(classification_report(y_test, y_pred))

# Confusion matrix
confusion_mtx = confusion_matrix(y_test, y_pred)

# Plot confusion matrix
plt.figure(figsize=(8, 8))
sns.heatmap(confusion_mtx, annot=True, fmt=".1f", cmap="Blues", linewidths=0.01, linecolor="gray")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()
