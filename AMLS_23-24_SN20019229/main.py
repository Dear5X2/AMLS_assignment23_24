import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
import random

model_A = tf.keras.models.load_model("PneumoniaMNIST.h5")
model_B = tf.keras.models.load_model("PathMNIST.h5")

data_A = np.load('pneumoniamnist.npz')
test_A = data_A['test_images']
data_B = np.load('pathminist.npz')
test_B = data_B['test_images']

def resize_images(images, new_size=(28,28)):
    resized_images = []
    for idx in images:
        resized = cv2.resize(idx, new_size)
        resized_images.append(resized)
    return resized_images

def convert_to_rgb(images):
    converted_images = []
    for idx in images:
        if idx.shape[-1] == 3:
            converted_images.append(idx)
        else:
            converted_idx = np.stack([idx]*3, axis = -1)
            converted_images.append(converted_idx)
    return converted_images

def normalize_images(images):
    normalized_images = []
    for idx in images:
        idx = idx.astype('float32')
        idx /= 255.0
        normalized_images.append(idx)
    return normalized_images

def get_random_image(data):
    random_index = random.randint(0, len(data) - 1)
    return random_index
  
A_Test = normalize_images(convert_to_rgb(resize_images(test_A,(28,28))))
predictions = model_A.predict(X_test_preprocessed)

# Convert predictions to label indices
predicted_labels = np.argmax(predictions, axis=1)

# Convert y_test back to label indices if it's in categorical format
true_labels = np.argmax(y_test, axis=1)

# Calculate the accuracy
accuracy = np.mean(predicted_labels == true_labels)
print(f"Accuracy on test set: {accuracy * 100:.2f}%")

# Optional: Display some predictions with their true labels
for i in range(10):  # Display first 10 comparisons
    predicted_str = vectorized_label_converter(predicted_labels[i])
    true_str = vectorized_label_converter(true_labels[i])
    print(f"Test image {i}: Predicted label - {predicted_str}, True label - {true_str}")
  from sklearn.metrics import classification_report

# Convert y_test back to label indices if it's in categorical format
true_labels = np.argmax(y_test, axis=1)

# Use the predicted_labels from the previous steps

# Generate the classification report
report = classification_report(true_labels, predicted_labels, target_names=['Normal', 'Pneumonia'])

print("Classification Report:")
print(report)

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Assuming true_labels and predicted_labels are already defined
cm = confusion_matrix(true_labels, predicted_labels)

# Print the raw confusion matrix to check the values
print("Raw Confusion Matrix:")
print(cm)

# Convert confusion matrix to logarithmic scale to handle wide value ranges
cm_log_scale = np.log1p(cm)  # log1p is used to handle zero values in the matrix

# Plotting the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm
            , fmt='d', cmap='Blues', xticklabels=['Normal', 'Pneumonia'], yticklabels=['Normal', 'Pneumonia'])
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
# Annotate each cell with the actual values from 'cm'
for i in range(cm_log_scale.shape[0]):
    for j in range(cm_log_scale.shape[1]):
        plt.text(j+0.5, i+0.5, cm[i, j], 
                 fontdict={'fontsize':8, 'weight':'bold', 'color':'black'},
                 ha='center', va='center')

plt.show()


if __name__ == "__main__
