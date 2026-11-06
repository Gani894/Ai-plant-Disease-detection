import pathlib
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------------------------
# Load the trained model
# -------------------------------------------------
model = tf.keras.models.load_model('plant_disease_model.h5')

# -------------------------------------------------
# Load the validation set (the directory contains sub‑folders for each class)
# -------------------------------------------------
val_dir = pathlib.Path('dataset/New Plant Diseases Dataset(Augmented)/valid')
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    directory=str(val_dir),
    label_mode='categorical',   # one‑hot vectors – matches model training
    image_size=(224, 224),
    batch_size=32,
    shuffle=False
)

# Use the class names that the dataset actually provides (ensures alignment)
class_names = val_ds.class_names

# -------------------------------------------------
# Gather true labels
# -------------------------------------------------
# `val_ds` yields (images, one-hot labels) tuples
true_onehot = np.concatenate([y for _, y in val_ds], axis=0)
y_true = np.argmax(true_onehot, axis=1)

# -------------------------------------------------
# Model predictions
# -------------------------------------------------
pred_probs = model.predict(val_ds)
y_pred = np.argmax(pred_probs, axis=1)

# -------------------------------------------------
# Overall Top‑1 accuracy
# -------------------------------------------------
accuracy = np.mean(y_pred == y_true)
print(f"Overall Top‑1 accuracy (validation set): {accuracy * 100:.2f}%")

# -------------------------------------------------
# Detailed classification report
# -------------------------------------------------
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names, digits=3))

# -------------------------------------------------
# Confusion matrix (saved as an image)
# -------------------------------------------------
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(12, 10))
sns = sns.heatmap(cm, annot=False, fmt='d', cmap='Blues')
plt.title('Confusion Matrix (validation)')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()
plt.savefig('confusion_matrix_validation.png')
print("\nConfusion matrix saved as confusion_matrix_validation.png")
