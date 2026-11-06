import json, pathlib, numpy as np, tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load model and class names
model = tf.keras.models.load_model('plant_disease_model.h5')
with open('class_labels.json') as f:
    class_names = json.load(f)

# Load validation data as a proxy for test data
val_dir = pathlib.Path('dataset/New Plant Diseases Dataset(Augmented)/valid')
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    directory=str(val_dir),
    label_mode='categorical',
    image_size=(224, 224),
    batch_size=32,
    shuffle=False
)

# Gather true labels
y_true_onehot = np.concatenate([y for _, y in valid_ds], axis=0)
y_true = np.argmax(y_true_onehot, axis=1)

# Predict
y_pred_probs = model.predict(valid_ds)
y_pred = np.argmax(y_pred_probs, axis=1)

# Overall accuracy
accuracy = np.mean(y_pred == y_true)
print(f"Overall Top-1 accuracy (validation set): {accuracy*100:.2f}%")

# Classification report
print("\nClassification Report:")
labels = np.arange(len(class_names))
print(classification_report(y_true, y_pred, labels=labels, target_names=class_names, digits=3))

# Confusion matrix (optional visual saved to file)
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(12,10))
heatmap = sns.heatmap(cm, annot=False, fmt='d', cmap='Blues')
plt.title('Confusion Matrix (validation)')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()
plt.savefig('confusion_matrix_validation.png')
print('\nConfusion matrix saved as confusion_matrix_validation.png')
