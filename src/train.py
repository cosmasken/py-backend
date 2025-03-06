import os
import numpy as np
from model import create_model
from utils import load_data, preprocess_data, split_data
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report

CLASS_NAMES = ['crosswalk', 'speedlimit', 'stop', 'trafficlight']  # Update with your class names

def train_model(images_path, csv_path, epochs=20, batch_size=32):
    images, labels = load_data(images_path, csv_path)
    images = preprocess_data(images)

    train_images, val_images, train_labels, val_labels = split_data(images, labels, train_size=0.8)

    model = create_model(input_shape=(224, 224, 3), num_classes=4)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    history = model.fit(train_images, train_labels, epochs=epochs, batch_size=batch_size, validation_data=(val_images, val_labels), callbacks=[early_stopping])

    # Save the trained model in native Keras format
    model.save('oort_captcha_model.keras')

    # Evaluate the model
    predictions = model.predict(val_images)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(val_labels, axis=1)
    print(classification_report(true_classes, predicted_classes, target_names=CLASS_NAMES))

if __name__ == "__main__":
    images_path = 'data/images'
    csv_path = 'data/annotations.csv'
    train_model(images_path, csv_path)