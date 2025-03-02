import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNet

def create_model(input_shape=(224, 224, 3), num_classes=2):
    base_model = MobileNet(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False  # Freeze the base model

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])

    return model