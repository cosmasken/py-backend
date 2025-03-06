import os
import pandas as pd
from PIL import Image
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

def load_images(image_dir, annotations):
    images = []
    filenames = [ann['filename'] for ann in annotations]
    for filename in filenames:
        img_path = os.path.join(image_dir, filename)
        if os.path.exists(img_path):
            image = Image.open(img_path).convert('RGB') 
            image = image.resize((224, 224))
            image = np.array(image)
            images.append(image)
    return np.array(images)

def load_annotations_csv(csv_path):
    df = pd.read_csv(csv_path)
    annotations = []
    for _, row in df.iterrows():
        annotation = {
            'filename': row['filename'],
            'width': row['width'],
            'height': row['height'],
            'class': row['class'],
            'bndbox': {
                'xmin': row['xmin'],
                'ymin': row['ymin'],
                'xmax': row['xmax'],
                'ymax': row['ymax']
            }
        }
        annotations.append(annotation)
    return annotations

def preprocess_data(images):

    images = images / 255.0
    return images

def load_data(image_dir, csv_path):
    annotations = load_annotations_csv(csv_path)
    images = load_images(image_dir, annotations)
    
    class_mapping = {
        'crosswalk': 0,
        'speedlimit': 1,
        'stop': 2,
        'trafficlight': 3
    }
    
    labels = [class_mapping[ann['class']] for ann in annotations if os.path.exists(os.path.join(image_dir, ann['filename']))]
    

    labels = to_categorical(labels, num_classes=4)
    
    return images, np.array(labels)

def split_data(images, labels, train_size=0.8):
    return train_test_split(images, labels, train_size=train_size)