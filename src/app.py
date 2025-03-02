from flask import Flask, request, jsonify, send_from_directory
import tensorflow as tf
from flask_cors import CORS, cross_origin
from PIL import Image
import numpy as np
import requests
from io import BytesIO
import io
import cv2
import os

app = Flask(__name__)
CORS(app) # Enable CORS
model = tf.keras.models.load_model('oort_captcha_model.keras')

CLASS_NAMES = ['crosswalk', 'speedlimit', 'stop', 'trafficlight']  # Update with your class names

def preprocess_image(image):
    image = image.convert('RGB')  # Ensure image is in RGB format
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    return np.expand_dims(image, axis=0)

import base64

def split_image(image, grid_size=(3, 3)):
    h, w, _ = image.shape
    grid_h, grid_w = h // grid_size[0], w // grid_size[1]
    grid_images = []
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            grid_image = image[i*grid_h:(i+1)*grid_h, j*grid_w:(j+1)*grid_w]
            _, buffer = cv2.imencode('.png', grid_image)
            grid_image_base64 = base64.b64encode(buffer).decode('utf-8')
            grid_images.append(grid_image_base64)
    return grid_images

@app.route('/generate_captcha', methods=['GET'])
def generate_captcha():
    # Programmatically select an image from an online source
    image_url = 'https://oortcaptcha.standard.us-east-1.oortstorage.com/road13.png'
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content))
    image = np.array(image)
    grid_images = split_image(image)
    return jsonify({'grid_images': grid_images})


@app.route('/images/<filename>')
def get_image(filename):
    return send_from_directory('data/images', filename)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    file = request.files['file']
    image = Image.open(io.BytesIO(file.read()))
    image = preprocess_image(image)
    prediction = model.predict(image)
    predicted_class = CLASS_NAMES[np.argmax(prediction[0])]
    result = {'prediction': predicted_class}
    return jsonify(result)

@app.route('/verify_captcha', methods=['POST'])
@cross_origin()  # Enable CORS for this route
def verify_captcha():
    data = request.get_json()
    selected_images = data['selectedImages']
    correct_selections = 0

    for filename in selected_images:
        image_path = os.path.join('data/images', filename)
        image = Image.open(image_path)
        image = preprocess_image(image)
        prediction = model.predict(image)
        predicted_class = CLASS_NAMES[np.argmax(prediction[0])]
        if predicted_class == 'trafficlight':  # Change this to the class you want to verify
            correct_selections += 1

    result = 'Correct' if correct_selections == len(selected_images) else 'Incorrect'
    return jsonify({'result': result})

from web3 import Web3

@app.route('/verify_signature', methods=['POST'])
def verify_signature():
    data = request.get_json()
    message = data['message']
    signature = data['signature']
    wallet_address = data['walletAddress']

    web3 = Web3(Web3.HTTPProvider('YOUR_INFURA_OR_ALCHEMY_URL'))
    message_hash = web3.sha3(text=message)
    recovered_address = web3.eth.account.recoverHash(message_hash, signature=signature)

    if recovered_address.lower() == wallet_address.lower():
        # Proceed with funds transfer
        return jsonify({'result': 'Signature verified'})
    else:
        return jsonify({'error': 'Invalid signature'}), 400

if __name__ == '__main__':
    app.run(debug=True)