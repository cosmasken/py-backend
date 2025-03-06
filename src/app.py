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
import random
import base64
from environment import CustomEnv
from agent import DQNAgent

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

model = tf.keras.models.load_model('oort_captcha_model.keras')

CLASS_NAMES = ['crosswalk', 'speedlimit', 'stop', 'trafficlight']  # Update with your class names

def preprocess_image(image):
    image = image.convert('RGB')  # Ensure image is in RGB format
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    return np.expand_dims(image, axis=0)

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
@cross_origin()  # Enable CORS for this route
def generate_captcha():
    try:
        # Programmatically select an image from an online source
        image_index = random.randint(0, 786)
        image_url = f'https://oortbucket.standard.us-east-1.oortstorage.com/road{image_index}.png'
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content))
        image = np.array(image)
        grid_images = split_image(image)
        return jsonify({'grid_images': grid_images})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/images/<filename>')
@cross_origin()  # Enable CORS for this route
def get_image(filename):
    return send_from_directory('data/images', filename)

@app.route('/predict', methods=['POST'])
@cross_origin()  # Enable CORS for this route
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        file = request.files['file']
        image = Image.open(io.BytesIO(file.read()))
        image = preprocess_image(image)
        prediction = model.predict(image)
        predicted_class = CLASS_NAMES[np.argmax(prediction[0])]
        result = {'prediction': predicted_class}
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/verify_captcha', methods=['POST'])
@cross_origin()  # Enable CORS for this route
def verify_captcha():
    try:
        data = request.get_json()
        selected_images = data['selectedImages']
        correct_selections = 0

        for base64_image in selected_images:
            image_data = base64.b64decode(base64_image)
            image = Image.open(BytesIO(image_data))
            image = preprocess_image(image)
            prediction = model.predict(image)
            predicted_class = CLASS_NAMES[np.argmax(prediction[0])]
            if predicted_class == 'trafficlight':  # Change this to the class you want to verify
                correct_selections += 1

        result = 'Correct' if correct_selections == len(selected_images) else 'Incorrect'
        return jsonify({'result': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/process_new_images', methods=['POST'])
@cross_origin()  # Enable CORS for this route
def process_new_images():
    try:
        data = request.get_json()
        image_urls = data['image_urls']
        results = []

        for image_url in image_urls:
            response = requests.get(image_url)
            image = Image.open(BytesIO(response.content))
            image = preprocess_image(image)
            prediction = model.predict(image)
            predicted_class = CLASS_NAMES[np.argmax(prediction[0])]
            results.append({'url': image_url, 'prediction': predicted_class})
            print(f"URL: {image_url}, Prediction: {predicted_class}")

        return jsonify({'results': results})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # # Reinforcement Learning Training Loop
    # image_url = 'https://oortbucket.standard.us-east-1.oortstorage.com/road0.png'  # Example URL
    # env = CustomEnv(image_url)
    # agent = DQNAgent(state_size=(224, 224, 3), action_size=4)

    # episodes = 1000
    # for e in range(episodes):
    #     state = env.reset()
    #     state = np.reshape(state, [1, 224, 224, 3])
    #     for time in range(500):
    #         action = agent.act(state)
    #         next_state, reward, done, _ = env.step(action)
    #         next_state = np.reshape(next_state, [1, 224, 224, 3])
    #         agent.train(state, action, reward, next_state, done)
    #         state = next_state
    #         if done:
    #             print(f"Episode: {e}/{episodes}, Score: {time}")
    #             break

    app.run(debug=True)