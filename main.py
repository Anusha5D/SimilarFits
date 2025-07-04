import os
import pickle
import random
import numpy as np
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.preprocessing import image
from numpy.linalg import norm
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import GlobalMaxPooling2D
import tensorflow as tf

# Load saved embeddings and filenames
feature_list = pickle.load(open('embeddings.pkl', 'rb'))
filenames = pickle.load(open('filenames_updated.pkl', 'rb'))  # Ensure correct file

# Load pre-trained ResNet50 model (same as in training)
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

model = tf.keras.Sequential([
    base_model,
    GlobalMaxPooling2D()
])

# Flask app setup
app = Flask(__name__)

# Function to extract features from an image (same preprocessing as training)
def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)  # Ensuring consistency
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)  # Normalize
    return normalized_result

# Function to get similar images
def recommend_similar_images(query_img, top_n=8):
    query_features = extract_features(query_img, model).reshape(1, -1)
    similarities = cosine_similarity(query_features, feature_list)[0]
    top_indices = np.argsort(similarities)[::-1][1:top_n + 1]  # Exclude query image
    recommended_images = [filenames[idx] for idx in top_indices]
    return recommended_images

# Route to display random images
@app.route('/')
def index():
    random_images = random.sample(filenames, 4)
    return render_template('index.html', images=random_images)

# Route to handle image selection and get recommendations
@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    selected_image = data.get("image")
    query_img_path = f"static/images/{selected_image}"  # Ensure correct path

    if not os.path.exists(query_img_path):
        return jsonify({"error": "Image not found"}), 400  # Error handling

    recommended_images = recommend_similar_images(query_img_path)
    return jsonify({"recommendations": recommended_images})

if __name__ == '__main__':
    app.run(debug=True)
