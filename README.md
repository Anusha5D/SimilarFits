# StyleSuggest
A deep learning-powered fashion recommendation system that suggests visually similar styles using ResNet50 and transfer learning.

## 🔧 Tech Stack

- Python 🐍
- TensorFlow / Keras
- NumPy, Pandas, Matplotlib
- ResNet50 (Transfer Learning)
- OpenCV / PIL for image preprocessing

## 🚀 How It Works

1. Load and preprocess fashion images.
2. Use ResNet50 (excluding the top layer) to extract 2048-dimensional feature vectors.
3. Apply cosine similarity or nearest neighbor search to recommend similar items.
4. Display the top-k recommended images.
