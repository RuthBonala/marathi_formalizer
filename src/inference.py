import sys
import os
import joblib

# Ensure we can import from src
sys.path.append(os.path.dirname(__file__))

# Import the function from train_model.py
from train_model import preprocess_sentence  

# Load the saved model with joblib (not pickle)
model = joblib.load("marathi_formalizer.pkl")

print("✅ Model loaded successfully!")

# Inference loop
while True:
    sentence = input("\nEnter an informal Marathi sentence (or 'quit' to exit): ")
    if sentence.lower() == "quit":
        break

    # Preprocess
    processed = preprocess_sentence(sentence)

    # Predict (sklearn-based model)
    prediction = model.predict([processed])[0]

    print("Formalized Sentence:", prediction)
