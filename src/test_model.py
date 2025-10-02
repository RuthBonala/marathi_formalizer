# test_model.py
import joblib
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from train_model import preprocess_sentence

# Load vectorizer + dataset
vectorizer = joblib.load("vectorizer.pkl")
df = pd.read_csv("processed_dataset.csv")

informal_sentences = df["input_processed"].tolist()
formal_sentences = df["target_processed"].tolist()
informal_vectors = vectorizer.transform(informal_sentences)

print("Marathi Formalizer (Retrieval Based)")
print("Type 'quit' to exit.\n")

while True:
    user_input = input("Enter an informal Marathi sentence (or 'quit' to exit): ")
    if user_input.lower() in ["quit", "exit"]:
        break

    processed = preprocess_sentence(user_input)
    user_vec = vectorizer.transform([processed])

    similarities = cosine_similarity(user_vec, informal_vectors)
    best_idx = similarities.argmax()

    print("✅ Formalized Sentence:", formal_sentences[best_idx], "\n")
