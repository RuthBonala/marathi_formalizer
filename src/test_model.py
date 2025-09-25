import joblib

# Load your trained pipeline (vectorizer + model)
model = joblib.load("marathi_formalizer.pkl")

print("Marathi Formalizer (type 'quit' to exit)\n")

while True:
    # Take user input
    sentence = input("Enter an informal Marathi sentence (or 'quit' to exit): ")

    if sentence.lower() in ["quit", "exit"]:
        print("👋 Exiting...")
        break

    # Predict formalized version
    prediction = model.predict([sentence])[0]

    print("✅ Formalized Sentence:", prediction, "\n")
