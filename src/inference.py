# inference.py
from transformers import MT5ForConditionalGeneration, MT5Tokenizer
import torch

# Load model and tokenizer
model_path = "../model_output"
tokenizer = MT5Tokenizer.from_pretrained(model_path)
model = MT5ForConditionalGeneration.from_pretrained(model_path)

# Function to formalize sentence
def formalize(text):
    input_text = "formalize: " + text
    input_ids = tokenizer.encode(input_text, return_tensors="pt", truncation=True)
    output = model.generate(input_ids, max_length=64)
    formal_sentence = tokenizer.decode(output[0], skip_special_tokens=True)
    return formal_sentence

# Try it out!
while True:
    informal = input("\nEnter informal Marathi sentence: ")
    if informal.lower() in ["exit", "quit"]:
        break
    formal = formalize(informal)
    print("Formalized →", formal)
