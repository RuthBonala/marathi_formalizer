import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
from indicnlp.tokenize.indic_tokenize import trivial_tokenize

# Load data
df = pd.read_csv("C:/marathi-formalizer/data/marathi_formalization_dataset.csv")

# Informal-to-formal sentence dictionary
marathi_dictionary = {
    "कधी येणार?": "आपण कधी येणार आहात?",
    "काय झालं रे तुला?": "तुला काय झाले आहे?",
    "काय झालं?": "आपल्याला काय अडचण आली आहे?",
    "काय चाललंय?": "सध्या काय सुरु आहे?",
    "काय रे हाच का?": "हीच व्यक्ती आहे का?",
    "काय रे मजा!": "हे खूप आनंददायक आहे!",
    "थांब जरा!": "कृपया थोडा वेळ थांबा.",
    "थोडं थांब.": "कृपया थोडा वेळ थांबा.",
    "तु काय करतोस?": "तुम्ही काय करत आहात?",
    "तुला माहित आहे का?": "तुम्हाला माहिती आहे का?",
    "तू कुठे जातोय?": "तुम्ही कुठे जात आहात?",
    "तू ठीक आहेस का?": "तुमची तब्येत ठीक आहे का?",
    "नको ना असं करूस.": "कृपया असे करू नका.",
    "पाहिलं का?": "तुम्ही ते पाहिलं का?",
    "बघितलंस का?": "तुम्ही ते पाहिलं का?",
    "भेटू नंतर.": "आपण नंतर भेटूया.",
    "माझं काम झालं.": "माझं काम पूर्ण झालं आहे.",
    "मस्त चाललंय!": "सर्व काही सुरळीत सुरू आहे.",
    "माझी चूक झाली.": "माझ्याकडून चूक झाली आहे.",
    "माफ कर.": "कृपया मला क्षमा करा.",
    "माफ कर ना!": "कृपया मला क्षमा करा.",
    "माझ्याकडं पैसे नाहीत.": "माझ्याकडे पैसे उपलब्ध नाहीत.",
    "माझ्यासाठी थांब.": "कृपया माझ्यासाठी थांबा.",
    "मला झोप येतेय.": "मला झोप येत आहे.",
    "मला माहिती नाही.": "माझ्या माहितीनुसार ते मला माहीत नाही.",
    "मला समजलं नाही.": "मला समजले नाही.",
    "मला काही विचारायचं आहे.": "मला काही विचारायचे आहे.",
    "मला वेळ नाही.": "माझ्याकडे वेळ नाही.",
    "मला पाहिजे तेच दे.": "कृपया मला आवश्यक वस्तू द्या.",
    "मला भेटायचं आहे.": "मला तुम्हाला भेटायचे आहे.",
    "मला वाटतं तसं नाही.": "माझ्या मते तसे नाही.",
    "मला हवयं!": "मला ते हवे आहे.",
    "माफ करशील का?": "कृपया मला क्षमा कराल का?",
    "सांग ना मला!": "कृपया मला सांगा.",
    "सुट्टी आहे का?": "आज सुट्टी आहे का?",
    "हे खरं आहे का?": "हे सत्य आहे का?",
    "हे घे.": "कृपया हे घ्या.",
    "हे मस्त आहे!": "हे अतिशय उत्कृष्ट आहे!",
    "हा भारी आहे!": "हा अत्यंत चांगला आहे!",
    "हो का?": "खरोखर असे आहे का?",
    "हो मला माहित आहे.": "होय, मला माहिती आहे.",
    "हो हो!": "होय.",
    "अजून वेळ लागेल का?": "थोडा अधिक वेळ लागेल का?",
    "आपण भेटलो का पूर्वी?": "आपण पूर्वी कधी भेटलो आहोत का?",
    "चल भेटू नंतर.": "आपण नंतर भेटूया.",
    "चल निघूया.": "चला आपण निघूया."
}

# Synonym dictionary
synonym_dict = {
    "सांग": "कृपया",
    "ना": "मला",
    "मला": "सांगा",
    "मला!": "सांगा.",
    "मला!!": "सांगा.",
    "चल": "आपण",
    "भेटू": "नंतर",
    "नंतर": "भेटूया",
    "नंतर.": "भेटूया.",
    "काय": "तुला",
    "का?!": "का?",
    "तुला": "तुम्हाला",
    "तुला?": "आहे?",
    "तुला?!": "आहे?",
    "रे": "व्यक्ती",
    "हाच": "आहे",
    "खरं": "सत्य",
    "झालं": "काय",
    "झाल": "काय",
    "झालंय": "काय",
    "झालेली": "काय",
    "झाली": "काय",
    "माहित": "माहिती",
    "हाच": "आहे"
}

def normalize_synonyms(text):
    words = text.split()
    normalized = [synonym_dict.get(word, word) for word in words]
    return ' '.join(normalized)

# Simple stemmer
def stem_word(word):
    suffixes = ['त आहे', 'ली आहे', 'ले आहे', 'लेला', 'लेली', 'लो', 'ला', 'ली', 'त', 'ले', 'ात', 'ा', 'ी', 'े', 'ो']
    for suffix in suffixes:
        if word.endswith(suffix):
            return word[:-len(suffix)]
    return word

# Indic NLP
factory = IndicNormalizerFactory()
normalizer = factory.get_normalizer("mr")

def marathi_lemmatize(text):
    text = normalizer.normalize(text)
    return ' '.join(trivial_tokenize(text, lang="mr"))

# Preprocess pipeline

def preprocess_sentence(sentence):
    if sentence in marathi_dictionary:
        sentence = marathi_dictionary[sentence]
    sentence = re.sub(r'[^\w\s]', '', sentence)
    sentence = sentence.strip().lower()
    sentence = marathi_lemmatize(sentence)
    sentence = normalize_synonyms(sentence)
    words = sentence.split()
    new_words = [stem_word(word) for word in words]
    return " ".join(new_words)

# Apply preprocessing
df["input_processed"] = df["informal"].apply(preprocess_sentence)
df["target_processed"] = df["formal"].apply(preprocess_sentence)

# Split data
X_train, X_test, y_train, y_test = train_test_split(df["input_processed"], df["target_processed"], test_size=0.2, random_state=42)

# ML model
model = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
    ("classifier", LogisticRegression(max_iter=200))
])

# Train
model.fit(X_train, y_train)

# Predict
preds = model.predict(X_test)

# Evaluate
print("\n🧾 Classification Report:")
print(classification_report(y_test, preds))

# Test function
def formalize(text):
    cleaned = preprocess_sentence(text)
    return model.predict([cleaned])[0]

# Try test
print("\n🔁 Formalized:", formalize("तु काय करतोस?"))
