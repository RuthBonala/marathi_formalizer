import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
from indicnlp.tokenize.indic_tokenize import trivial_tokenize

# -----------------
# Dictionaries
# -----------------
marathi_dictionary = {
    # Greetings & common
    "हाय": "नमस्कार",
    "बाय": "पुन्हा भेटू",
    "थॅंक्यू": "धन्यवाद",
    "थॅंक्स": "मनःपूर्वक आभार",
    "काय चाललंय?": "सध्या काय सुरू आहे?",
    "कसा आहेस?": "आपण कसे आहात?",
    "कशी आहेस?": "आपण कशा आहात?",
    "कसं काय?": "आपण कसे आहात?",
    "कुठं जातोस?": "आपण कुठे जात आहात?",
    "भेटू नंतर.": "आपण नंतर भेटूया.",
    "चल भेटू नंतर.": "चला आपण नंतर भेटूया.",
    "चल निघूया.": "चला आपण निघूया.",

    # Actions
    "चल": "चला",
    "ये": "या",
    "बस": "बसून घ्या",
    "थांब": "कृपया थांबा",
    "थांब जरा!": "कृपया थोडा वेळ थांबा.",
    "थोडं थांब.": "कृपया थोडा वेळ थांबा.",
    "खाऊ": "भोजन करा",
    "झोप": "विश्रांती घ्या",
    "उठ": "उठा",
    "बघ": "पहा",
    "ऐक": "कृपया ऐका",
    "सांग": "कृपया सांगा",
    "कर": "कृपया करा",

    # Negatives
    "नको": "नको आहे",
    "नको ना असं करूस.": "कृपया असे करू नका.",
    "करू नको": "कृपया करू नका",
    "चल नको": "कृपया जाऊ नका",
    "नाही": "नाही आहे",
    "काही नाही": "काहीही उपलब्ध नाही",

    # Emotions / slang
    "मस्त": "उत्तम",
    "झकास": "उत्कृष्ट",
    "धम्माल": "आनंददायी",
    "भारी": "अत्यंत चांगले",
    "जबरी": "अतिशय उत्कृष्ट",
    "हे मस्त आहे!": "हे अतिशय उत्कृष्ट आहे!",
    "हा भारी आहे!": "हा अत्यंत चांगला आहे!",

    # Personal / pronouns
    "तु काय करतोस?": "आपण काय करत आहात?",
    "तू कुठे जातोय?": "आपण कुठे जात आहात?",
    "तू ठीक आहेस का?": "आपली तब्येत ठीक आहे का?",
    "तुला माहित आहे का?": "आपल्याला माहिती आहे का?",
    "काय झालं रे तुला?": "आपल्याला काय झाले आहे?",
    "माझ्याकडं पैसे नाहीत.": "माझ्याकडे पैसे उपलब्ध नाहीत.",
    "मला भेटायचं आहे.": "मला आपल्याला भेटायचे आहे.",
    "माझं काम झालं.": "माझं काम पूर्ण झालं आहे.",
    "माझी चूक झाली.": "माझ्याकडून चूक झाली आहे.",
    "मला झोप येतेय.": "मला झोप येत आहे.",
    "मला समजलं नाही.": "मला समजले नाही.",
    "मला माहिती नाही.": "माझ्या माहितीनुसार मला माहिती नाही.",
    "मला काही विचारायचं आहे.": "मला काही विचारायचे आहे.",
    "मला वेळ नाही.": "माझ्याकडे वेळ नाही.",
    "मला हवयं!": "मला ते हवे आहे.",

    # Polite apologies & requests
    "माफ कर.": "कृपया मला क्षमा करा.",
    "माफ कर ना!": "कृपया मला क्षमा करा.",
    "माफ करशील का?": "कृपया मला क्षमा कराल का?",
    "सांग ना मला!": "कृपया मला सांगा.",

    # Questions
    "कधी येणार?": "आपण कधी येणार आहात?",
    "काय झालं?": "आपल्याला काय अडचण आली आहे?",
    "काय रे मजा!": "हे खूप आनंददायक आहे!",
    "पाहिलं का?": "आपण पाहिले का?",
    "बघितलंस का?": "आपण पाहिले का?",
    "सुट्टी आहे का?": "आज सुट्टी आहे का?",
    "हे खरं आहे का?": "हे सत्य आहे का?",
    "हो का?": "खरोखर तसे आहे का?",
    "हो मला माहित आहे.": "होय, मला माहिती आहे.",
    "हो हो!": "होय.",
    "अजून वेळ लागेल का?": "थोडा अधिक वेळ लागेल का?",
    "आपण भेटलो का पूर्वी?": "आपण पूर्वी कधी भेटलो आहोत का?",
}

# Synonym-level replacements (used inside sentences)
synonym_dict = {
    "रे": "महोदय",
    "अगं": "महोदया",
    "हाय": "आहे",
    "हो": "होय",
    "काय": "काय आहे",
    "झालं": "झाले आहे",
    "झालंय": "झाले आहे",
    "माहित": "माहिती",
    "चल": "चला",
    "बघ": "पहा",
    "सांग": "कृपया सांगा",
    "ना": "कृपया",
    "मला": "माझ्यासाठी",
    "नंतर": "पश्चात",
    "भेटू": "भेटूया",
}


# ---------
# Cleaning
# ---------
PUNCT = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~।…–—'
TRANS = str.maketrans({c: " " for c in PUNCT})

def normalize_synonyms(text: str) -> str:
    words = text.split()
    return ' '.join(synonym_dict.get(w, w) for w in words)

factory = IndicNormalizerFactory()
normalizer = factory.get_normalizer("mr")

def marathi_lemmatize(text: str) -> str:
    text = normalizer.normalize(text)
    # tokenization keeps Unicode intact
    return ' '.join(trivial_tokenize(text, lang="mr"))

def preprocess_sentence(sentence: str) -> str:
    # dictionary replacement
    if sentence in marathi_dictionary:
        sentence = marathi_dictionary[sentence]
    # keep Devanagari, only strip punctuation
    sentence = sentence.translate(TRANS)
    sentence = sentence.strip()
    # (lower has no effect for Devanagari, but harmless)
    sentence = sentence.lower()
    sentence = marathi_lemmatize(sentence)
    sentence = normalize_synonyms(sentence)
    # IMPORTANT: no naive stemming (kept OFF to preserve matras)
    return sentence

# -------------
# Load & prep
# -------------
df = pd.read_csv(r"C:/marathi-formalizer/data/marathi_dataset_500.csv")
df["input_processed"]  = df["informal"].apply(preprocess_sentence)
df["target_processed"] = df["formal"].apply(preprocess_sentence)

X_train, X_test, y_train, y_test = train_test_split(
    df["input_processed"], df["target_processed"], test_size=0.2, random_state=42
)

# -------------
# Model (char n-grams + LinearSVC)
# -------------
model = Pipeline([
    ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(3,6), min_df=1)),
    ("clf", LinearSVC())
])

model.fit(X_train, y_train)

preds = model.predict(X_test)
print("\n🧾 Classification Report:")
print(classification_report(y_test, preds))

joblib.dump(model, "marathi_formalizer.pkl")
print("✅ Model saved as marathi_formalizer.pkl")
