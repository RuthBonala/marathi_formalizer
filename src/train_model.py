# train.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
from indicnlp.tokenize.indic_tokenize import trivial_tokenize

# -----------------
# Expanded Dictionaries
# -----------------
marathi_dictionary = {
    "हाय": "नमस्कार",
    "बाय": "पुन्हा भेटूया",
    "थॅंक्यू": "धन्यवाद",
    "काय चाललंय?": "सध्या काय सुरू आहे?",
    "कसा आहेस?": "आपण कसे आहात?",
    "कशी आहेस?": "आपण कशा आहात?",
    "कसं काय?": "आपण कसे आहात?",
    "कुठं जातोस?": "आपण कुठे जात आहात?",
    "मी येतो": "मी येत आहे",
    "मी तुला कॉल करतो": "मी तुम्हाला फोन करतो",
    "चल भेटू": "आपण भेटूया",
    "मला झोप येतेय": "मला झोप येत आहे",
    "मला काही विचारायचं आहे": "मला काही विचारायचे आहे",
    "तू कुठे आहेस?": "आपण कुठे आहात?",
    "तुला कळलं का?": "आपल्याला समजले का?",
    "जेवण झालं का?": "आपले जेवण झाले आहे का?",
    "मस्त आहे": "छान आहे",
    "भारी आहे": "खूप छान आहे",
    "आज हवामान छान आहे": "आजचे हवामान सुंदर आहे",
    "गाडी लेट आली": "गाडी उशिरा आली आहे",
    "मी झोपलो": "मी झोपलो आहे",
    "मी पोहचलो": "मी पोहोचलो आहे",
    "उद्या सुट्टी आहे का?": "उद्या सुट्टी आहे का?",
    "काय म्हणतोस?": "आपण काय म्हणत आहात?",
    "थांब जरा": "कृपया थांबा",
    "पाऊस पडतोय": "पाऊस पडत आहे",
    "खूप भूक लागलीय": "मला खूप भूक लागली आहे",
    "गरमी आहे": "खूप उष्णता आहे",
    "थंडी आहे": "खूप थंडी आहे",
    "किती वाजले?": "सध्या किती वाजले आहेत?",
    "कुठे भेटू?": "आपण कुठे भेटूया?",
    "मला मदत कर": "कृपया मला मदत करा",
}

synonym_dict = {
    "रे": "महोदय",
    "अगं": "महोदया",
    "हाय": "नमस्कार",
    "हो": "होय",
    "काय": "काय आहे",
    "झालं": "झाले आहे",
    "झालंय": "झाले आहे",
    "झालेय": "झाले आहे",
    "माहित": "माहिती",
    "चल": "चला",
    "बघ": "पहा",
    "सांग": "कृपया सांगा",
    "ना": "कृपया",
    "मला": "माझ्यासाठी",
    "नंतर": "पश्चात",
    "भेटू": "भेटूया",
    "कॉल": "फोन",
    "भारी": "छान",
    "मस्त": "छान",
    "खूप": "अत्यंत",
    "थांब": "थांबा",
}

# ---------
# Cleaning / preprocessing
# ---------
PUNCT = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~।…–—'
TRANS = str.maketrans({c: " " for c in PUNCT})

factory = IndicNormalizerFactory()
normalizer = factory.get_normalizer("mr")

def marathi_lemmatize(text: str) -> str:
    text = normalizer.normalize(text)
    return ' '.join(trivial_tokenize(text, lang="mr"))

def normalize_synonyms(text: str) -> str:
    words = text.split()
    return ' '.join(synonym_dict.get(w, w) for w in words)

def preprocess_sentence(sentence: str) -> str:
    if sentence in marathi_dictionary:  # direct mapping first
        return marathi_dictionary[sentence]
    sentence = sentence.translate(TRANS)
    sentence = sentence.strip()
    sentence = marathi_lemmatize(sentence)
    sentence = normalize_synonyms(sentence)
    return sentence

# -----------------
# Load dataset
# -----------------
df = pd.read_csv(r"C:/Users/admin/marathi_formalizer/data/marathi_formalization_dataset.csv")
df["input_processed"] = df["informal"].astype(str).apply(preprocess_sentence)
df["target_processed"] = df["formal"].astype(str).apply(preprocess_sentence)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    df["input_processed"], df["target_processed"], test_size=0.2, random_state=42
)

# -----------------
# Vectorizer (TF-IDF with ngrams)
# -----------------
vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3,6), min_df=1)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Save vectorizer and dataset for retrieval
joblib.dump(vectorizer, "vectorizer.pkl")
df.to_csv("processed_dataset.csv", index=False)

print("✅ Training data processed and vectorizer saved.")
