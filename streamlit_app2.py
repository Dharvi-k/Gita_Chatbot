import streamlit as st
import pandas as pd
import numpy as np
import faiss
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from sklearn.linear_model import LogisticRegression

# ----------------------------
# NLTK setup
# ----------------------------
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


# ----------------------------
# Preprocessing function
# ----------------------------
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = nltk.word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)


# ----------------------------
# Cached model loaders
# ----------------------------
@st.cache_resource
def load_embedding_model():
    try:
        return SentenceTransformer("all-MiniLM-L6-v2")
    except Exception:
        # fallback if HuggingFace is rate-limiting
        return SentenceTransformer("paraphrase-MiniLM-L3-v2")


@st.cache_resource
def load_llm():
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="float16"
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config=bnb_config
    )
    return pipeline("text-generation", model=model, tokenizer=tokenizer)


embedding_model = load_embedding_model()
generator = load_llm()


# ----------------------------
# Load dataset
# ----------------------------
@st.cache_data
def load_data():
    df = pd.read_json("gita_translation_data.json")  # <-- make sure your file is here
    if "cleaned_verse" not in df.columns:
        df["cleaned_verse"] = df["text"].apply(preprocess_text)
    return df


df = load_data()


# ----------------------------
# Build FAISS index
# ----------------------------
@st.cache_resource
def build_faiss_index(embeddings):
    embeddings = np.array(embeddings).astype("float32")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index


embeddings = embedding_model.encode(df["cleaned_verse"].tolist())
index = build_faiss_index(embeddings)


# ----------------------------
# Intent classifier (simple)
# ----------------------------
intents = {
    "greeting": ["hello", "hi", "how are you", "good morning", "hey there", "namaste"],
    "gita": [
        "what does the gita say about attachment",
        "teachings of krishna on duty",
        "explain karma yoga",
        "bhagavad gita on meditation",
        "what does arjuna learn in the gita",
        "summary of bhagavad gita chapter 2"
    ],
    "out_of_domain": [
        "who is the president of india",
        "what is the most expensive car",
        "tell me a joke",
        "weather forecast",
        "capital of france",
        "who won the fifa world cup"
    ]
}

X, y = [], []
for label, samples in intents.items():
    for s in samples:
        X.append(embedding_model.encode(s))
        y.append(label)

X = np.array(X)
y = np.array(y)

clf = LogisticRegression(max_iter=1000)
clf.fit(X, y)


def detect_intent(query):
    vec = embedding_model.encode([query])
    return clf.predict(vec)[0]


# ----------------------------
# FAISS retrieval
# ----------------------------
def retrieve_verses(query, k=3):
    cleaned_query = preprocess_text(query)
    query_emb = embedding_model.encode([cleaned_query]).astype("float32")
    D, I = index.search(query_emb, k=k)
    return [df.iloc[idx]["text"] for idx in I[0]]


# ----------------------------
# Generate explanation
# ----------------------------
def generate_explanation(query, verses):
    context = "\n".join(verses)
    prompt = f"""
    You are a Bhagavad Gita expert. The user asked: "{query}"

    Here are the relevant verses:
    {context}

    Please explain them in simple, everyday language with practical meaning.
    """
    response = generator(prompt, max_new_tokens=250, temperature=0.7, do_sample=True)
    return response[0]["generated_text"]


# ----------------------------
# Streamlit UI
# ----------------------------
st.title("📖 Gita Chatbot")
st.write("Ask me anything about the Bhagavad Gita 🙏")

user_query = st.text_input("Your question:")

if user_query:
    intent = detect_intent(user_query)

    if intent == "greeting":
        st.success("Hello! How are you doing today?")
    elif intent == "out_of_domain":
        st.warning("🙏 I can only answer questions related to the Bhagavad Gita.")
    else:
        verses = retrieve_verses(user_query, k=3)
        explanation = generate_explanation(user_query, verses)

        st.subheader("🔹 Relevant Verses")
        for v in verses:
            st.write(f"- {v}")

        st.subheader("🔹 Explanation")
        st.write(explanation)
