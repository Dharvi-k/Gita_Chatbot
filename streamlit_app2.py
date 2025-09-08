# app.py
import streamlit as st
import pandas as pd
import numpy as np
import faiss
import re
import nltk

from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from sklearn.linear_model import LogisticRegression
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# --------------------------
# 1. Load Data (JSON instead of CSV)
# --------------------------
@st.cache_resource
def load_data():
    
    df = pd.read_json("gita_translation_data.json")  
    return df

df = load_data()

# --------------------------
# 2. Load Embedding Model
# --------------------------
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

embeddings = embedding_model.encode(df['cleaned_verse'])
embeddings = np.array(embeddings).astype('float32')

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# --------------------------
# 3. Load LLM (Mistral 7B)
# --------------------------
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
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
    return generator

generator = load_llm()

# --------------------------
# 4. Intent Classifier
# --------------------------
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

# --------------------------
# 5. Preprocessing
# --------------------------
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess_query(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# --------------------------
# 6. Retrieval + Explanation
# --------------------------
def retrieve_verses(query, k=3):
    cleaned_query = preprocess_query(query)
    query_embedding = embedding_model.encode([cleaned_query]).astype('float32')
    D, I = index.search(query_embedding, k=k)
    verses = [df.iloc[idx]['text'] for idx in I[0]]
    return verses

def detect_task(query):
    q = query.lower()
    if any(x in q for x in ["summarize", "summary", "what is bhagavad gita", "essence of gita"]):
        return "summarize"
    elif "translate" in q:
        return "translate"
    elif "compare" in q or "difference" in q:
        return "compare"
    elif "explain" in q or "meaning" in q:
        return "explain"
    elif "what does" in q or "say about" in q:
        return "thematic"
    else:
        return "default"

def generate_explanation(query, retrieved_verses):
    context = "\n".join(retrieved_verses)
    task = detect_task(query)

    if task == "summarize":
        prompt = f"You are a Bhagavad Gita expert. Write a concise summary in 150 words for: {query}"
        response = generator(prompt, max_new_tokens=300, temperature=0.7)
        return None, response[0]["generated_text"]

    elif task == "translate":
        prompt = f"Translate these Bhagavad Gita verses into simple English:\n{context}"
        max_tokens = 200

    elif task == "compare":
        prompt = f"Compare and explain similarities/differences of these verses:\n{context}"
        max_tokens = 300

    elif task == "explain":
        prompt = f"Explain these verses in simple, everyday language:\n{context}"
        max_tokens = 250

    else:
        prompt = f"Answer clearly based on Bhagavad Gita:\n{context}"
        max_tokens = 300

    response = generator(prompt, max_new_tokens=max_tokens, temperature=0.7, do_sample=True)
    return retrieved_verses, response[0]["generated_text"]

# --------------------------
# 7. Chatbot Function
# --------------------------
def gita_chatbot(user_query, k=3):
    intent = detect_intent(user_query)

    if intent == "greeting":
        return None, "Hello! How are you doing today?"

    elif intent == "out_of_domain":
        return None, "🙏 I can only answer questions related to the Bhagavad Gita."

    elif intent == "gita":
        verses = retrieve_verses(user_query, k=k)
        verses, explanation = generate_explanation(user_query, verses)
        return verses, explanation

    else:
        return None, "I'm not sure how to respond to that."

# --------------------------
# 8. Streamlit UI
# --------------------------
st.title("📖 Gita Chatbot")
st.write("Ask me anything about the Bhagavad Gita.")

user_query = st.text_input("Your Question:")

if st.button("Ask"):
    with st.spinner("Thinking..."):
        verses, answer = gita_chatbot(user_query)

    if verses:
        st.subheader("🔹 Relevant Verses")
        for v in verses:
            st.write(v)

    st.subheader("✨ Explanation")
    st.write(answer)
