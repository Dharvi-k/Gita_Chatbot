import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
import numpy as np
import re
import nltk
import os

# ---------------------------------------------
# Load Hugging Face model (Mistral-7B)
# ---------------------------------------------
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

# ---------------------------------------------
# Load Embedding Model
# ---------------------------------------------
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Example intents (expand later with your dataset)
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

# ---------------------------------------------
# Helper Functions
# ---------------------------------------------
def detect_task(query):
    query = query.lower()
    if any(x in query for x in ["summarize", "summary", "essence of gita"]):
        return "summarize"
    elif "translate" in query:
        return "translate"
    elif "compare" in query or "difference" in query:
        return "compare"
    elif "explain" in query or "meaning" in query:
        return "explain"
    elif "what does" in query or "say about" in query:
        return "thematic"
    else:
        return "default"

def detect_intent(query):
    vec = embedding_model.encode([query])
    pred = clf.predict(vec)[0]
    return pred

def generate_explanation(query, verses):
    context = "\n".join(verses)
    task = detect_task(query)

    if task == "summarize":
        prompt = f"""
        You are a Bhagavad Gita expert. The user asked: "{query}"
        Write a concise **summary of the Bhagavad Gita** in about 150 words.
        """
        response = generator(prompt, max_new_tokens=300, temperature=0.7)
        return response[0]["generated_text"]

    elif task == "translate":
        prompt = f"Translate these verses:\n{context}"
        response = generator(prompt, max_new_tokens=200, temperature=0.7)
        return response[0]["generated_text"]

    elif task == "compare":
        prompt = f"Compare these verses:\n{context}"
        response = generator(prompt, max_new_tokens=300, temperature=0.7)
        return response[0]["generated_text"]

    elif task == "explain":
        prompt = f"Explain these verses in simple words:\n{context}"
        response = generator(prompt, max_new_tokens=250, temperature=0.7)
        return response[0]["generated_text"]

    else:  # default
        prompt = f"Answer clearly based on Bhagavad Gita:\n{context}"
        response = generator(prompt, max_new_tokens=300, temperature=0.7)
        return response[0]["generated_text"]

def gita_chatbot(user_query):
    intent = detect_intent(user_query)

    if intent == "greeting":
        return "Hello! How are you doing today?"

    elif intent == "out_of_domain":
        return "🙏 Sorry, I can only answer questions related to the Bhagavad Gita."

    elif intent == "gita":
        # For now, skip FAISS retrieval (you can plug in your verse retrieval later)
        verses = ["Sample Bhagavad Gita verse..."]
        explanation = generate_explanation(user_query, verses)
        return explanation

    else:
        return "I'm not sure how to respond to that."

# ---------------------------------------------
# Streamlit UI
# ---------------------------------------------
st.set_page_config(page_title="Bhagavad Gita Chatbot", page_icon="📖")

st.title("📖 Bhagavad Gita Chatbot")
st.write("Ask me anything about the Bhagavad Gita!")

user_input = st.text_input("Your Question:")

if st.button("Ask"):
    if user_input.strip() != "":
        with st.spinner("Thinking..."):
            answer = gita_chatbot(user_input)
        st.success(answer)
    else:
        st.warning("Please enter a question.")