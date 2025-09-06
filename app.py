import streamlit as st
import requests
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression

# ---------------- Hugging Face Inference API ----------------
HF_API_KEY = st.secrets["HF_API_KEY"]  # Add in Streamlit Cloud -> Settings -> Secrets
API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
headers = {"Authorization": f"Bearer {HF_API_KEY}"}

def query_huggingface(prompt, max_tokens=300):
    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": max_tokens, "temperature": 0.7}
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    response.raise_for_status()
    data = response.json()
    if isinstance(data, list) and "generated_text" in data[0]:
        return data[0]["generated_text"]
    else:
        return "⚠️ Unexpected response from Hugging Face API."

# ---------------- Intent Classification ----------------
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

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
    pred = clf.predict(vec)[0]
    return pred

# ---------------- Chatbot Logic ----------------
def gita_chatbot(user_query):
    intent = detect_intent(user_query)

    if intent == "greeting":
        return "Hello! How are you doing today?"

    elif intent == "out_of_domain":
        return "🙏 Sorry, I can only answer questions related to the Bhagavad Gita."

    elif intent == "gita":
        prompt = f"""
        You are a Bhagavad Gita expert. The user asked: "{user_query}".
        Answer with spiritual depth but in simple, everyday language.
        """
        answer = query_huggingface(prompt)
        return answer

    else:
        return "I'm not sure how to respond to that."

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="Bhagavad Gita Chatbot", page_icon="📖")

st.title("📖 Bhagavad Gita Chatbot")
st.write("Ask me anything about the Bhagavad Gita!")

user_input = st.text_input("Your Question:")

if st.button("Ask"):
    if user_input.strip():
        with st.spinner("Thinking..."):
            answer = gita_chatbot(user_input)
        st.success(answer)
    else:
        st.warning("Please enter a question.")