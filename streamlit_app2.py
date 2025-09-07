import streamlit as st
import requests

# ==========================
# Hugging Face API Settings
# ==========================
HF_API_KEY = st.secrets["HF_API_KEY"]  # Add in Streamlit -> Settings -> Secrets
HEADERS = {"Authorization": f"Bearer {HF_API_KEY}"}
MODEL = "facebook/bart-large-cnn"  # Stable free model

# ==========================
# Query Hugging Face Model
# ==========================
def query_huggingface(prompt, max_tokens=300):
    url = f"https://api-inference.huggingface.co/models/{MODEL}"
    payload = {"inputs": prompt, "parameters": {"max_new_tokens": max_tokens}}

    try:
        response = requests.post(url, headers=HEADERS, json=payload, timeout=60)
    except requests.exceptions.RequestException as e:
        return f"⚠️ Network error: {e}"

    if response.status_code != 200:
        try:
            return f"⚠️ Error {response.status_code}: {response.json().get('error', response.text)}"
        except ValueError:
            return f"⚠️ Error {response.status_code}: {response.text}"

    try:
        data = response.json()
    except ValueError:
        return f"⚠️ Failed to parse JSON. Raw: {response.text}"

    if isinstance(data, list) and data and "summary_text" in data[0]:
        return data[0]["summary_text"]
    elif isinstance(data, list) and "generated_text" in data[0]:
        return data[0]["generated_text"]
    return str(data)

# ==========================
# Task Detection Logic
# ==========================
def detect_task(query: str) -> str:
    q = query.lower()
    if "translate" in q:
        return "translate"
    elif "summarize" in q or "short" in q:
        return "summarize"
    elif "compare" in q:
        return "compare"
    else:
        return "explain"

# ==========================
# Chatbot Logic
# ==========================
def gita_chatbot(user_query):
    task = detect_task(user_query)

    if task == "summarize":
        prompt = f"Summarize this Bhagavad Gita concept in simple words (around 150 words): {user_query}"
    elif task == "translate":
        prompt = f"Translate this Bhagavad Gita verse into plain English with meaning: {user_query}"
    elif task == "compare":
        prompt = f"Compare the Bhagavad Gita's view on {user_query} with modern life (200 words)."
    else:  # explain (default)
        prompt = f"Explain this Bhagavad Gita teaching in detail, in at least 200 words, with spiritual depth but simple everyday language: {user_query}"

    return query_huggingface(prompt)

# ==========================
# Streamlit UI
# ==========================
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
