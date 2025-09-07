import streamlit as st
import requests

# ==========================
# Hugging Face API Settings
# ==========================
HF_API_KEY = st.secrets["HF_API_KEY"]  # Add in Streamlit -> Settings -> Secrets
HEADERS = {"Authorization": f"Bearer {HF_API_KEY}"}

# Free-tier friendly model
MODEL = "gpt2"

# ==========================
# Query Hugging Face Model
# ==========================
def query_huggingface(prompt, max_tokens=200):
    # Add ?wait_for_model=true so HF spins up the model instead of returning 404
    url = f"https://api-inference.huggingface.co/models/{MODEL}?wait_for_model=true"
    payload = {"inputs": prompt, "parameters": {"max_new_tokens": max_tokens}}

    try:
        response = requests.post(url, headers=HEADERS, json=payload, timeout=60)
    except requests.exceptions.RequestException as e:
        return f"⚠️ Network error: {e}"

    if response.status_code != 200:
        try:
            return f"⚠️ Error {response.status_code}: {response.json()}"
        except Exception:
            return f"⚠️ Error {response.status_code}: {response.text}"

    try:
        data = response.json()
    except ValueError:
        return f"⚠️ Failed to parse JSON. Raw response: {response.text}"

    # Extract generated text
    if isinstance(data, list) and data and "generated_text" in data[0]:
        return data[0]["generated_text"]
    elif isinstance(data, dict) and "generated_text" in data:
        return data["generated_text"]
    else:
        return str(data)  # Fallback for debugging

# ==========================
# Chatbot Logic
# ==========================
def gita_chatbot(user_query):
    prompt = f"""
    You are a Bhagavad Gita expert.
    The user asked: "{user_query}"

    Answer with spiritual depth but in simple, everyday language.
    """
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
