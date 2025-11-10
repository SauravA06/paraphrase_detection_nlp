import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

# --- Load models ---
st.title("Paraphrase Detection")

# --- Model Loading (Cached) ---
# This is the correct way to load models in Streamlit.
# It loads them once and keeps them in memory.
@st.cache_resource
def load_models():
    print("--- LOADING ALL MODELS (This should only run once) ---")
    
    # TF-IDF
    vectorizer = TfidfVectorizer()
    
    # SBERT (Using the best model)
    sbert_model = SentenceTransformer('all-mpnet-base-v2', device='cpu')
    sbert_model.eval()
    
    
    return vectorizer, sbert_model

# Load the models using the cached function
# You will see the "LOADING" print message in your terminal the first time.
vectorizer, sbert_model= load_models()
# --- End Model Loading ---


# --- Input Section ---

# The long default text has been removed from both lines
sent1 = st.text_area("Sentence 1", placeholder="Enter the first paragraph here...")
sent2 = st.text_area("Sentence 2", placeholder="Enter the second paragraph here...")

model_choice = st.selectbox("Choose model", ["SBERT", "TF-IDF + Cosine"])

# --- Run on button click ---
if st.button("Check Paraphrase"):
    if model_choice == "TF-IDF + Cosine":
        vectors = vectorizer.fit_transform([sent1, sent2])
        sim = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
        st.write(f"**Similarity:** {sim:.2f}")
        st.write("✅ Paraphrased" if sim > 0.6 else "❌ Not Paraphrased")

    elif model_choice == "SBERT":
        emb1 = sbert_model.encode(sent1)
        emb2 = sbert_model.encode(sent2)

        # Calculate cosine similarity manually with numpy
        sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

        st.write("--- RESULT ---")
        st.write(f"**Similarity:** {sim:.2f}")
        st.write("✅ Paraphrased" if sim > 0.7 else "❌ Not Paraphrased")