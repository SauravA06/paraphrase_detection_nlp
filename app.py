import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

# --- Configuration ---
# Set thresholds based on typical project results (these can be fine-tuned)
SBERT_THRESHOLD = 0.70
TFIDF_THRESHOLD = 0.60
BERT_MODEL_NAME = 'textattack/bert-base-uncased-MRPC'
# This BERT model is pre-trained/fine-tuned on the Microsoft Research Paraphrase Corpus (MRPC)
# Class 1 = Paraphrase, Class 0 = Not Paraphrase

# --- Application Title ---
st.set_page_config(page_title="Paraphrase Detection App", layout="centered")
st.title("Paraphrase Detection App")
st.markdown("Use different models to assess the semantic similarity between two sentences.")


# --- Model Loading (Cached) ---
# Use st.cache_resource to load large models only once, which is critical for performance.
@st.cache_resource
def load_models():
    st.info("🔄 Loading all models...")
    
    # 1. TF-IDF
    vectorizer = TfidfVectorizer()
    
    # 2. SBERT (Universal Sentence Embeddings)
    sbert_model = SentenceTransformer('all-mpnet-base-v2', device='cpu')
    sbert_model.eval()
    
    # 3. BERT (Sequence Classification)
    bert_tokenizer = AutoTokenizer.from_pretrained('textattack/bert-base-uncased-MRPC')
    bert_model = AutoModelForSequenceClassification.from_pretrained('textattack/bert-base-uncased-MRPC')
    
    return vectorizer, sbert_model, bert_tokenizer, bert_model

# Load the models using the cached function
vectorizer, sbert_model, bert_tokenizer, bert_model = load_models()
st.success("✅ Models loaded successfully!")


# --- Input Section ---
st.header("Input Sentences")
sent1 = st.text_area("Sentence 1", placeholder="Enter the first sentence here...", height=100)
sent2 = st.text_area("Sentence 2", placeholder="Enter the second sentence here...", height=100)

model_choice = st.selectbox(
    "Choose a Model for Prediction", 
    [ "TF-IDF + Cosine", "BERT", "SBERT"]
)

# --- Run on button click ---
if st.button("Check Paraphrase", type="primary"):
    if not sent1 or not sent2:
        st.error("Please enter text in both Sentence 1 and Sentence 2 fields.")
        st.stop()
        
    st.subheader(f"Results using {model_choice}:")
    
    # --- 1. TF-IDF + Cosine ---
    if model_choice == "TF-IDF + Cosine":
        # Fit on both sentences and transform
        vectors = vectorizer.fit_transform([sent1, sent2])
        # Calculate similarity of the two vectors
        sim = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
        
        st.metric(label="Cosine Similarity Score", value=f"{sim:.4f}")
        
        if sim >= TFIDF_THRESHOLD:
            st.success(f"✅ Paraphrased (Score > {TFIDF_THRESHOLD})")
        else:
            st.warning(f"❌ Not Paraphrased (Score ≤ {TFIDF_THRESHOLD})")


    # --- 2. SBERT ---
    elif model_choice == "SBERT":
        # Encode sentences into dense vectors
        with st.spinner('Encoding sentences...'):
            emb1 = sbert_model.encode(sent1, convert_to_tensor=True)
            emb2 = sbert_model.encode(sent2, convert_to_tensor=True)

        # Calculate cosine similarity using PyTorch/NumPy
        sim = torch.nn.functional.cosine_similarity(emb1, emb2, dim=0).item()
        
        st.metric(label="Cosine Similarity Score (Semantic)", value=f"{sim:.4f}")
        
        if sim >= SBERT_THRESHOLD:
            st.success(f"✅ Paraphrased (Score > {SBERT_THRESHOLD})")
        else:
            st.warning(f"❌ Not Paraphrased (Score ≤ {SBERT_THRESHOLD})")


    # --- 3. BERT (Classification) ---
    elif model_choice == "BERT":
        with st.spinner('Running BERT...'):
            # 1. Tokenize the sentence pair
            inputs = bert_tokenizer(
                sent1, 
                sent2, 
                return_tensors="pt", 
                padding=True, 
                truncation=True
            )

            # 2. Run inference
            with torch.no_grad():
                outputs = bert_model(**inputs)
            
            # 3. Get predicted class and confidence
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1).squeeze()
            
            # predicted_class_id: 1 is Paraphrase, 0 is Not Paraphrase
            predicted_class_id = torch.argmax(logits, dim=1).item()
            
            # Get the confidence score for the predicted class
            confidence = probabilities[predicted_class_id].item()

            
            # Determine the final result based on the model's output
            if predicted_class_id == 1:
                result_text = "✅ Paraphrased"
                st.metric(label="Prediction", value=result_text)
                st.success(f"Confidence in 'Paraphrased': {confidence:.4f}")
            else:
                result_text = "❌ Not Paraphrased"
                st.metric(label="Prediction", value=result_text)
                st.warning(f"Confidence in 'Not Paraphrased': {confidence:.4f}")




