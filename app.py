import streamlit as st
import cv2
import numpy as np
import os
import json
import joblib
import pickle
from typing import Dict, Any
from sentence_transformers import SentenceTransformer, CrossEncoder
from langdetect import detect
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from vit_keras.layers import ClassToken, AddPositionEmbs, TransformerBlock


# ================== 🔁 CACHING HEAVY LOADS ==================

@st.cache_resource
def load_all_models():
    # CNN & ViT
    cnn_model = load_model("D:/banana_disease_app/Main_py/banana_cnn_model.keras", compile=False)
    vit_model = load_model(
        "D:/banana_disease_app/Main_py/banana_vit_model.keras",
        compile=False,
        custom_objects={
            'ClassToken': ClassToken,
            'AddPositionEmbs': AddPositionEmbs,
            'TransformerBlock': TransformerBlock
        }
    )

    # Extractors
    cnn_feat_ext = Model(inputs=cnn_model.input, outputs=cnn_model.get_layer(index=-4).output)
    vit_feat_ext = Model(inputs=vit_model.input, outputs=vit_model.get_layer(index=-4).output)

    return cnn_model, vit_model, cnn_feat_ext, vit_feat_ext

@st.cache_resource
def load_all_assets():
    # Load ML models
    scaler = joblib.load("D:/banana_disease_app/Main_py/feature_scaler.pkl")
    mlp_model = joblib.load("D:/banana_disease_app/Main_py/lightgbm_model.pkl")
    outlier_detector = joblib.load("D:/banana_disease_app/Main_py/isolation_forest.pkl")

    # Label encoder
    with open("D:/banana_disease_app/Main_py/label_encoder.pkl", "rb") as f:
        le = pickle.load(f)

    # Knowledge bases
    with open("D:/banana_disease_app/Main_py/banana_disease_knowledge_base_DL.json", "r", encoding="utf-8") as f:
        kb_data_image = {entry["Disease"]: entry for entry in json.load(f)}
    with open("D:/banana_disease_app/Main_py/banana_disease_knowledge_base.json", "r", encoding="utf-8") as f:
        kb_data_text = json.load(f)

    return scaler, mlp_model, le, kb_data_image, kb_data_text, outlier_detector

@st.cache_resource
def load_nlp_models():
    embedder = SentenceTransformer('sentence-transformers/paraphrase-xlm-r-multilingual-v1')
    cross_encoder = CrossEncoder('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1')
    return embedder, cross_encoder


# ================== 🧠 IMAGE PREDICTION ==================

def identify_disease_from_image(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("❌ Cannot load image")

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (224, 224))

        cnn_input = np.expand_dims(img_resized / 255.0, axis=0)
        vit_input = np.expand_dims(mobilenet_preprocess(img_resized), axis=0)

        cnn_feat = cnn_feature_extractor.predict(cnn_input, verbose=0)
        vit_feat = vit_feature_extractor.predict(vit_input, verbose=0)
        combined_feat = np.concatenate([cnn_feat, vit_feat], axis=1)
        combined_scaled = scaler.transform(combined_feat)

        y_pred = mlp_model.predict(combined_scaled)
        predicted_label = le.inverse_transform(y_pred)[0]

        confidence = None
        try:
            probs = mlp_model.predict_proba(combined_scaled)
            confidence = np.max(probs)
        except:
            probs = None

        st.image(img_rgb, caption="Uploaded Image", use_column_width=True)
        st.write(f"**Predicted Disease**: {predicted_label} ({confidence:.2f} confidence)" if confidence else predicted_label)

        result = {
            "predicted_disease": predicted_label,
            "confidence": confidence,
            "info_available": False,
            "all_probabilities": probs[0].tolist() if probs is not None else None
        }

        normalized_pred = predicted_label.lower().replace(" ", "")
        for disease in kb_data_image:
            if normalized_pred in disease.lower().replace(" ", ""):
                matched = kb_data_image[disease]
                result["info_available"] = True
                st.subheader("Image-Based Prediction (Marathi)")
                st.write(f"**रोग**: {matched['Local_Name']['mr']}")
                st.write(f"**लक्षणे**: {matched['Symptoms_MR']}")
                st.write(f"**कारण**: {matched['Cause_MR']}")
                st.write(f"**कीटकनाशक शिफारस**: {matched['Pesticide_Recommendation_MR']}")
                st.write(f"**कीटकनाशक**: {matched['Pesticide']}")
                st.write(f"**परजीवी**: {matched['Pathogen']}")
                st.write(f"**व्यवस्थापन उपाय**: {matched['Management_Practices_MR']}")
                break
        else:
            st.warning("❌ Disease not found in knowledge base.")

        return result

    except Exception as e:
        st.error(f"❌ Error: {e}")
        return {"error": str(e), "predicted_disease": None}


# ================== 🧠 TEXT PREDICTION ==================

def detect_language(query: str) -> str:
    try:
        lang = detect(query)
        return lang if lang in ["mr", "hi"] else "en"
    except:
        return "en"

def predict_disease(query: str) -> Dict[str, Any]:
    lang = detect_language(query)
    query_emb = embedder.encode([query], normalize_embeddings=True)

    symptom_key = f"Symptoms_{lang.upper()}" if lang != "en" else "Symptoms"
    pairs = [[query, entry[symptom_key]] for entry in kb_data_text]
    scores = cross_encoder.predict(pairs)

    best_idx = np.argmax(scores)
    if scores[best_idx] < 0.2:
        return {
            "message": {
                "mr": "हा रोग आमच्या डेटाबेसमध्ये नाही.",
                "hi": "यह रोग हमारे डेटाबेस में नहीं है।",
                "en": "This disease is not in our database."
            }[lang]
        }

    entry = kb_data_text[best_idx]
    return {
        "Crop": entry["Crop"],
        "Disease": entry["Local_Name"][lang],
        "Symptoms": entry[symptom_key],
        "Cause": entry[f"Cause_{lang.upper()}" if lang != "en" else "Cause"],
        "Pesticide_Recommendation": entry[f"Pesticide_Recommendation_{lang.upper()}" if lang != "en" else "Pesticide_Recommendation"],
        "Pesticide": entry["Pesticide"],
        "Pathogen": entry["Pathogen"],
        "Management_Practices": entry[f"Management_{lang.upper()}" if lang != "en" else "Management_Practices"]
    }


# ================== 🌿 STREAMLIT APP ==================

st.set_page_config(page_title="Banana Disease Detection", layout="centered")

st.title("🍌 Banana Disease Detection App")
st.write("Detect diseases from either an image or a symptom query (Marathi, Hindi, or English).")

option = st.radio("Choose method:", ("Image Only", "Text Only", "Both"))

# Load everything once
cnn_model, vit_model, cnn_feature_extractor, vit_feature_extractor = load_all_models()
scaler, mlp_model, le, kb_data_image, kb_data_text, outlier_detector = load_all_assets()
embedder, cross_encoder = load_nlp_models()

# Image
if option in ["Image Only", "Both"]:
    st.subheader("📷 Upload Image")
    uploaded_image = st.file_uploader("Choose a banana leaf image", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        temp_path = "temp_image.jpg"
        with open(temp_path, "wb") as f:
            f.write(uploaded_image.getbuffer())
        identify_disease_from_image(temp_path)
        os.remove(temp_path)

# Text
if option in ["Text Only", "Both"]:
    st.subheader("📝 Enter Symptoms")
    symptoms = st.text_area("Describe symptoms (in Marathi, Hindi, or English):")
    if symptoms and st.button("Predict Disease from Text"):
        result = predict_disease(symptoms)
        if "message" in result:
            st.warning(result["message"])
        else:
            st.subheader("Text-Based Prediction")
            st.write(f"**Crop**: {result['Crop']}")
            st.write(f"**Disease**: {result['Disease']}")
            st.write(f"**Symptoms**: {result['Symptoms']}")
            st.write(f"**Cause**: {result['Cause']}")
            st.write(f"**Pesticide Recommendation**: {result['Pesticide_Recommendation']}")
            st.write(f"**Pesticide**: {result['Pesticide']}")
            st.write(f"**Pathogen**: {result['Pathogen']}")
            st.write(f"**Management Practices**: {result['Management_Practices']}")
