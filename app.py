import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

# =========================
# IMPORTS
# =========================
import io
import base64
import json
import pickle
import difflib
from collections import defaultdict
from typing import Dict, Any, List
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import streamlit as st
import torch
from transformers import CLIPModel, CLIPTokenizer, CLIPFeatureExtractor
from ultralytics import YOLO
from sentence_transformers import SentenceTransformer, CrossEncoder
from langdetect import detect, detect_langs
from sklearn.ensemble import IsolationForest
import joblib
import pkg_resources

# Debug: Verify imports and library versions
try:
    from sentence_transformers import CrossEncoder
    st.write("sentence_transformers.CrossEncoder imported successfully")
    st.write(f"sentence-transformers version: {pkg_resources.get_distribution('sentence-transformers').version}")
except ImportError as e:
    st.error(f"Failed to import CrossEncoder from sentence_transformers: {str(e)}")
    st.stop()

try:
    import einops
    st.write(f"einops version: {pkg_resources.get_distribution('einops').version}")
except ImportError:
    st.error("einops package is missing. Please ensure 'einops==0.8.0' is installed via requirements.txt.")
    st.stop()

# Use tf_keras imports to match environment
from tf_keras.models import load_model
from tf_keras.preprocessing.image import img_to_array, load_img

# =========================
# PATHS
# =========================
save_dir = "Main_py"  # Adjusted for Hugging Face Spaces
os.makedirs(save_dir, exist_ok=True)
cnn_json_path = os.path.join(save_dir, "banana_disease_knowledge_base_updated_shuffled.json")
nlp_json_path = os.path.join(save_dir, "banana_disease_knowledge_base (1).json")
model_path = os.path.join(save_dir, "best_cnn_model_finetuned.keras")
label_path = os.path.join(save_dir, "label_encoder.pkl")
YOLO_WEIGHTS = "yolov8l.pt"

# =========================
# CONFIG
# =========================
YOLO_CONF = 0.1
YOLO_IOU = 0.7
PAD_RATIO = 0.05
CLIP_RATIO = 1.5
CLIP_BATCH = 16
CNN_SIZE = (224, 224)
CNN_BATCH = 32

CLIP_PROMPTS = [
    "a photo of a banana leaf",
    "a photo of something that is not a banana leaf"
]

# =========================
# DEVICE
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# LOAD MODELS & DATA
# =========================
@st.cache_resource
def load_cnn_clip_kb():
    try:
        model = load_model(model_path, compile=False)
        with open(label_path, "rb") as f:
            le = pickle.load(f)
        with open(cnn_json_path, "r", encoding="utf-8") as f:
            cnn_kb_data = json.load(f)
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        clip_fe = CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch32")
        clip_model.to(DEVICE)
        return model, le, cnn_kb_data, clip_model, clip_tokenizer, clip_fe
    except Exception as e:
        st.error(f"Error loading CNN/CLIP models or data: {str(e)}")
        return None, None, None, None, None, None

@st.cache_resource
def load_nlp_models():
    try:
        embedder = SentenceTransformer("intfloat/multilingual-e5-large", device=DEVICE)
        from sentence_transformers import CrossEncoder

        cross_encoder = CrossEncoder(
            "jinaai/jina-reranker-v2-base-multilingual",
            trust_remote_code=True,
            device=DEVICE,
            automodel_args={'torch_dtype': torch.float32}
        )

        outlier_detector = joblib.load(os.path.join(save_dir, "isolation_forest(1).pkl"))
        with open(nlp_json_path, "r", encoding="utf-8") as f:
            nlp_kb_data = json.load(f)
        return embedder, cross_encoder, outlier_detector, nlp_kb_data
    except Exception as e:
        st.error(f"Error loading NLP models or data: {str(e)}")
        return None, None, None, None

@st.cache_resource
def load_yolo_model():
    try:
        return YOLO(YOLO_WEIGHTS)
    except Exception as e:
        st.error(f"Error loading YOLO model: {str(e)}")
        return None

# Load resources
cnn_model, le, cnn_kb_data, clip_model, clip_tokenizer, clip_fe = load_cnn_clip_kb()
embedder, cross_encoder, outlier_detector, nlp_kb_data = load_nlp_models()
yolo_model = load_yolo_model()


# =========================
# HELPERS: BM25 (unused but included for completeness)
# =========================
class BM25:
    def __init__(self, corpus: List[str], k1: float = 1.5, b: float = 0.75):
        self.corpus = [doc.split() for doc in corpus]
        self.N = len(self.corpus)
        self.avgdl = sum(len(doc) for doc in self.corpus) / self.N if self.N > 0 else 0
        self.k1 = k1
        self.b = b
        self.df = {}
        for doc in self.corpus:
            for word in set(doc):
                self.df[word] = self.df.get(word, 0) + 1

    def get_scores(self, query: str) -> np.ndarray:
        query_terms = query.split()
        scores = np.zeros(self.N)
        for q in query_terms:
            if q not in self.df:
                continue
            idf = np.log(((self.N - self.df[q] + 0.5) / (self.df[q] + 0.5)) + 1)
            for i, doc in enumerate(self.corpus):
                tf = doc.count(q)
                denom = tf + self.k1 * (1 - self.b + self.b * (len(doc) / self.avgdl if self.avgdl > 0 else 0))
                scores[i] += idf * (tf * (self.k1 + 1)) / denom if denom != 0 else 0
        return scores

# =========================
# HELPERS: Image Processing (Unchanged)
# =========================
def preprocess_cnn_image(image: Image.Image):
    img = image.resize(CNN_SIZE)
    arr = img_to_array(img).astype("float32") / 255.0
    return arr

def classify_batch(pil_images):
    if not pil_images:
        return [], []
    batch = np.stack([preprocess_cnn_image(im) for im in pil_images], axis=0)
    preds = cnn_model.predict(batch, batch_size=CNN_BATCH, verbose=0)
    idxs = np.argmax(preds, axis=1)
    labels = le.inverse_transform(idxs)
    confs = preds[np.arange(len(idxs)), idxs]
    return labels.tolist(), confs.tolist()

def verify_with_clip_batch(pil_images):
    if not pil_images:
        return [], []
    accepted, scores = [], []
    text_inputs = clip_tokenizer(CLIP_PROMPTS, return_tensors="pt", padding=True).to(DEVICE)
    for i in range(0, len(pil_images), CLIP_BATCH):
        chunk = pil_images[i:i+CLIP_BATCH]
        img_inputs = clip_fe(images=chunk, return_tensors="pt")
        pixel_values = img_inputs["pixel_values"].to(DEVICE)
        with torch.no_grad():
            outputs = clip_model(pixel_values=pixel_values, input_ids=text_inputs["input_ids"], attention_mask=text_inputs.get("attention_mask", None))
            logits = outputs.logits_per_image
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        for row in probs:
            p_banana, p_not = float(row[0]), float(row[1])
            if p_banana >= p_not * CLIP_RATIO:
                accepted.append(True)
                scores.append(p_banana)
            else:
                accepted.append(False)
                scores.append(p_not)
    return accepted, scores

def get_marathi_info(predicted, conf=None):
    names = [e["Disease"].strip().lower() for e in cnn_kb_data]
    match = difflib.get_close_matches(predicted.strip().lower(), names, n=1, cutoff=0.5)
    if match:
        for e in cnn_kb_data:
            if e["Disease"].strip().lower() == match[0]:
                return {
                    "पिक": e.get("Crop", "केळी"),
                    "रोग": e.get("Local_Name", {}).get("mr", predicted),
                    "लक्षणे": e.get("Symptoms_MR", ""),
                    "कारण": e.get("Cause_MR", ""),
                    "किटकनाशके": e.get("Pesticide_MR", ""),
                    "किटकनाशक शिफारस": e.get("Pesticide_Recommendation", {}).get("mr", ""),
                    "नियंत्रण पद्धती": e.get("Management_MR", ""),
                    "रोगजन्य घटक": e.get("Pathogen", ""),
                    "विश्वासार्हता": f"{conf:.2%}" if conf is not None else "N/A"
                }
    return None

def crop_with_padding(image, box, pad_ratio=PAD_RATIO):
    w, h = image.size
    x1, y1, x2, y2, conf, cls = [float(v) for v in box[:6]]
    bw, bh = x2 - x1, y2 - y1
    pad_x, pad_y = bw * pad_ratio, bh * pad_ratio
    nx1, ny1 = max(int(x1 - pad_x), 0), max(int(y1 - pad_y), 0)
    nx2, ny2 = min(int(x2 + pad_x), w), min(int(y2 + pad_y), h)
    return image.crop((nx1, ny1, nx2, ny2))

# =========================
# MAIN PIPELINE: YOLO → CLIP → CNN (Unchanged)
# =========================
def predict_image_pipeline_bytes(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    results = yolo_model.predict(image, conf=YOLO_CONF, iou=YOLO_IOU, verbose=False)
    result = results[0]
    num_boxes = len(result.boxes)
    if num_boxes <= 1:
        candidates = [image]
    else:
        candidates = [
            crop_with_padding(image, box, pad_ratio=PAD_RATIO)
            for box in result.boxes.data
        ]
        if not candidates:
            candidates = [image]
    accepted_mask, clip_scores = verify_with_clip_batch(candidates)
    kept = [im for im, ok in zip(candidates, accepted_mask) if ok]
    kept_scores = [s for s, ok in zip(clip_scores, accepted_mask) if ok]
    if not kept:
        return {
            "is_banana": False,
            "kept_images": [],
            "labels": [],
            "confs": [],
            "clip_scores": []
        }
    labels, confs = classify_batch(kept)
    return {
        "is_banana": True,
        "kept_images": kept,
        "labels": labels,
        "confs": confs,
        "clip_scores": kept_scores
    }

# =========================
# TEXT PREDICTION (NLP)
# =========================
def detect_language(query: str) -> str:
    try:
        langs = detect_langs(query)
        if len(langs) > 1 and langs[0].prob < 0.95:
            return 'mr'
        return langs[0].lang if langs[0].lang in ['en', 'hi', 'mr'] else 'en'
    except:
        return 'en'

def predict_disease(query: str) -> Dict[str, Any]:
    lang = detect_language(query)
    query_emb = embedder.encode([query], normalize_embeddings=True)
    is_outlier = outlier_detector.predict(query_emb)[0] == -1

    symptom_pairs = []
    kb_index_map = []
    for i, entry in enumerate(nlp_kb_data):
        symptoms = entry.get("Symptoms", {}).get(lang, [])
        for s in symptoms:
            symptom_pairs.append([query, s])
            kb_index_map.append(i)

    # ✅ Patch: force float32 conversion
    raw_scores = cross_encoder.predict(symptom_pairs, convert_to_tensor=True)
    scores = raw_scores.to(torch.float32).cpu().numpy().tolist()

    best_idx = int(np.argmax(scores))
    best_score = float(scores[best_idx])
    threshold = 0.10

    if best_score < threshold:
        return {
            "message": {
                'mr': "हा रोग आमच्या डेटाबेसमध्ये नाही。",
                'hi': "यह रोग हमारे डेटाबेस में नहीं है।",
                'en': "This disease is not in our database."
            }[lang],
            "debug": {
                "is_outlier": is_outlier,
                "best_score": best_score,
                "query_embedding_norm": float(np.linalg.norm(query_emb)),
                "cross_encoder_scores": scores
            }
        }

    entry = nlp_kb_data[kb_index_map[best_idx]]
    response = {
        "Crop": entry.get("Crop", "Banana"),
        "Canonical_Label": entry.get("Canonical_Label", "unknown"),
        "Disease": entry.get("Disease", {}).get(lang, entry.get("Canonical_Label")),
        "Symptoms": entry.get("Symptoms", {}).get(lang, []),
        "Cause": entry.get("Cause", {}).get(lang, ""),
        "Pathogen": entry.get("Pathogen", "Unknown"),
        "Pesticide": entry.get("Pesticide", []),
        "Pesticide_Recommendation": entry.get("Pesticide_Recommendation", {}).get(lang, ""),
        "Management_Practices": entry.get("Management_Practices", {}).get(lang, ""),
        "debug": {
            "is_outlier": is_outlier,
            "best_score": best_score,
            "query_embedding_norm": float(np.linalg.norm(query_emb)),
            "cross_encoder_scores": scores
        }
    }
    return response



# =========================
# STREAMLIT UI
# =========================
st.set_page_config(page_title="🍌 Banana Disease Detection (YOLO+CLIP+CNN + NLP)", layout="centered")
st.title("केळीच्या पानांवरील रोगांचे निदान")
st.markdown("प्रतिमा किंवा लक्षणे वापरून केळीवरील रोगांचे निदान करा (मराठी, हिंदी, इंग्रजी भाषांमध्ये).")

option = st.radio("इनपुट पद्धत निवडा:", ["Image Only", "Text Only", "Both"])

# IMAGE FLOW (Unchanged)
if option in ["Image Only", "Both"]:
    st.subheader("प्रतिमा अपलोड करा")
    uploaded_img = st.file_uploader("JPG / PNG / JPEG", type=["png", "jpg", "jpeg"])
    if uploaded_img:
        image_bytes = uploaded_img.getvalue()
        st.info("YOLO → CLIP → CNN पाइपलाइन चालू आहे... (ही प्रक्रिया काही वेळ घेऊ शकते)")
        out = predict_image_pipeline_bytes(image_bytes)
        if not out["is_banana"]:
            st.error("🚫 हे केळीचे पान नाही. कृपया केळीचे पान अपलोड करा.")
        else:
            grouped = defaultdict(lambda: {"images": [], "confs": [], "clip_scores": []})
            for img, lab, conf, cs in zip(out["kept_images"], out["labels"], out["confs"], out["clip_scores"]):
                grouped[lab]["images"].append(img)
                grouped[lab]["confs"].append(conf)
                grouped[lab]["clip_scores"].append(cs)
            for disease, data in grouped.items():
                avg_conf = float(np.mean(data["confs"])) if data["confs"] else 0.0
                st.markdown(f"### रोग: **{disease}**  |  पाने: **{len(data['images'])}**  |  सरासरी विश्वासार्हता: **{avg_conf:.2%}**")
                info = get_marathi_info(disease, avg_conf)
                if info:
                    st.subheader("मराठी शिफारस:")
                    for k, v in info.items():
                        st.markdown(f"**{k}**: {v}")
                else:
                    st.warning("ज्ञानतळात रोगासाठी माहिती नाही.")
                max_cols = 4
                num_images = len(data["images"])
                for row_start in range(0, num_images, max_cols):
                    row_end = min(row_start + max_cols, num_images)
                    cols = st.columns(row_end - row_start)
                    for col_idx, i in enumerate(range(row_start, row_end)):
                        im = data["images"][i]
                        cols[col_idx].image(im, caption=f"{disease} ({data['confs'][i]:.2%})", width=150)

# TEXT FLOW
if option in ["Text Only", "Both"]:
    st.subheader("लक्षणे लिहा")
    symptoms = st.text_area("लक्षणे (मराठी / हिंदी / इंग्रजी):")
    if symptoms and st.button("रोग ओळखा"):
        result = predict_disease(symptoms)
        if "message" in result:
            st.warning(result["message"])
            st.subheader("Debug Info:")
            for k, v in result["debug"].items():
                st.markdown(f"**{k}**: {v}")
        else:
            st.subheader("शिफारस:")
            for k, v in result.items():
                if k != "debug":
                    st.markdown(f"**{k}**: {v}")
            st.subheader("Debug Info:")
            for k, v in result["debug"].items():
                st.markdown(f"**{k}**: {v}")
