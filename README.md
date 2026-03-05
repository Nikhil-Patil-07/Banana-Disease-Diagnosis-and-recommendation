# 🍌 AI-Based Crop Disease Diagnosis and Recommendation System
### Multilingual Support for Banana Farmers in Jalgaon District


## 📌 Overview

This project is a hybrid AI system designed to detect banana leaf diseases and provide actionable treatment recommendations to farmers in **Marathi, Hindi, and English**. It combines **computer vision** and **natural language processing** to support both image-based and text query-based diagnosis.

The system targets three major fungal diseases affecting banana crops in Jalgaon, Maharashtra — India's "Banana Capital":
- **Cordana** (*Cordana musae*)
- **Sigatoka** (*Pseudocercospora fijiensis / musae*)
- **Pestalotiopsis** (*Pestalotiopsis microspora*)

---

## 🎯 Key Features

- 📷 **Image-based disease detection** — upload a leaf photo and get an instant diagnosis
- 💬 **Multilingual text queries** — ask symptom questions in Marathi, Hindi, or English
- 🌐 **Multilingual recommendations** — pesticide and management advice in all three languages
- 🧠 **Hybrid AI pipeline** — MobileNetV2 + CLIP for vision; Sentence Transformers + Cross-Encoder for NLP
- 🗃️ **Structured knowledge base** — disease info, pathogens, pesticides, and IPM practices
- 🚀 **Deployed on Hugging Face Spaces** via Gradio

---


## 🗂️ Dataset

**BananaLSD (Banana Leaf Spot Diseases Dataset)**  
Collected at Bangabandhu Sheikh Mujibur Rahman Agricultural University, Bangladesh.

| Class | Original Images | Augmented | Total |
|---|---|---|---|
| Healthy | 129 | 1,371 | 2,000 |
| Sigatoka | 473 | 1,027 | 2,000 |
| Cordana | 162 | 1,338 | 2,000 |
| Pestalotiopsis | 173 | 1,327 | 2,000 |
| **Total** | **937** | **5,063** | **6,000** |

Images resized to **224×224 pixels**. Augmentations include rotation, flipping, cropping, brightness/contrast adjustment, gamma correction, and hue-saturation shifting.

- 📦 Dataset: [BananaLSD on Kaggle / Data in Brief](https://doi.org/10.1016/j.dib.2023.109608)

---

## 🏗️ System Architecture

The system has two parallel pipelines:

### 🔍 Vision Pipeline
```
Input Image
    └─► YOLOv8-L (Leaf Detection & ROI Localization)
            └─► CLIP ViT-B/32 (Banana vs. Non-Banana Verification)
                    └─► MobileNetV2 (Disease Classification)
                            └─► Knowledge Base Lookup → Multilingual Output
```

### 💬 NLP Pipeline
```
User Query (Mr / Hi / En)
    └─► Language Detection
            └─► intfloat/multilingual-e5-large (Semantic Embedding)
                    ├─► BM25 (Keyword Matching Score)
                    ├─► Isolation Forest (Outlier Detection)
                    └─► jinaai/jina-reranker-v2-base-multilingual (Cross-Encoder Reranking)
                                └─► Threshold Sweep → Knowledge Base → Multilingual Output
```

---

## 🧠 Models Used

| Component | Model | Purpose |
|---|---|---|
| Leaf Detection | YOLOv8-L | Localize leaf ROIs in input images |
| Leaf Verification | CLIP ViT-B/32 | Filter non-banana inputs |
| Disease Classifier | MobileNetV2 (fine-tuned) | Classify disease from leaf image |
| Semantic Embedder | intfloat/multilingual-e5-large | Multilingual query embeddings |
| Lexical Retriever | BM25 | Keyword-based knowledge base matching |
| Reranker | jinaai/jina-reranker-v2-base-multilingual | Fine-grained relevance scoring |
| Outlier Detection | Isolation Forest | Reject out-of-scope queries |

---

## 📊 Results

### Vision Model (MobileNetV2 + CLIP)

| Metric | Training | Testing |
|---|---|---|
| Accuracy | 99.81% | **99.00%** |
| Macro Precision | 0.9981 | 0.9900 |
| Macro Recall | 0.9981 | 0.9900 |
| Macro F1-Score | 0.9981 | **0.9900** |

**Per-class Test F1-Scores:**

| Disease | Precision | Recall | F1-Score |
|---|---|---|---|
| Cordana | 0.9933 | 0.9933 | 0.9933 |
| Healthy | 0.9868 | 0.9933 | 0.9900 |
| Pestalotiopsis | 0.9932 | 0.9767 | **0.9849** |
| Sigatoka | 0.9868 | 0.9967 | 0.9917 |

### NLP Pipeline

| Metric | Value |
|---|---|
| Overall Accuracy | 77.00% |
| Macro F1-Score | 0.75 |
| Best Threshold | 0.10 |

**Per-class NLP F1-Scores:**

| Class | Precision | Recall | F1-Score |
|---|---|---|---|
| Cordana | 0.89 | 0.84 | 0.86 |
| Healthy | 0.94 | 0.73 | 0.82 |
| Pestalotiopsis | 0.72 | 0.73 | 0.72 |
| Sigatoka | 0.60 | 0.80 | 0.69 |
| Unknown | 0.70 | 0.62 | 0.66 |


---

## 🛠️ Tech Stack

- **Deep Learning:** TensorFlow / Keras, PyTorch
- **Vision Models:** MobileNetV2, CLIP (OpenAI), YOLOv8 (Ultralytics)
- **NLP:** Sentence Transformers, BM25, Jina AI Reranker
- **Augmentation:** Albumentations
- **Deployment:** Hugging Face Spaces (Gradio)
- **Knowledge Base:** JSON / structured relational schema

---

## ⚙️ Model Configuration (Key Parameters)

### CNN Classifier (MobileNetV2)
```
- Input: (224, 224, 3), pretrained on ImageNet
- Phase 1: All base layers frozen, classification head trained
- Phase 2: Last 40 layers unfrozen for fine-tuning
- Head: GAP → Dense(128, relu) → Dropout(0.3) → Dense(64, relu) → Dropout(0.3) → Softmax
- Optimizer: Adam(lr=1e-4)
- Loss: sparse_categorical_crossentropy
- Callbacks: EarlyStopping(patience=5), ModelCheckpoint
```

### YOLOv8
```
- Weights: yolov8l.pt (pretrained)
- Used for banana-leaf ROI localization
- NMS applied; multi-leaf crops padded by PAD_RATIO
```

### Isolation Forest (Outlier Detection)
```
- contamination = 0.1
- random_state = 42
- Trained on KB symptom embeddings
```

---

## 🌐 Deployment

The system is deployed as an interactive web application on **Hugging Face Spaces** using **Gradio**, supporting:
- Image-only mode
- Text/query-only mode
- Combined (both) mode

🔗 **Live Demo:** (https://huggingface.co/spaces/NikhilPatil/Banana_Disease_Preediction/tree/main)

## 📝Download the train and saved models from the hugging face from the Main_py folder

