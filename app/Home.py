import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import re
from transformers import AutoModel, AutoTokenizer
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file  # Untuk load safetensors
from datetime import datetime
import os

# ==============================================================================
# 1. KONFIGURASI GLOBAL & KONSTANTA
# ==============================================================================

# ⚠️ GANTI DENGAN REPO HUGGING FACE KAMU
HF_MODEL_REPO = "minhajulkhairi/absa-mipa-unhas"

# Konfigurasi Model
NUM_TOTAL_LABELS = 12
MAX_SEQ_LEN = 128
NUM_SENTIMENT_CLASSES = 3
SENTIMENT_LABELS = ['Negatif', 'Netral', 'Positif']
ASPECT_COLS = ['Layanan_Akademik', 'Layanan_Penunjang', 'Fasilitas_Inti', 'Sarana_Penunjang']

# Path untuk data
KAMUS_ALAY_FILE_PATH = "data/kamus_alay.csv"
LOG_DATA_PATH = "dynamic_prediction_log.csv"

# Metrik Model (dari hasil penelitian Persona-Based Full+BCE)
MODEL_METRICS = {
    'synthetic_test': {
        'macro_f1': 0.8868,
        'subset_accuracy': 0.7211
    },
    'real_test': {
        'macro_f1': 0.7549,
        'subset_accuracy': 0.4717
    },
    'gap': '14.87%'
}

# ==============================================================================
# 2. PAGE CONFIG & CSS
# ==============================================================================

st.set_page_config(
    page_title="Sistem Penjaminan Mutu - ABSA MIPA",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* 1. KUNCI SIDEBAR */
    section[data-testid="stSidebar"] { width: 300px !important; }
    div[data-testid="stSidebar"] + div { display: none; }
    [data-testid="stSidebar"] img { display: block; margin: 0 auto; max-width: 100%; height: auto; }

    /* 2. WHITE THEME */
    .stApp { background-color: #ffffff; }
    h1, h2, h3, h4, h5, h6 { color: #0f172a !important; font-family: 'Segoe UI', sans-serif; font-weight: 700; }
    p, div, label, span { color: #334155; }
    
    /* Hero Section */
    .hero-box {
        background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%);
        padding: 3rem 2rem; border-radius: 16px; color: white !important;
        text-align: center; margin-bottom: 2rem;
        box-shadow: 0 10px 25px rgba(37, 99, 235, 0.2);
    }
    .hero-title { font-size: 2.5rem; font-weight: 800; margin-bottom: 0.5rem; color: white !important; }
    .hero-subtitle { font-size: 1.2rem; opacity: 0.9; font-weight: 400; color: #e0e7ff !important; }
    
    /* KPI Cards */
    .kpi-card {
        background: white; padding: 1.5rem; border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05); text-align: center;
        border: 1px solid #e2e8f0; border-top: 5px solid #2563eb;
    }
    .kpi-value { font-size: 2.2rem; font-weight: 800; color: #1e293b; }
    .kpi-label { font-size: 0.85rem; color: #64748b; text-transform: uppercase; letter-spacing: 1px; font-weight: 600; }
    
    /* Features */
    .feature-card {
        background: #f8fafc; padding: 1.5rem; border-radius: 12px;
        border: 1px solid #e2e8f0; height: 100%; transition: transform 0.2s;
    }
    .feature-card:hover { transform: translateY(-5px); border-color: #3b82f6; background: white; box-shadow: 0 10px 15px -3px rgba(0,0,0,0.1); }
    .feature-icon { font-size: 2.5rem; margin-bottom: 1rem; }
    
    /* Sidebar */
    [data-testid="stSidebar"] { background-color: #f1f5f9; border-right: 1px solid #e2e8f0; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 3. MODEL CLASS (Harus sama dengan saat training)
# ==============================================================================

class IndoBERTForABSA(nn.Module):
    """Custom ABSA Model - arsitektur harus IDENTIK dengan saat training"""
    def __init__(self, num_labels=12):
        super().__init__()
        self.bert = AutoModel.from_pretrained("indobenchmark/indobert-base-p2")
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.bert(
            input_ids=input_ids, 
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        pooled = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits

# ==============================================================================
# 4. UTILITY FUNCTIONS
# ==============================================================================

@st.cache_resource
def load_normalization_map():
    """Load kamus normalisasi kata alay/slang"""
    DEFAULT_MAP = {
        'yg': 'yang', 'bgt': 'banget', 'gak': 'tidak', 'ga': 'tidak', 'tp': 'tapi',
        'krn': 'karena', 'dgn': 'dengan', 'sy': 'saya', 'tdk': 'tidak', 'utk': 'untuk',
        'mantap': 'bagus', 'oke': 'baik', 'jelek': 'buruk', 'parah': 'buruk',
        'gw': 'saya', 'gue': 'saya', 'lu': 'kamu', 'lo': 'kamu',
        'bgs': 'bagus', 'jlk': 'jelek', 'lmyn': 'lumayan', 'bngt': 'banget',
        'gmn': 'bagaimana', 'kyk': 'seperti', 'gitu': 'begitu', 'gini': 'begini',
        'makasih': 'terima kasih', 'thx': 'terima kasih', 'thanks': 'terima kasih'
    }
    try:
        if os.path.exists(KAMUS_ALAY_FILE_PATH):
            df = pd.read_csv(KAMUS_ALAY_FILE_PATH)
            custom_map = dict(zip(df['slang'], df['formal']))
            DEFAULT_MAP.update(custom_map)
        return DEFAULT_MAP
    except:
        return DEFAULT_MAP

NORMALIZATION_MAP = load_normalization_map()

def normalize_text_preserve_punctuation(text):
    """Normalisasi teks dengan mempertahankan tanda baca penting"""
    if pd.isna(text): 
        return ""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    words = text.split()
    normalized = [NORMALIZATION_MAP.get(w, w) for w in words]
    return " ".join(normalized)

def clean_for_model(text):
    """Bersihkan teks untuk input model"""
    text = normalize_text_preserve_punctuation(text)
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def detect_error_patterns(raw_text):
    """Deteksi pola linguistik yang berpotensi menyebabkan error prediksi"""
    text = normalize_text_preserve_punctuation(raw_text)
    patterns = {
        'mixed_sentiment': r'\b(tapi|namun|padahal|walaupun|meskipun|sayangnya)\b',
        'sarcasm': r'(wk{2,}|ha{2,}|he{2,}|lol|mantap\s*betul|luar\s*biasa)(?=.*[!]{2,})|\b(dingin\s*banget|panas\s*banget)\b',
        'negation': r'\b(tidak|bukan|gak|ga|jangan)\s+(bagus|baik|ramah|jelas|bersih|dingin)\b',
        'implicit_critique': r'\b(harusnya|seharusnya|cuma|hanya|sayang|agak)\b'
    }
    detected = []
    for name, regex in patterns.items():
        if re.search(regex, text, re.IGNORECASE):
            detected.append(name)
    return detected

def get_pattern_message(name):
    """Dapatkan pesan deskriptif untuk pola error"""
    msgs = {
        'mixed_sentiment': 'Indikasi kalimat pertentangan (Mixed Sentiment)',
        'sarcasm': 'Indikasi gaya bahasa sarkasme/informal',
        'negation': 'Indikasi negasi (pembalikan makna)',
        'implicit_critique': 'Indikasi kritik implisit/saran'
    }
    return msgs.get(name, name)

def load_dynamic_log():
    """Load log prediksi dari CSV"""
    try:
        if os.path.exists(LOG_DATA_PATH):
            return pd.read_csv(LOG_DATA_PATH)
        return pd.DataFrame(columns=['timestamp', 'text', 'aspect', 'prediction', 'probability', 'status'])
    except:
        return pd.DataFrame(columns=['timestamp', 'text', 'aspect', 'prediction', 'probability', 'status'])

def append_to_log(text, results, status):
    """Menambahkan log prediksi baru"""
    try:
        df = load_dynamic_log()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        new_rows = []
        for r in results:
            new_rows.append({
                'timestamp': timestamp,
                'text': text,
                'aspect': r['Aspek'],
                'prediction': r['Sentimen'],
                'probability': r['Probabilitas'],
                'status': status
            })
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
        df.to_csv(LOG_DATA_PATH, index=False)
    except Exception as e:
        st.warning(f"⚠️ Log tidak tersimpan: {e}")

def update_log_entry(text, corrected_results):
    """Update log dengan hasil koreksi manual"""
    try:
        if not os.path.exists(LOG_DATA_PATH):
            return False, "File log tidak ditemukan"
            
        df = pd.read_csv(LOG_DATA_PATH)
        mask = df['text'] == text
        
        if mask.any():
            for idx in df[mask].index:
                aspect = df.at[idx, 'aspect']
                new_sent = corrected_results.get(aspect) or corrected_results.get(aspect.replace('_', ' '))
                if new_sent:
                    df.at[idx, 'prediction'] = new_sent
                    df.at[idx, 'status'] = "VERIFIED (MANUAL)"
            df.to_csv(LOG_DATA_PATH, index=False)
            return True, "Sukses"
        else:
            return False, "Data tidak ditemukan di log"
    except Exception as e:
        return False, str(e)

# ==============================================================================
# 5. MODEL LOADING (DARI HUGGING FACE HUB - SAFETENSORS)
# ==============================================================================

@st.cache_resource
def load_model_and_tokenizer():
    """
    Load model dan tokenizer dari Hugging Face Hub.
    Mendukung format safetensors dan pytorch_model.bin
    """
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained("indobenchmark/indobert-base-p2")
        
        # Inisialisasi model dengan arsitektur yang sama
        model = IndoBERTForABSA(num_labels=NUM_TOTAL_LABELS)
        
        # Coba download safetensors dulu, jika tidak ada coba .bin
        try:
            model_path = hf_hub_download(
                repo_id=HF_MODEL_REPO,
                filename="model.safetensors"
            )
            # Load safetensors
            state_dict = load_file(model_path)
            model.load_state_dict(state_dict)
            print("✅ Loaded model.safetensors")
        except:
            # Fallback ke pytorch_model.bin
            model_path = hf_hub_download(
                repo_id=HF_MODEL_REPO,
                filename="pytorch_model.bin"
            )
            state_dict = torch.load(model_path, map_location='cpu', weights_only=False)
            model.load_state_dict(state_dict)
            print("✅ Loaded pytorch_model.bin")
        
        # Setup device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()
        
        return tokenizer, model, device
        
    except Exception as e:
        st.error(f"❌ Gagal memuat model: {e}")
        return None, None, None

def predict_absa(text, tokenizer, model, device):
    """Prediksi sentiment per aspek"""
    clean_text = clean_for_model(text)
    
    inputs = tokenizer(
        clean_text, 
        return_tensors="pt", 
        padding='max_length', 
        truncation=True, 
        max_length=MAX_SEQ_LEN
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        # Model custom menggunakan forward langsung
        logits = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask']
        )
        probs = torch.sigmoid(logits).cpu().numpy()[0]
    
    probs_reshaped = probs.reshape(len(ASPECT_COLS), NUM_SENTIMENT_CLASSES)
    
    results = []
    for i, aspect in enumerate(ASPECT_COLS):
        p = probs_reshaped[i]
        label_idx = np.argmax(p)
        results.append({
            'Aspek': aspect,
            'Sentimen': SENTIMENT_LABELS[label_idx],
            'Probabilitas': float(p[label_idx])
        })
    return results

# ==============================================================================
# 6. SIDEBAR
# ==============================================================================

with st.sidebar:
    try:
        if os.path.exists("assets/logo_unhas.png"):
            st.image("assets/logo_unhas.png", use_container_width=True)
    except:
        pass
    
    st.markdown("<h3 style='text-align: center; color: #1e293b;'>Fakultas MIPA<br>Universitas Hasanuddin</h3>", unsafe_allow_html=True)
    st.markdown("---")
    st.info("""
    **Navigasi:**
    
    🤖 **Prediksi Ulasan**
    Analisis real-time ulasan.
    
    📊 **Evaluasi Model**
    Monitoring performa dan log data.
    """)
    st.markdown("---")
    
    # Model Status
    st.markdown("**Status Model:**")
    tokenizer, model, device = load_model_and_tokenizer()
    if model is not None:
        st.success(f"✅ Model aktif ({device})")
    else:
        st.error("❌ Model tidak tersedia")
    
    st.markdown("---")
    st.caption("Program Studi Sistem Informasi")
    st.caption("Universitas Hasanuddin")
    st.caption("© 2025 | Minhajul Yusri Khairi")

# ==============================================================================
# 7. MAIN CONTENT
# ==============================================================================

# Hero Section
st.markdown("""
<div class="hero-box">
    <div class="hero-title">Dashboard ABSA Multi-Label</div>
    <div class="hero-subtitle">Analisis Sentimen Berbasis Aspek untuk Evaluasi Layanan Fakultas</div>
</div>
""", unsafe_allow_html=True)

# KPI Cards
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-value">{MODEL_METRICS['synthetic_test']['macro_f1']*100:.1f}%</div>
        <div class="kpi-label">F1 Training (Sintetis)</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-value">{MODEL_METRICS['real_test']['macro_f1']*100:.2f}%</div>
        <div class="kpi-label">F1 Testing (Real)</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-value">{MODEL_METRICS['gap']}</div>
        <div class="kpi-label">Synthetic-to-Real Gap</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    log_df = load_dynamic_log()
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-value">{len(log_df)}</div>
        <div class="kpi-label">Total Prediksi</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Features Section
st.markdown("### ⚡ Fitur Utama")
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">🧠</div>
        <h4>IndoBERT Architecture</h4>
        <p>Model Transformer yang di-finetune dengan strategi <em>Persona-Based Prompting</em> untuk memahami konteks bahasa mahasiswa.</p>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">🛡️</div>
        <h4>Confidence Scoring</h4>
        <p><em>Confidence Scoring</em> dan <em>Pattern Detection</em> untuk mencegah kesalahan prediksi pada kasus sulit.</p>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">👤</div>
        <h4>Verifikasi Manual</h4>
        <p>Validasi manual untuk prediksi dengan confidence &lt;75%, menjamin akurasi laporan akhir.</p>
    </div>
    """, unsafe_allow_html=True)

