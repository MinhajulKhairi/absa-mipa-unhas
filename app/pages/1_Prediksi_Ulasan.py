import streamlit as st
import pandas as pd
import time
import plotly.graph_objects as go
from datetime import datetime
from Home import (
    load_model_and_tokenizer, predict_absa, append_to_log, update_log_entry,
    detect_error_patterns, get_pattern_message,
    ASPECT_COLS
)

st.set_page_config(page_title="Prediksi & Validasi - ABSA", page_icon="🤖", layout="wide")

# ==============================================================================
# 1. CSS KHUSUS HALAMAN PREDIKSI
# ==============================================================================
st.markdown("""
<style>
    /* 1. KUNCI SIDEBAR (Wajib Dicopy ke setiap halaman) */
    section[data-testid="stSidebar"] { width: 300px !important; }
    div[data-testid="stSidebar"] + div { display: none; }
    [data-testid="stSidebar"] img { display: block; margin: 0 auto; }

    /* Input Area Styling */
    .stTextArea textarea { 
        font-size: 1.1rem; 
        background-color: #ffffff !important;
        border: 1px solid #cbd5e1;
        color: #0f172a !important;
    }
    
    /* Status Box */
    .status-box { padding: 1.5rem; border-radius: 10px; color: white; margin-bottom: 1.5rem; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    .status-green { background: linear-gradient(135deg, #10b981, #059669); }
    .status-yellow { background: linear-gradient(135deg, #f59e0b, #d97706); }
    .status-red { background: linear-gradient(135deg, #ef4444, #dc2626); }
    
    /* Result Card */
    .result-card {
        background: #ffffff; padding: 1.2rem; border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05); margin-bottom: 1rem;
        border: 1px solid #e2e8f0; border-left-width: 6px;
        transition: transform 0.2s;
    }
    .result-card:hover { transform: translateX(5px); border-color: #cbd5e1; }
    .pos { border-left-color: #10b981; } .neg { border-left-color: #ef4444; } .neu { border-left-color: #94a3b8; }
    
    /* Badge */
    .sent-badge { display: inline-block; padding: 0.25rem 0.75rem; border-radius: 20px; font-size: 0.85rem; font-weight: 700; text-transform: uppercase; }
    .badge-Positif { background-color: #dcfce7; color: #166534; }
    .badge-Negatif { background-color: #fee2e2; color: #991b1b; }
    .badge-Netral  { background-color: #f1f5f9; color: #475569; }
    .aspect-title { font-size: 0.9rem; color: #64748b; margin-bottom: 0.3rem; text-transform: uppercase; font-weight: 600; }
    .conf-label { font-size: 0.75rem; color: #94a3b8; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. LOGIKA VALIDASI
# ==============================================================================
def determine_status(min_conf, avg_conf, patterns):
    if 'sarcasm' in patterns or 'negation' in patterns:
        return 'HIGH RISK', 'red', '🔴'
    if min_conf < 0.75:
        return 'REVIEW NEEDED', 'yellow', '⚠️'
    return 'VERIFIED', 'green', '✅'

# ==============================================================================
# 3. SIDEBAR (KONSISTEN)
# ==============================================================================
with st.sidebar:
    try:
        st.image("assets/logo_unhas.png", use_container_width=True)
    except:
        pass
    st.markdown("<h3 style='text-align: center; color: #1e293b;'>Fakultas MIPA<br>Universitas Hasanuddin</h3>", unsafe_allow_html=True)
    st.markdown("---")
    st.info("""
    **Navigasi:**
    
    🤖 **Prediksi Ulasan**
    Simulasi analisis real-time dengan Human-in-the-Loop.
    
    📊 **Evaluasi Model**
    Monitoring performa dan log data.
    """)
    st.markdown("---")
    st.caption("Program Studi Sistem Informasi")
    st.caption("Universitas Hasanuddin")
    st.caption("© 2025 | Minhajul Yusri Khairi")

# ==============================================================================
# 4. CONTENT UTAMA
# ==============================================================================
st.markdown("## 🤖 Prediksi Ulasan")
st.caption("Masukkan ulasan untuk analisis sentimen real-time.")

col_input, col_info = st.columns([2, 1])

with col_input:
    text_input = st.text_area("Input Ulasan:", height=150, 
                             placeholder="Contoh: Dosennya enak ngajar tapi AC di kelas panas banget...",
                             key="input_ulasan") # Tambahkan Key agar state terjaga
    
    analyze = st.button("🔍 Analisis Sentimen", type="primary", use_container_width=True)

with col_info:
    st.info("**Panduan Validasi:**")
    st.markdown("""
    - ✅ **Verified:** Confidence > 75% & tidak ada pola ambigu.
    - ⚠️ **Review Needed:** Confidence < 75%.
    - 🔴 **High Risk:** Terdeteksi sarkasme/negasi kompleks.
    """)

# Session State untuk menyimpan hasil prediksi agar tidak hilang saat tombol verifikasi ditekan
if 'prediction_result' not in st.session_state:
    st.session_state['prediction_result'] = None

if analyze:
    if not text_input or len(text_input) < 10:
        st.warning("⚠️ Teks terlalu pendek. Masukkan minimal 10 karakter agar prediksi valid.")
        st.stop()

    tokenizer, model, device = load_model_and_tokenizer()
    
    if not model:
        st.error("Gagal memuat model.")
        st.stop()

    with st.spinner("Sedang memproses..."):
        time.sleep(0.5)
        patterns = detect_error_patterns(text_input)
        results = predict_absa(text_input, tokenizer, model, device)
        
        all_probs = [r['Probabilitas'] for r in results]
        avg_conf = sum(all_probs) / 4
        min_conf = min(all_probs)
        status_label, color_class, icon = determine_status(min_conf, avg_conf, patterns)
        
        # Simpan ke Log Awal
        append_to_log(text_input, results, status_label)
        
        # Simpan ke Session State
        st.session_state['prediction_result'] = {
            'text': text_input,
            'results': results,
            'patterns': patterns,
            'avg_conf': avg_conf,
            'min_conf': min_conf,
            'status_label': status_label,
            'color_class': color_class,
            'icon': icon
        }

# TAMPILKAN HASIL DARI SESSION STATE
if st.session_state['prediction_result']:
    data = st.session_state['prediction_result']
    
    st.markdown(f"""
    <div class="status-box status-{data['color_class']}">
        <h2 style="color:white; margin:0">{data['icon']} Status: {data['status_label']}</h2>
        <p style="margin-top:0.5rem; color:white;">Rata-rata Confidence: <strong>{data['avg_conf']*100:.1f}%</strong> (Min: {data['min_conf']*100:.1f}%)</p>
    </div>
    """, unsafe_allow_html=True)

    if data['patterns']:
        st.warning(f"⚠️ **Peringatan Linguistik:** Ditemukan pola {' & '.join([get_pattern_message(p) for p in data['patterns']])}")

    c1, c2 = st.columns([1, 1])
    
    with c1:
        st.markdown("### 📋 Hasil Prediksi")
        for res in data['results']:
            sent = res['Sentimen']
            conf_val = res['Probabilitas']
            border_cls = 'pos' if sent == 'Positif' else 'neg' if sent == 'Negatif' else 'neu'
            conf_color = "#166534" if conf_val >= 0.75 else "#b45309" if conf_val >= 0.6 else "#991b1b"
            
            st.markdown(f"""
            <div class="result-card {border_cls}">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div>
                        <div class="aspect-title">{res['Aspek'].replace('_',' ')}</div>
                        <span class="sent-badge badge-{sent}">{sent}</span>
                    </div>
                    <div style="text-align:right;">
                        <div style="font-size:1.5rem; font-weight:800; color:{conf_color}">{conf_val:.1%}</div>
                        <div class="conf-label">Confidence</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with c2:
        st.markdown("### 📈 Visualisasi Confidence")
        df_chart = pd.DataFrame(data['results'])
        fig = go.Figure(go.Bar(
            x=df_chart['Probabilitas'],
            y=df_chart['Aspek'].str.replace('_',' '),
            orientation='h',
            marker=dict(color=df_chart['Probabilitas'], colorscale='RdYlGn', cmin=0, cmax=1),
            text=df_chart['Sentimen'],
            textposition='auto',
            textfont=dict(color='white')
        ))
        fig.add_vline(x=0.75, line_dash="dash", line_color="red", annotation_text="Threshold 0.75")
        fig.update_layout(
            xaxis_title="Confidence Score", 
            margin=dict(l=0, r=0, t=0, b=0), 
            height=300,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#1e293b')
        )
        st.plotly_chart(fig, use_container_width=True)

    # ==========================================================================
    # 6. HUMAN-IN-THE-LOOP (DENGAN FEEDBACK YANG BENAR)
    # ==========================================================================
    if data['status_label'] in ['REVIEW NEEDED', 'HIGH RISK']:
        st.markdown("---")
        st.error("### 👤 Verifikasi Manual Diperlukan")
        
        with st.form("verification_form"):
            st.markdown("#### 📝 Formulir Koreksi Label")
            
            cols = st.columns(4)
            corrected_results = {}
            
            for idx, res in enumerate(data['results']):
                aspect_name = res['Aspek']
                current_sentiment = res['Sentimen']
                display_name = aspect_name.replace('_', ' ')
                
                with cols[idx]:
                    st.markdown(f"**{display_name}**")
                    corrected_sent = st.selectbox(
                        f"Koreksi {display_name}",
                        options=['Negatif', 'Netral', 'Positif'],
                        index=['Negatif', 'Netral', 'Positif'].index(current_sentiment),
                        key=f"fix_{idx}",
                        label_visibility="collapsed"
                    )
                    corrected_results[aspect_name] = corrected_sent
            
            st.markdown("---")
            catatan = st.text_input("Catatan Validator (Opsional)")
            submit_correction = st.form_submit_button("✅ Simpan Koreksi & Validasi", type="primary", use_container_width=True)

            if submit_correction:
                with st.spinner("Mengupdate database log..."):
                    # PANGGIL FUNGSI UPDATE DAN TANGKAP HASILNYA
                    success, message = update_log_entry(data['text'], corrected_results)
                    time.sleep(1.0)
                
                if success:
                    st.success(f"✅ Berhasil! Status berubah menjadi VERIFIED (MANUAL).")
                    # Tampilkan perubahan
                    changes = []
                    for res in data['results']:
                        asp = res['Aspek']
                        if res['Sentimen'] != corrected_results[asp]:
                            changes.append(f"{asp}: {res['Sentimen']} ➝ {corrected_results[asp]}")
                    
                    if changes:
                        st.info(f"Perubahan: {', '.join(changes)}")
                    
                    # Reset session state agar form bersih kembali
                    time.sleep(2)
                    st.session_state['prediction_result'] = None
                    st.rerun() # Refresh halaman
                else:
                    st.error(f"Gagal menyimpan: {message}")