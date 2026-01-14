import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from Home import load_dynamic_log, ASPECT_COLS, SENTIMENT_LABELS

st.set_page_config(page_title="Evaluasi Model - ABSA", page_icon="📊", layout="wide")

# CSS (TIDAK DIUBAH - UI BARU)
st.markdown("""
<style>
    section[data-testid="stSidebar"] { width: 300px !important; }
    div[data-testid="stSidebar"] + div { display: none; }
    [data-testid="stSidebar"] img { display: block; margin: 0 auto; }
</style>
""", unsafe_allow_html=True)

# Load Data
df = load_dynamic_log()

# Sidebar (TIDAK DIUBAH)
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
    Analisis real-time ulasan.
    
    📊 **Evaluasi Model**
    Monitoring performa dan log data.
    """)
    st.markdown("---")
    st.caption("Program Studi Sistem Informasi")
    st.caption("Universitas Hasanuddin")
    st.caption("© 2025 | Minhajul Yusri Khairi")

# Main Content
st.markdown("## 📊 Evaluasi & Monitoring Model")
st.caption("Statistik real-time dari penggunaan model di lingkungan produksi.")

if df.empty:
    st.warning("Belum ada data log prediksi. Silakan lakukan prediksi di menu 'Prediksi Ulasan' terlebih dahulu.")
    st.stop()

# ==============================================================================
# LOGIKA PERHITUNGAN METRIK (DIPERBAIKI)
# ==============================================================================

total_preds = len(df)
avg_conf_global = df['probability'].mean()

# Fix 1: Hitung Low Confidence murni (< 0.75)
low_conf_count = len(df[df['probability'] < 0.75])

# Fix 2: Hitung Risiko Sebenarnya (Menggunakan Status dari Logika Canggih)
# Risiko bukan cuma Low Confidence, tapi juga "HIGH RISK" (Sarkasme) dan "DANGER" (Konflik)
# Kita hitung semua yang statusnya TIDAK mengandung kata 'VERIFIED'
if 'status' in df.columns:
    risky_preds = df[~df['status'].astype(str).str.contains('VERIFIED', case=False, na=False)]
    risk_count = len(risky_preds)
else:
    risk_count = low_conf_count # Fallback jika kolom status belum ada

risk_ratio = (risk_count / total_preds) * 100 if total_preds > 0 else 0

# ==============================================================================
# TAMPILAN METRIK (UI BARU)
# ==============================================================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Prediksi", total_preds)
with col2:
    st.metric("Rata-rata Confidence", f"{avg_conf_global:.1%}")
with col3:
    st.metric("Prediksi Low Confidence", low_conf_count, delta_color="inverse")
with col4:
    # Menggunakan risk_ratio yang sudah diperbaiki logikanya
    st.metric("Rasio Risiko (Flagged)", f"{risk_ratio:.1f}%", delta_color="inverse", help="Persentase prediksi yang membutuhkan verifikasi manual (Low Conf + Sarkasme + Konflik)")

st.markdown("---")

# Tabs
tab1, tab2 = st.tabs(["📈 Distribusi Sentimen", "🛡️ Audit Confidence"])

with tab1:
    st.subheader("Sebaran Sentimen per Aspek")
    agg_df = df.groupby(['aspect', 'prediction']).size().reset_index(name='count')
    fig = px.bar(
        agg_df, x='aspect', y='count', color='prediction',
        title="Distribusi Sentimen",
        color_discrete_map={'Positif': '#10b981', 'Netral': '#94a3b8', 'Negatif': '#ef4444'},
        barmode='group'
    )
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', 
        plot_bgcolor='rgba(0,0,0,0)', 
        font=dict(color='#1e293b')
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("#### Detail Data")
    # Pastikan menampilkan kolom status agar user tahu kenapa data ditandai risiko
    cols_to_show = ['timestamp', 'text', 'aspect', 'prediction', 'probability', 'status']
    # Filter hanya kolom yang ada di df
    cols_to_show = [c for c in cols_to_show if c in df.columns]
    
    st.dataframe(
        df[cols_to_show].sort_values('timestamp', ascending=False),
        use_container_width=True
    )

with tab2:
    st.subheader("Analisis Keandalan Model")
    col_a, col_b = st.columns(2)
    
    with col_a:
        fig_hist = px.histogram(
            df, x="probability", nbins=20, 
            title="Distribusi Confidence Score",
            color_discrete_sequence=['#3b82f6']
        )
        fig_hist.add_vline(x=0.75, line_dash="dash", line_color="red", annotation_text="Threshold 0.75")
        fig_hist.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', 
            plot_bgcolor='rgba(0,0,0,0)', 
            font=dict(color='#1e293b')
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_b:
        # Visualisasi Status Pie Chart sangat penting untuk melihat proporsi Sarkasme/Konflik
        if 'status' in df.columns:
            status_counts = df['status'].value_counts().reset_index()
            status_counts.columns = ['status', 'count']
            
            # Mapping warna diperluas untuk mencakup status dari logika lama
            color_map = {
                'VERIFIED': '#10b981', 
                'VERIFIED (MANUAL)': '#3b82f6',
                'REVIEW NEEDED': '#f59e0b', 
                'CAUTION': '#f59e0b',       # Tambahan dari logika lama
                'HIGH RISK': '#ef4444',
                'DANGER (CONFLICT)': '#7f1d1d' # Tambahan dari logika lama
            }
            
            fig_pie = px.pie(
                status_counts, values='count', names='status',
                title="Proporsi Status Validasi",
                color='status',
                color_discrete_map=color_map
            )
            fig_pie.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', 
                plot_bgcolor='rgba(0,0,0,0)', 
                font=dict(color='#1e293b')
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("Data status belum tersedia.")

# Download
st.markdown("---")
st.subheader("📥 Ekspor Laporan")
csv = df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Download Log CSV",
    data=csv,
    file_name='absa_log_report.csv',
    mime='text/csv',
    key='download-csv'
)