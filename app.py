import streamlit as st
import numpy as np
import joblib
import tempfile
import pandas as pd
from utils.feature_extraction import extract_features
from st_audiorec import st_audiorec

# ===== CONFIG =====
st.set_page_config(page_title="Voice Identification", page_icon="🎧", layout="centered")

# ===== CSS Custom =====
st.markdown("""
<style>
    body {
        background: #0f172a;
        color: #e2e8f0;
    }
    .main {
        background-color: #1e293b;
        border-radius: 16px;
        padding: 30px;
        box-shadow: 0px 4px 15px rgba(0,0,0,0.3);
    }
    h1, h2, h3, h4 {
        color: #38bdf8 !important;
    }
    .stButton>button {
        background-color: #38bdf8 !important;
        color: white !important;
        border-radius: 12px !important;
        border: none;
        padding: 8px 20px !important;
    }
    .stRadio>div {
        background: #334155;
        border-radius: 12px;
        padding: 10px;
    }
    .stAlert {
        border-radius: 12px !important;
    }
    .css-1q8dd3e {
        background-color: #1e293b !important;
    }
    .stExpander {
        background: #1e293b !important;
        border-radius: 10px !important;
    }
</style>
""", unsafe_allow_html=True)

# ===== HEADER =====
st.markdown("<h1 style='text-align:center;'>🎙️ Voice Identification System</h1>", unsafe_allow_html=True)
st.markdown("""
<p style='text-align:center; color:#cbd5e1;'>
Sistem ini mengidentifikasi suara <b>buka</b> atau <b>tutup</b> dari dua pengguna:  
<span style='color:#38bdf8'><b>user1</b></span> dan <span style='color:#38bdf8'><b>user2</b></span>.<br>
Silakan pilih metode input untuk mulai menganalisis suara Anda.
</p>
""", unsafe_allow_html=True)
st.markdown("---")

# ===== LOAD MODEL =====
try:
    model_user = joblib.load("models/user_model.pkl")
    model_status = joblib.load("models/status_model.pkl")
    feature_cols = joblib.load("models/feature_cols.pkl")
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.stop()

# ===== PILIH INPUT =====
st.subheader("🎧 Pilih Metode Input")
input_option = st.radio("", ["🎤 Rekam suara", "📁 Upload file .wav"], horizontal=True)

# ===== PROSES AUDIO =====
def process_audio(audio_path):
    import librosa
    import speech_recognition as sr

    features = extract_features(audio_path)
    if not features:
        st.error("Gagal mengekstraksi fitur dari suara.")
        return

    # ===== DETEKSI SILENCE =====
    y, sr_audio = librosa.load(audio_path, sr=None)
    rms = np.mean(librosa.feature.rms(y=y))
    duration = len(y) / sr_audio
    if duration < 0.1 or rms < 1e-4:
        st.error("Tidak ada suara yang terdeteksi. Silakan rekam ulang.")
        return

    feature_df = pd.DataFrame([features])
    X = feature_df[feature_cols].to_numpy().reshape(1, -1)
    st.info(f"📏 Jumlah fitur yang dikirim ke model: {X.shape[1]}")

    # ===== PREDIKSI =====
    user_pred_proba = model_user.predict_proba(X)
    status_pred_proba = model_status.predict_proba(X)
    user_pred = np.argmax(user_pred_proba)
    status_pred = np.argmax(status_pred_proba)
    user_confidence = np.max(user_pred_proba)
    status_confidence = np.max(status_pred_proba)

    user_label = f"user{user_pred + 1}"
    status_label = "buka" if status_pred == 0 else "tutup"

    # ===== SPEECH TO TEXT =====
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(audio_path) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data, language="id-ID")
            st.markdown("### 📝 Hasil Speech-to-Text")
            st.markdown(f"<div style='background:#0f172a;padding:10px;border-radius:10px;color:#e2e8f0;'>{text}</div>", unsafe_allow_html=True)

            text_lower = text.lower()
            if "buka" in text_lower:
                status_label = "buka"
            elif "tutup" in text_lower:
                status_label = "tutup"
    except Exception as e:
        st.warning(f"Teks tidak dapat dikenali: {e}")

    # ===== HASIL PREDIKSI =====
    st.markdown("### 📊 Hasil Prediksi")
    st.success(f"✅ **Prediksi:** {user_label.upper()} — {status_label.upper()}")
    st.write(f"**User:** {user_label} ({user_confidence*100:.1f}%)")
    st.write(f"**Status:** {status_label} ({status_confidence*100:.1f}%)")

    if status_label == "buka":
        st.markdown(f"🔊 Suara mirip **{user_label}** saat **membuka** sesuatu.")
    else:
        st.markdown(f"🔊 Suara mirip **{user_label}** saat **menutup** sesuatu.")

    # ===== FITUR =====
    with st.expander("📈 Lihat fitur yang diekstraksi"):
        st.dataframe(feature_df.T, use_container_width=True)

# ===== INPUT REKAM =====
if input_option == "🎤 Rekam suara":
    st.info("🎤 Klik tombol di bawah untuk merekam suara Anda.")
    audio_bytes = st_audiorec()
    if audio_bytes is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
            tmpfile.write(audio_bytes)
            audio_path = tmpfile.name
        st.audio(audio_path, format="audio/wav")
        process_audio(audio_path)

# ===== INPUT UPLOAD =====
elif input_option == "📁 Upload file .wav":
    st.info("📂 Upload file suara Anda dalam format .wav.")
    uploaded_file = st.file_uploader("Unggah file suara:", type=["wav"])
    if uploaded_file is not None:
        st.audio(uploaded_file, format="audio/wav")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
            tmpfile.write(uploaded_file.read())
            audio_path = tmpfile.name
        process_audio(audio_path)