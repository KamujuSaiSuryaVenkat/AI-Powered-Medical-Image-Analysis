import streamlit as st
import requests
import matplotlib.pyplot as plt
import datetime
from PIL import Image
import io
import numpy as np

# --- PAGE CONFIG ---
st.set_page_config(page_title="DocAI - Radiology Dashboard", page_icon="🩻", layout="wide")

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .main-header { font-size: 38px; font-weight: 800; color: #2E4053; }
    .sub-header { font-size: 16px; color: #7F8C8D; font-style: italic;}
    .alert-danger { background-color: #FDEDEC; padding: 15px; border-left: 5px solid #E74C3C; border-radius: 5px; color: #78281F;}
    .alert-success { background-color: #EAFAF1; padding: 15px; border-left: 5px solid #2ECC71; border-radius: 5px; color: #145A32;}
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">DocAI - Radiology Dashboard</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Powered by MobileNetV2 Deep Transfer Learning & Grad-CAM Interpretability</p>', unsafe_allow_html=True)
st.divider()

# --- SIDEBAR ---
with st.sidebar:
    st.header("📋 Patient Information")
    patient_id = st.text_input("Patient ID", value="PT-89421")
    patient_age = st.number_input("Age", min_value=1, max_value=120, value=45)
    patient_gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    
    st.divider()
    uploaded_file = st.file_uploader("Upload Chest X-Ray", type=["jpg", "png", "jpeg"])

# --- REPORT FUNCTION ---
def generate_html_report(prediction, confidence, probs, labels):
    date_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    theme_color = "#2ECC71" if prediction == "Normal" else "#E74C3C"
    bg_color = "#EAFAF1" if prediction == "Normal" else "#FDEDEC"

    analysis_text = f"AI detected features consistent with {prediction}. Clinical correlation recommended."

    warning_html = ""
    if confidence < 0.70:
        sorted_indices = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
        second_pred = labels[sorted_indices[1]]
        second_conf = probs[sorted_indices[1]] * 100

        warning_html = f"""
        <div style="background-color:#FEF9E7;padding:10px;border-left:4px solid #F1C40F;">
        ⚠️ Secondary possibility: <b>{second_pred}</b> ({second_conf:.1f}%)
        </div>
        """

    html_report = f"""
    <div style="padding:20px;border-top:5px solid #2E86C1;">
    <h3>📄 AI Radiology Report</h3>
    <p>Date: {date_str}</p>

    <p><b>Patient:</b> {patient_id} | {patient_age} yrs | {patient_gender}</p>

    <div style="background:{bg_color};padding:10px;border-left:5px solid {theme_color};">
    <b>{prediction}</b> ({confidence*100:.2f}%)
    </div>

    <p>{analysis_text}</p>
    {warning_html}
    </div>
    """

    text_report = f"{prediction} ({confidence*100:.2f}%)"

    return html_report, text_report

# --- MAIN ---
if uploaded_file:
    with st.spinner("Processing..."):
        try:
            files = {"file": uploaded_file.getvalue()}
            res = requests.post("http://127.0.0.1:8000/predict", files=files)
            result = res.json()

            prediction = result['prediction']
            confidence = result['confidence']
            probs_dict = result['all_probs']
            labels = list(probs_dict.keys())
            probs = list(probs_dict.values())

            # ALERT
            if prediction == "Normal":
                st.markdown(f'<div class="alert-success">✅ {prediction} ({confidence*100:.1f}%)</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="alert-danger">⚠️ {prediction} ({confidence*100:.1f}%)</div>', unsafe_allow_html=True)

            col1, col2, col3 = st.columns([1,1,1.2])

            # INPUT IMAGE
            with col1:
                st.subheader("📥 Input")
                st.image(uploaded_file, width="stretch")

            # 🔥 HEATMAP (UPDATED)
            with col2:
                st.subheader("🔥 Heatmap")
                heatmap = result["heatmap"]

                if isinstance(heatmap, list):
                    heatmap = np.array(heatmap)

                    # reshape 1D → 2D
                    if heatmap.ndim == 1:
                        size = int(np.ceil(np.sqrt(len(heatmap))))
                        padded = np.zeros(size * size)
                        padded[:len(heatmap)] = heatmap
                        heatmap = padded.reshape(size, size)

                    fig, ax = plt.subplots()
                    ax.imshow(heatmap, cmap='jet')
                    ax.set_title("AI Attention Map")
                    ax.axis('off')

                    st.pyplot(fig)

                else:
                    try:
                        img = Image.open(io.BytesIO(heatmap))
                        st.image(img, width="stretch")
                    except:
                        st.warning("⚠️ Heatmap not available")

            # PROBABILITIES
            with col3:
                st.subheader("📊 Probabilities")
                fig, ax = plt.subplots()
                ax.bar(labels, probs)
                plt.xticks(rotation=45)
                ax.set_ylabel("Confidence")
                st.pyplot(fig)

            st.divider()

            html_report, text_report = generate_html_report(prediction, confidence, probs, labels)
            st.markdown(html_report, unsafe_allow_html=True)

            st.download_button("⬇️ Download Report", text_report, file_name="report.txt")

        except Exception as e:
            st.error(f"Error: {e}")

else:
    st.info("👈 Upload an image to start analysis.")