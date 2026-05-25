import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

from huggingface_hub import hf_hub_download
import os

st.set_page_config(
    page_title="SMART AGRICULTURE VISION SYSTEM",
    page_icon="Assets/page icon SAVIS.png",
    layout="wide",
    initial_sidebar_state=None,
)

st.title("SMART AGRICULTURE VISION SYSTEM")

tab1, tab2, tab3 = st.tabs(["Prediksi", "Karakteristik Visual Defisiensi", "Tentang Model"])
with tab1:
    st.header("Unggah foto daun tanaman")
    st.write("Ambil foto daun yang jelas, hindari bayangan dan blur")

    #Kotak unggah foto
    st.markdown("""
    <style>
    [data-testid="stFileUploader"] {
        border: 2px dashed #1B4332;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Seret & lepas foto di sini atau tekan untuk memilih", 
        type=["jpg", "jpeg", "png"],
        max_upload_size=20,
        accept_multiple_files=False
    )

    st.write("atau gunakan kamera")
    
    enable = st.checkbox("Izinkan kamera")
    camera_photo = st.camera_input(
        "",
        key=None, 
        help=None,
        disabled=not enable
    )

    image_source = uploaded_file or camera_photo
      
    # Pastikan file 'best.pt' ada di folder yang sama dengan file app.py ini
    @st.cache_resource 
    def load_model():
        model_path = hf_hub_download(
            repo_id="https://huggingface.co/NewbieFinalBosss/SAVIS/tree/main",
            filename="best.pt")
        model = YOLO(model_path)
        return model
    
    
    model = load_model()
           
    if image_source is not None:
        # Konversi file ke format gambar PIL
        image = Image.open(image_source)
                # Buat dua kolom untuk membandingkan Original vs Hasil
        col1, col2 = st.columns(2)
        #Buat slide ambang kepercayaan
        conf_threshold = st.slider("Ambang kepercayaan", 0.0, 1.0, 0.838)

        with col1:
            st.subheader("Gambar Asli")
            st.image(image, use_container_width=True)

        with col2:
            st.subheader("Hasil Deteksi")
            # Jalankan Prediksi
            with st.spinner('Sedang mendeteksi...'):
                # Convert PIL image ke format yang dimengerti YOLO (numpy array)
                img_array = np.array(image)
                results = model.predict(source=img_array, conf=conf_threshold)
                # Ambil gambar hasil plot (bounding boxes)
                # results[0].plot() mengembalikan array gambar dengan kotak deteksi
                res_plotted = results[0].plot()
                # Tampilkan hasil
                st.image(res_plotted, channels="BGR", use_container_width=True)

            #Tampilkan Informasi Tambahan
        st.divider()
        st.subheader("Detail Deteksi")
        if len(results[0].boxes) > 0:
            for box in results[0].boxes:
                class_id = int(box.cls[0])
                label = model.names[class_id]
                prob = float(box.conf[0])
                st.write(f"- Menemukan **{label}** dengan tingkat keyakinan **{prob:.2f}**")
        else:
            st.write("Tidak ada objek yang terdeteksi.")

with tab2:
    st.header("Karakteristik Visual Daun Hara Tercukupi", divider="blue")
    st.write("Daun yang tercukupi haranya menampilkan warna hijau merata tanpa perubahan warna abnormal")
    st.image("Assets/Hara Tercukupi.JPG", width="stretch")
    st.write("")
    st.write("")
    st.write("")

    st.header("Karakteristik Visual Daun Defisiensi Nitrogen", divider="blue")
    st.write("Defisiensi nitrogen menyebabkan tanaman tumbuh kerdil dengan daun yang sempit. Gejala klorosis dimulai dari daun tua karena nitrogen diremobilisasi ke daun muda untuk pertumbuhan. Secara visual, tanaman tampak hijau pucat atau kuning (Marschner, 2012).")
    st.image("Assets/Defisiensi N.JPG", width="stretch")
    st.write("")
    st.write("")
    st.write("")

    st.header("Karakteristik Visual Daun Defisiensi Fosfor", divider="blue")
    st.write("Defisiensi fosfor menghambat pertumbuhan tanaman sehingga daun yang tumbuh relatif sedikit. Gejala visual dimulai dari daun tua, berupa klorosis antartulang daun dan nekrosis yang berkembang menyatu di sepanjang tepi daun (Yara Canada, 2018).")
    st.image("Assets/Defisiensi P.JPG", width="stretch")
    st.write("")
    st.write("")
    st.write("")

    st.header("Karakteristik Visual Daun Defisiensi Kalium", divider="blue")
    st.write("Defisiensi kalium menyebabkan klorosis antarvena yang dimulai pada daun tua, sementara vena utama tetap berwarna hijau untuk sementara. Klorosis kemudian berlanjut ke arah pangkal daun dan diikuti munculnya nekrosis di bagian tepi daun (Yara Canada, 2018).")
    st.image("Assets/Defisiensi K.JPG", width="stretch")
    st.write("")
    st.write("")
    st.write("")

    st.header("Referensi")
    st.write("Marschner, P. (2012). Marschner's Mineral Nutrition of Higher Plants (Third Edition). Academic Press. DOI:10.1016/C2009-0-63043-9")
    st.write("Yara Canada. (2018). Nutrient Deficiencies in Soybean. Diakeses 17 Mei 2026, dari https://www.yaracanada.ca/crop-nutrition/soybean/nutrient-deficiencies/")
with tab3:
    st.header("Tentang Model")








