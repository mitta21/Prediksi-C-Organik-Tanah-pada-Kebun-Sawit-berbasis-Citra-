import streamlit as st
import cv2
import numpy as np
import pickle
import pandas as pd
import io
from datetime import datetime
from scipy.stats import skew
from skimage.feature import graycomatrix, graycoprops
from PIL import Image
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

st.set_page_config(
    page_title="Klasifikasi C-Organik Tanah pada Kebun Sawit Berbasis Citra ",
    page_icon="🌴",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    .block-container {
        padding: 1.5rem 2rem;
        max-width: 1200px;
        margin: auto;
    }
    @media (max-width: 768px) {
        .block-container { padding: 1rem; }
    }
    .header-box {
        background: linear-gradient(135deg, #1b4332 0%, #2d6a4f 50%, #40916c 100%);
        padding: 2rem 1.5rem;
        border-radius: 16px;
        text-align: center;
        margin-bottom: 1.5rem;
        box-shadow: 0 6px 20px rgba(27,67,50,0.3);
    }
    .header-box h1 {
        color: white;
        font-size: clamp(1.4rem, 3vw, 2.2rem);
        margin: 0 0 0.4rem 0;
        font-weight: 800;
        letter-spacing: 1px;
    }
    .header-box p { color: #b7e4c7; margin: 0; font-size: clamp(0.8rem, 1.5vw, 1rem); }
    .header-badge {
        display: inline-block;
        background: rgba(255,255,255,0.15);
        color: #d8f3dc;
        border-radius: 20px;
        padding: 0.3rem 1rem;
        font-size: 0.8rem;
        margin-top: 0.8rem;
        border: 1px solid rgba(255,255,255,0.2);
    }
    .card {
        background: white;
        border-radius: 14px;
        padding: 1.4rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.07);
        margin-bottom: 1rem;
        border: 1px solid #e9f5ee;
    }
    .card h3 { color: #1b4332; margin-top: 0; font-size: 1.1rem; }
    .step-box {
        background: linear-gradient(90deg, #d8f3dc, #f0faf2);
        border-radius: 10px;
        padding: 0.7rem 1rem;
        margin: 0.4rem 0;
        border-left: 4px solid #2d6a4f;
        font-size: 0.92rem;
        color: #1b4332;
    }
    .result-card {
        background: white;
        border-radius: 14px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 2px 12px rgba(0,0,0,0.07);
        border: 1px solid #e9f5ee;
        height: 100%;
    }
    .lapisan-title { font-weight: 700; font-size: 1rem; color: #1b4332; margin-bottom: 0.2rem; }
    .lapisan-depth { color: #74c69d; font-size: 0.85rem; margin-bottom: 0.8rem; }
    .label-badge {
        display: inline-block;
        padding: 0.5rem 1.2rem;
        border-radius: 25px;
        font-weight: 700;
        font-size: 0.95rem;
        margin-top: 0.5rem;
        width: 90%;
    }
    .sangat-rendah { background:#ffebee; color:#c62828; border:2px solid #ef9a9a; }
    .rendah        { background:#fff3e0; color:#bf360c; border:2px solid #ffcc80; }
    .sedang        { background:#fffde7; color:#f57f17; border:2px solid #fff176; }
    .tinggi        { background:#e8f5e9; color:#1b5e20; border:2px solid #a5d6a7; }
    .sangat-tinggi { background:#e3f2fd; color:#0d47a1; border:2px solid #90caf9; }
    .riwayat-row {
        background: white;
        border-radius: 10px;
        padding: 0.8rem 1rem;
        margin: 0.4rem 0;
        border: 1px solid #e9f5ee;
        box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    }
    .kelas-box {
        background: linear-gradient(135deg, #d8f3dc, #f0faf2);
        border-radius: 10px;
        padding: 0.7rem 1rem;
        margin: 0.35rem 0;
        display: flex;
        align-items: center;
        gap: 0.8rem;
        border: 1px solid #b7e4c7;
    }
    .kelas-box span { font-size: 1.2rem; }
    .kelas-name { font-weight: 700; color: #1b4332; font-size: 0.9rem; }
    .kelas-range { color: #52b788; font-size: 0.8rem; }
    .upload-hint {
        background: #f0faf2;
        border-radius: 10px;
        padding: 0.8rem;
        text-align: center;
        color: #52b788;
        font-size: 0.85rem;
        border: 2px dashed #b7e4c7;
        margin-top: 0.5rem;
    }
    .footer {
        background: linear-gradient(135deg, #1b4332, #2d6a4f);
        border-radius: 14px;
        padding: 1.5rem;
        text-align: center;
        margin-top: 2rem;
        color: #b7e4c7;
        font-size: 0.85rem;
        line-height: 1.8;
    }
    .footer b { color: #d8f3dc; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD MODEL
# ============================================================
@st.cache_resource
def load_model():
    with open('model_svm.pkl', 'rb') as f:
        return pickle.load(f)

model_data        = load_model()
hasil_per_lapisan = model_data['hasil_per_lapisan']
le                = model_data['label_encoder']
IMG_SIZE          = model_data['img_size']

# ============================================================
# FUNGSI EKSTRAKSI FITUR
# ============================================================
def extract_features(img_bgr):
    img_bgr = cv2.resize(img_bgr, IMG_SIZE, interpolation=cv2.INTER_AREA)
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    hsv_features = []
    for i in range(3):
        ch = img_hsv[:, :, i].flatten().astype(np.float32)
        hsv_features.append(np.mean(ch))
        hsv_features.append(np.std(ch))
        hsv_features.append(float(skew(ch)))
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_gray = (img_gray // 4).astype(np.uint8)
    glcm = graycomatrix(img_gray, distances=[1],
                        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                        levels=64, symmetric=True, normed=True)
    glcm_features = []
    for prop in ['contrast','dissimilarity','homogeneity','energy','correlation','ASM']:
        glcm_features.append(float(np.mean(graycoprops(glcm, prop))))
    return hsv_features + glcm_features

# ============================================================
# FUNGSI BUAT PDF
# ============================================================
def buat_pdf(nama_file, waktu, hasil, info_kelas, nama_lap, kedalaman):
    buffer = io.BytesIO()
    doc    = SimpleDocTemplate(buffer, pagesize=A4,
                               rightMargin=2*cm, leftMargin=2*cm,
                               topMargin=2*cm, bottomMargin=2*cm)
    styles  = getSampleStyleSheet()
    elemen  = []

    # Judul
    judul_style = ParagraphStyle('judul', parent=styles['Title'],
                                 fontSize=16, textColor=colors.HexColor('#1b4332'),
                                 spaceAfter=6)
    sub_style   = ParagraphStyle('sub', parent=styles['Normal'],
                                 fontSize=10, textColor=colors.HexColor('#52b788'),
                                 spaceAfter=20)
    elemen.append(Paragraph("🌴 Klasifikasi C-Organik Tanah pada Kebun Sawit Berbasis Citra", judul_style))
    elemen.append(Paragraph("Laporan Prediksi Kadar C-Organik Tanah Sawit", sub_style))

    # Info analisis
    info_style = ParagraphStyle('info', parent=styles['Normal'],
                                fontSize=10, spaceAfter=4)
    elemen.append(Paragraph(f"<b>Nama File   :</b> {nama_file}", info_style))
    elemen.append(Paragraph(f"<b>Waktu Analisis :</b> {waktu}", info_style))
    elemen.append(Spacer(1, 0.5*cm))

    # Tabel hasil
    elemen.append(Paragraph("<b>Hasil Prediksi per Lapisan</b>",
                            ParagraphStyle('h', parent=styles['Normal'],
                                           fontSize=12, spaceAfter=8,
                                           textColor=colors.HexColor('#1b4332'))))
    data_tabel = [['Lapisan', 'Kedalaman', 'Kelas C-Organik', 'Kadar']]
    for lap_num in [1, 2, 3]:
        label = hasil[lap_num]['label']
        info  = info_kelas[label]
        data_tabel.append([
            nama_lap[lap_num],
            kedalaman[lap_num],
            f"{info['emoji']} {info['desc']}",
            info['kadar']
        ])

    tabel = Table(data_tabel, colWidths=[3.5*cm, 4*cm, 6*cm, 3*cm])
    tabel.setStyle(TableStyle([
        ('BACKGROUND',  (0,0), (-1,0),  colors.HexColor('#2d6a4f')),
        ('TEXTCOLOR',   (0,0), (-1,0),  colors.white),
        ('FONTNAME',    (0,0), (-1,0),  'Helvetica-Bold'),
        ('FONTSIZE',    (0,0), (-1,0),  11),
        ('ALIGN',       (0,0), (-1,-1), 'CENTER'),
        ('VALIGN',      (0,0), (-1,-1), 'MIDDLE'),
        ('ROWBACKGROUNDS', (0,1), (-1,-1),
         [colors.HexColor('#f0faf2'), colors.white]),
        ('GRID',        (0,0), (-1,-1), 0.5, colors.HexColor('#b7e4c7')),
        ('ROUNDEDCORNERS', [5]),
        ('TOPPADDING',  (0,0), (-1,-1), 8),
        ('BOTTOMPADDING',(0,0), (-1,-1), 8),
    ]))
    elemen.append(tabel)
    elemen.append(Spacer(1, 0.8*cm))

    # Keterangan kelas
    elemen.append(Paragraph("<b>Keterangan Kelas C-Organik</b>",
                            ParagraphStyle('h2', parent=styles['Normal'],
                                           fontSize=12, spaceAfter=8,
                                           textColor=colors.HexColor('#1b4332'))))
    ket_data = [['Kelas', 'Kadar C-Organik']]
    for kelas, info in info_kelas.items():
        ket_data.append([f"{info['emoji']} {info['desc']}", info['kadar']])

    ket_tabel = Table(ket_data, colWidths=[8*cm, 8*cm])
    ket_tabel.setStyle(TableStyle([
        ('BACKGROUND',  (0,0), (-1,0),  colors.HexColor('#2d6a4f')),
        ('TEXTCOLOR',   (0,0), (-1,0),  colors.white),
        ('FONTNAME',    (0,0), (-1,0),  'Helvetica-Bold'),
        ('ALIGN',       (0,0), (-1,-1), 'CENTER'),
        ('VALIGN',      (0,0), (-1,-1), 'MIDDLE'),
        ('ROWBACKGROUNDS', (0,1), (-1,-1),
         [colors.HexColor('#f0faf2'), colors.white]),
        ('GRID',        (0,0), (-1,-1), 0.5, colors.HexColor('#b7e4c7')),
        ('TOPPADDING',  (0,0), (-1,-1), 6),
        ('BOTTOMPADDING',(0,0), (-1,-1), 6),
    ]))
    elemen.append(ket_tabel)
    elemen.append(Spacer(1, 1*cm))

    # Footer PDF
    footer_style = ParagraphStyle('footer', parent=styles['Normal'],
                                  fontSize=8, textColor=colors.HexColor('#74c69d'),
                                  alignment=1)
    elemen.append(Paragraph(
        "Institut Teknologi Sawit Indonesia | Fakultas Sains dan Teknologi | "
        "Jurusan Sistem dan Teknologi Informasi", footer_style))

    doc.build(elemen)
    buffer.seek(0)
    return buffer

# ============================================================
# SESSION STATE untuk riwayat
# ============================================================
if 'riwayat' not in st.session_state:
    st.session_state.riwayat = []

# ============================================================
# DATA KELAS
# ============================================================
info_kelas = {
    'sangat rendah': {'emoji':'🔴','css':'sangat-rendah','kadar':'< 1%',  'desc':'Sangat Rendah'},
    'rendah'       : {'emoji':'🟠','css':'rendah',       'kadar':'1 - 2%','desc':'Rendah'},
    'sedang'       : {'emoji':'🟡','css':'sedang',       'kadar':'2 - 3%','desc':'Sedang'},
    'tinggi'       : {'emoji':'🟢','css':'tinggi',       'kadar':'3 - 5%','desc':'Tinggi'},
    'sangat tinggi': {'emoji':'🔵','css':'sangat-tinggi','kadar':'> 5%',  'desc':'Sangat Tinggi'},
}
nama_lap  = {1:'Lapisan 1', 2:'Lapisan 2', 3:'Lapisan 3'}
kedalaman = {1:'0 – 20 cm', 2:'20 – 40 cm', 3:'40 – 60 cm'}

# ============================================================
# HEADER
# ============================================================
st.markdown("""
<div class="header-box">
    <h1>Klasifikasi Kadar C-Organik Tanah pada Kebun Sawit Berbasis Citra</h1>
    <div class="header-badge"> Powered by Support Vector Machine (SVM)</div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# LAYOUT UTAMA
# ============================================================
col_kiri, col_kanan = st.columns([1, 1.6], gap="large")

# ---- KOLOM KIRI ----
with col_kiri:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("###  Cara Penggunaan")
    for step in [
        "1️⃣ Siapkan foto tanah sawit tampak profil",
        "2️⃣ Pastikan 3 lapisan tanah terlihat jelas",
        "3️⃣ Upload foto di bawah ini",
        "4️⃣ Klik tombol <b>Analisis</b> dan lihat hasilnya"
    ]:
        st.markdown(f'<div class="step-box">{step}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("###  Upload Foto Tanah")
    uploaded_file = st.file_uploader("Upload foto",
                                      type=['jpg','jpeg','png'],
                                      label_visibility='collapsed')
    st.markdown("""
    <div class="upload-hint">
         Format: JPG, JPEG, PNG<br>
        Pastikan foto menampilkan 3 lapisan tanah secara vertikal
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption=' Foto yang diupload', use_column_width=True)
        tombol = st.button(" Analisis Kadar C-Organik",
                           type='primary', use_container_width=True)
    else:
        tombol = False

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("###  Keterangan Kelas C-Organik")
    for kelas, info in info_kelas.items():
        st.markdown(f"""
        <div class="kelas-box">
            <span>{info['emoji']}</span>
            <div>
                <div class="kelas-name">{info['desc']}</div>
                <div class="kelas-range">Kadar: {info['kadar']}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ---- KOLOM KANAN ----
with col_kanan:
    if uploaded_file and tombol:
        img_bgr   = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        h         = img_bgr.shape[0]
        sepertiga = h // 3
        potongan  = {
            1: img_bgr[0:sepertiga, :],
            2: img_bgr[sepertiga:2*sepertiga, :],
            3: img_bgr[2*sepertiga:, :]
        }
        waktu = datetime.now().strftime('%d-%m-%Y %H:%M:%S')

        with st.spinner('🔄 Sedang menganalisis citra tanah...'):
            hasil = {}
            for lap_num, pot in potongan.items():
                fitur  = extract_features(pot)
                scaled = hasil_per_lapisan[lap_num]['scaler'].transform([fitur])
                pred   = hasil_per_lapisan[lap_num]['model'].predict(scaled)
                label  = le.inverse_transform(pred)[0]
                hasil[lap_num] = {'label': label, 'pot': pot}

        # Simpan ke riwayat
        st.session_state.riwayat.append({
            'Waktu'     : waktu,
            'Nama File' : uploaded_file.name,
            'Lapisan 1' : hasil[1]['label'],
            'Lapisan 2' : hasil[2]['label'],
            'Lapisan 3' : hasil[3]['label'],
        })

        st.success(" Analisis selesai!")
        st.markdown("###  Hasil Prediksi per Lapisan")

        c1, c2, c3 = st.columns(3, gap="small")
        for lap_num, col in zip([1,2,3], [c1,c2,c3]):
            label   = hasil[lap_num]['label']
            pot_rgb = cv2.cvtColor(
                cv2.resize(hasil[lap_num]['pot'], (120,120)),
                cv2.COLOR_BGR2RGB)
            info = info_kelas[label]
            with col:
                st.markdown('<div class="result-card">', unsafe_allow_html=True)
                st.image(pot_rgb, use_column_width=True)
                st.markdown(f'<div class="lapisan-title">{nama_lap[lap_num]}</div>',
                            unsafe_allow_html=True)
                st.markdown(f'<div class="lapisan-depth"> {kedalaman[lap_num]}</div>',
                            unsafe_allow_html=True)
                st.markdown(
                    f'<span class="label-badge {info["css"]}">'
                    f'{info["emoji"]} {info["desc"].upper()}</span>',
                    unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

        # Ringkasan
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("###  Ringkasan Hasil")
        st.markdown('<div class="card">', unsafe_allow_html=True)
        header_cols = st.columns([1.5, 1.5, 1.5, 1])
        header_cols[0].markdown("**Lapisan**")
        header_cols[1].markdown("**Kedalaman**")
        header_cols[2].markdown("**Kelas**")
        header_cols[3].markdown("**Kadar**")
        st.markdown("<hr style='margin:0.3rem 0;border-color:#d8f3dc'>",
                    unsafe_allow_html=True)
        for lap_num in [1,2,3]:
            label = hasil[lap_num]['label']
            info  = info_kelas[label]
            row   = st.columns([1.5, 1.5, 1.5, 1])
            row[0].markdown(f"**{nama_lap[lap_num]}**")
            row[1].markdown(kedalaman[lap_num])
            row[2].markdown(
                f'<span class="label-badge {info["css"]}" '
                f'style="padding:0.2rem 0.6rem;font-size:0.8rem">'
                f'{info["emoji"]} {info["desc"]}</span>',
                unsafe_allow_html=True)
            row[3].markdown(f"**{info['kadar']}**")
        st.markdown('</div>', unsafe_allow_html=True)

        # Tombol Download
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("###  Simpan Hasil")
        dl1, dl2 = st.columns(2, gap="small")

        # Download CSV
        with dl1:
            df_hasil = pd.DataFrame([{
                'Waktu'           : waktu,
                'Nama File'       : uploaded_file.name,
                'Lapisan 1 (0-20cm)'  : hasil[1]['label'],
                'Lapisan 2 (20-40cm)' : hasil[2]['label'],
                'Lapisan 3 (40-60cm)' : hasil[3]['label'],
            }])
            csv = df_hasil.to_csv(index=False).encode('utf-8')
            st.download_button(
                label=" Download CSV",
                data=csv,
                file_name=f"corganik_{uploaded_file.name}_{waktu[:10]}.csv",
                mime='text/csv',
                use_container_width=True
            )

        # Download PDF
        with dl2:
            pdf_buffer = buat_pdf(
                uploaded_file.name, waktu,
                hasil, info_kelas, nama_lap, kedalaman)
            st.download_button(
                label=" Download PDF",
                data=pdf_buffer,
                file_name=f"corganik_{uploaded_file.name}_{waktu[:10]}.pdf",
                mime='application/pdf',
                use_container_width=True
            )

    elif not uploaded_file:
        st.markdown('<div class="card" style="text-align:center;padding:3rem 1rem">',
                    unsafe_allow_html=True)
        st.markdown("""
        <div style="font-size:4rem"></div>
        <h3 style="color:#2d6a4f">Selamat Datang!</h3>
        <p style="color:#74c69d">
            Upload foto profil tanah sawit di sebelah kiri<br>
            untuk memulai analisis kadar C-Organik
        </p>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("###  Tentang Model")
        st.markdown("""
        <p style="color:#52b788; line-height:1.8">
        Model ini menganalisis kadar C-Organik tanah pada kebun sawit
        berbasis citra digital menggunakan algoritma
        <b style="color:#1b4332">Support Vector Machine (SVM)</b>.<br><br>
        Model ini menganalisis <b style="color:#1b4332">3 lapisan tanah</b>
        (0–20 cm, 20–40 cm, 40–60 cm) secara otomatis dari satu foto
        profil tanah, menggunakan ekstraksi fitur
        <b style="color:#1b4332">HSV Color Moment</b> dan
        <b style="color:#1b4332">GLCM Tekstur</b>.
        </p>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


# RIWAYAT ANALISIS

if st.session_state.riwayat:
    st.markdown("---")
    st.markdown("###  Riwayat Analisis")

    col_riwayat, col_dl_riwayat = st.columns([3, 1], gap="large")

    with col_riwayat:
        for i, r in enumerate(reversed(st.session_state.riwayat)):
            st.markdown(f"""
            <div class="riwayat-row">
                <b> {r['Nama File']}</b> &nbsp;|&nbsp;
                 {r['Waktu']}<br>
                <small>
                    Lap.1: {info_kelas[r['Lapisan 1']]['emoji']} {r['Lapisan 1'].title()} &nbsp;|&nbsp;
                    Lap.2: {info_kelas[r['Lapisan 2']]['emoji']} {r['Lapisan 2'].title()} &nbsp;|&nbsp;
                    Lap.3: {info_kelas[r['Lapisan 3']]['emoji']} {r['Lapisan 3'].title()}
                </small>
            </div>
            """, unsafe_allow_html=True)

    with col_dl_riwayat:
        st.markdown("<br><br>", unsafe_allow_html=True)
        df_riwayat = pd.DataFrame(st.session_state.riwayat)
        csv_riwayat = df_riwayat.to_csv(index=False).encode('utf-8')
        st.download_button(
            label=" Download Semua Riwayat (CSV)",
            data=csv_riwayat,
            file_name=f"riwayat_corganik_{datetime.now().strftime('%d%m%Y')}.csv",
            mime='text/csv',
            use_container_width=True
        )
        if st.button(" Hapus Riwayat", use_container_width=True):
            st.session_state.riwayat = []
            st.rerun()

# FOOTER

st.markdown("""
<div class="footer">
    <b>Klasifikasi Kadar C-Organik Tanah pada Kebun Sawit Berbasis Citra</b><br>
    <b>Institut Teknologi Sawit Indonesia</b><br>
    Fakultas Sains dan Teknologi &nbsp;|&nbsp; Jurusan Sistem dan Teknologi Informasi
</div>
""", unsafe_allow_html=True)
