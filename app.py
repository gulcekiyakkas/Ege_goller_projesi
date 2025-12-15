import streamlit as st
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
from fpdf import FPDF
import tempfile
from sklearn.linear_model import LinearRegression

# ===============================
# MODEL ve DATA AYARLARI
# ===============================
YOLO_MODEL_PATH = r"C:\Users\gulce\OneDrive\Masaüstü\YOLO_Training\runs\segment\seg_train\weights\best.pt"
yolo_model = YOLO(YOLO_MODEL_PATH)

# ===============================
# YOLO MODEL METRİKLERİ (mAP, Precision, Recall)
# ===============================
try:
    model_info = yolo_model.info(verbose=False)
    yolo_metrics = {
        "mAP50": model_info.get("metrics/mAP50", "N/A"),
        "mAP50-95": model_info.get("metrics/mAP50-95", "N/A"),
        "precision": model_info.get("metrics/precision", "N/A"),
        "recall": model_info.get("metrics/recall", "N/A")
    }
except:
    yolo_metrics = {
        "mAP50": "N/A",
        "mAP50-95": "N/A",
        "precision": "N/A",
        "recall": "N/A"
    }


DATASET_DIR = Path("data")

lakes = ["burdur", "eber", "isikli", "salda"]
years = ["1990", "2000", "2010", "2020"]


# ===============================
# Yardımcı Fonksiyonlar
# ===============================

def compute_ndvi(image):
    img = np.array(image).astype(float)
    red = img[:, :, 2]
    green = img[:, :, 1]

    ndvi = (green - red) / (green + red + 1e-6)

    ndvi_norm = cv2.normalize(ndvi, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return ndvi_norm, ndvi


def compute_ndwi(image):
    img = np.array(image).astype(float)
    green = img[:, :, 1]
    blue = img[:, :, 0]

    ndwi = (green - blue) / (green + blue + 1e-6)

    ndwi_norm = cv2.normalize(ndwi, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return ndwi_norm, ndwi


def ndvi_ndwi_stats(ndvi, ndwi):
    # Sabit eşikler
    WATER_THRESHOLD = 0.10     # NDWI su eşiği
    GREEN_THRESHOLD = 0.20     # NDVI yeşil bitki eşiği

    water_mask = ndwi > WATER_THRESHOLD
    green_mask = ndvi > GREEN_THRESHOLD

    total = ndwi.size

    water_percent = np.sum(water_mask) / total * 100
    green_percent = np.sum(green_mask) / total * 100

    return water_percent, green_percent



def yolo_area_percent(image_path):
    img = cv2.imread(str(image_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = yolo_model(img)

    h, w, _ = img.shape
    total_area = h * w

    r = results[0]

    if r.masks is None:
        return {"water_%": 0}

    water_pixels = 0

    for mask in r.masks.data:
        mask_np = mask.cpu().numpy()
        water_pixels += np.sum(mask_np > 0.5)  # True sayısı

    water_percent = (water_pixels / total_area) * 100

    return {"water_%": water_percent}

def estimate_water_area_km2(image_path, water_percent):
    """
    1 pikselin metre çözünürlüğünü (Google Earth görüntü çözünürlüğü) kullanarak
    toplam su alanını km² olarak hesaplar.

    Varsayılan çözünürlük: 30 m/piksel
    Eğer senin dataset farklıysa burayı değiştirirsin.
    """
    PIXEL_RESOLUTION_M = 30     # metre/piksel
    PIXEL_AREA_M2 = PIXEL_RESOLUTION_M ** 2  # her pikselin kapladığı m²

    img = cv2.imread(str(image_path))
    h, w, _ = img.shape
    total_pixels = h * w
    water_pixels = total_pixels * (water_percent / 100)

    total_water_area_m2 = water_pixels * PIXEL_AREA_M2
    total_water_area_km2 = total_water_area_m2 / 1_000_000

    return total_water_area_km2


# ===============================
# ARAYÜZ
# ===============================

st.title("Ege Gölleri Zaman Serisi + YOLO + NDWI/NDVI Analizi")

selected_lake = st.selectbox("Göl Seç", lakes)

results = []
ndvi_maps = {}
ndwi_maps = {}

for year in years:
    img_path = DATASET_DIR / f"{selected_lake}_{year}.jpg"
    if not img_path.exists():
        continue

    img = Image.open(img_path)

    # NDVI & NDWI
    ndvi_map, ndvi_raw = compute_ndvi(img)
    ndwi_map, ndwi_raw = compute_ndwi(img)

    water_ndwi, green_ndvi = ndvi_ndwi_stats(ndvi_raw, ndwi_raw)

    # YOLO
    yolo_res = yolo_area_percent(img_path)

    results.append([
        int(year),
        float(np.mean(ndvi_raw)),
        float(np.mean(ndwi_raw)),
        water_ndwi,
        green_ndvi,
        yolo_res["water_%"]
    ])

    ndvi_maps[year] = ndvi_map
    ndwi_maps[year] = ndwi_map


df = pd.DataFrame(results, columns=[
    "Yıl", "NDVI Ort", "NDWI Ort", "Su (NDWI %)",
    "Yeşil (NDVI %)", "YOLO Su (%)"
])


st.subheader(f"{selected_lake.capitalize()} Gölü - Zaman Serisi Analizi")
st.dataframe(df)

# ===============================
# Göl Alanı Hesabı (km²)
# ===============================
first_year = df.iloc[0]
last_year = df.iloc[-1]

area_1990 = estimate_water_area_km2(DATASET_DIR / f"{selected_lake}_1990.jpg", first_year["YOLO Su (%)"])
area_2020 = estimate_water_area_km2(DATASET_DIR / f"{selected_lake}_2020.jpg", last_year["YOLO Su (%)"])

st.subheader("🌍 Su Alanı Değişimi (km²)")
st.write(f"**1990 Su Alanı:** {area_1990:.2f} km²")
st.write(f"**2020 Su Alanı:** {area_2020:.2f} km²")
st.write(f"**Toplam Kayıp:** {(area_1990 - area_2020):.2f} km²")

# ===============================
# GRAFİK
# ===============================

# ===============================
# GRAFİK + TREND ÇİZGİLERİ
# ===============================

fig, ax = plt.subplots(figsize=(8, 5))

years_arr = df["Yıl"].values.reshape(-1, 1)
model = LinearRegression()

# --- NDWI ---
ax.plot(df["Yıl"], df["Su (NDWI %)"], "o-", label="NDWI - Su (%)")
model.fit(years_arr, df["Su (NDWI %)"])
ndwi_trend = model.predict(years_arr)
ax.plot(df["Yıl"], ndwi_trend, "--", label="NDWI Trend")

# --- NDVI ---
ax.plot(df["Yıl"], df["Yeşil (NDVI %)"], "o-", label="NDVI - Yeşil (%)")
model.fit(years_arr, df["Yeşil (NDVI %)"])
ndvi_trend = model.predict(years_arr)
ax.plot(df["Yıl"], ndvi_trend, "--", label="NDVI Trend")

# --- YOLO Su ---
ax.plot(df["Yıl"], df["YOLO Su (%)"], "o-", label="YOLO Su (%)")
model.fit(years_arr, df["YOLO Su (%)"])
yolo_trend = model.predict(years_arr)
ax.plot(df["Yıl"], yolo_trend, "--", label="YOLO Trend")

ax.set_title(f"{selected_lake.capitalize()} Gölü – Su & Bitki Değişimi + Trend Çizgileri")
ax.set_xlabel("Yıl")
ax.legend()
ax.grid(True)

st.pyplot(fig)

# ===============================
# NDVI & NDWI HARİTALARI
# ===============================

st.subheader("NDVI / NDWI Haritaları")

for year in years:
    if year in ndvi_maps:
        st.markdown(f"### {year}")

        col1, col2 = st.columns(2)
        with col1:
            st.image(ndvi_maps[year], caption=f"{year} - NDVI", use_container_width=True)
        with col2:
            st.image(ndwi_maps[year], caption=f"{year} - NDWI", use_container_width=True)

# ===============================

# ===============================
# 2050 & 2100 Su Tahmini (Lineer Regresyon)
# ===============================
def forecast_future(df):
    years = df["Yıl"].values.reshape(-1, 1)
    y_water = df["YOLO Su (%)"].values

    model = LinearRegression()
    model.fit(years, y_water)

    pred_2050 = float(model.predict([[2050]]))
    pred_2100 = float(model.predict([[2100]]))

    return pred_2050, pred_2100

pred_2050, pred_2100 = forecast_future(df)

st.subheader("📈 Gelecek Su Tahmini")
st.write(f"**2050 Su Tahmini (YOLO):** %{pred_2050:.2f}")
st.write(f"**2100 Su Tahmini (YOLO):** %{pred_2100:.2f}")

# PDF
# Türkçe karakter düzeltici
def fix_text(txt):
    replace_map = {
        "ı": "i", "İ": "I",
        "ş": "s", "Ş": "S",
        "ğ": "g", "Ğ": "G",
        "ö": "o", "Ö": "O",
        "ü": "u", "Ü": "U",
        "ç": "c", "Ç": "C"
    }
    for k, v in replace_map.items():
        txt = txt.replace(k, v)
    return txt

def create_trend_plot(df, lake_name):
    years = df["Yıl"].values.reshape(-1, 1)

    # Trend çizgisi için regresyon modeli
    model = LinearRegression()

    fig, ax = plt.subplots(figsize=(8, 4))

    # NDWI Trend
    model.fit(years, df["Su (NDWI %)"])
    trend_ndwi = model.predict(years)
    ax.plot(df["Yıl"], df["Su (NDWI %)"], marker="o", label="NDWI Su (%)")
    ax.plot(df["Yıl"], trend_ndwi, "--", label="NDWI Trend")

    # NDVI Trend
    model.fit(years, df["Yeşil (NDVI %)"])
    trend_ndvi = model.predict(years)
    ax.plot(df["Yıl"], df["Yeşil (NDVI %)"], marker="o", label="NDVI Yeşil (%)")
    ax.plot(df["Yıl"], trend_ndvi, "--", label="NDVI Trend")

    # YOLO Su Trend
    model.fit(years, df["YOLO Su (%)"])
    trend_yolo = model.predict(years)
    ax.plot(df["Yıl"], df["YOLO Su (%)"], marker="o", label="YOLO Su (%)")
    ax.plot(df["Yıl"], trend_yolo, "--", label="YOLO Trend")

    ax.set_title(f"{lake_name.capitalize()} Gölü - Trend Analizi")
    ax.set_xlabel("Yıl")
    ax.legend()
    ax.grid(True)

    # Geçici dosyaya kaydet
    tmp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    fig.savefig(tmp_file.name, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return tmp_file.name

# ===============================
# PDF
# ===============================
if st.button("PDF Rapor Oluştur"):
    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Arial", "B", 16)
    pdf.cell(200, 10, fix_text(f"{selected_lake.capitalize()} Gölü Analiz Raporu"), ln=True, align="C")
    pdf.ln(5)

    # Tablo başlıkları
    headers = ["Yil", "NDVI Ort", "NDWI Ort", "Su (NDWI %)", "Yesil (NDVI %)", "YOLO Su (%)"]
    col_widths = [20, 30, 30, 35, 35, 30]

    pdf.set_font("Arial", "B", 12)
    for i, h in enumerate(headers):
        pdf.cell(col_widths[i], 10, fix_text(h), border=1, align="C")
    pdf.ln()

    # Satırlar
    pdf.set_font("Arial", "", 11)
    for row in results:
        for i, item in enumerate(row):
            if isinstance(item, float):
                txt = f"{item:.3f}"
            else:
                txt = str(item)
            pdf.cell(col_widths[i], 10, fix_text(txt), border=1, align="C")
        pdf.ln()

    # PDF üret
    pdf_bytes = pdf.output(dest="S").encode("latin1", "replace")

    
   
    st.download_button(
        "PDF İndir",
        data=pdf_bytes,
        file_name=fix_text(f"{selected_lake}_raporu.pdf"),
        mime="application/pdf"

        
    )
# ===============================
# Trend Değerlerini Hesaplama
# ===============================
def compute_trends(df):
    years = df["Yıl"].values.reshape(-1, 1)
    model = LinearRegression()

    # NDWI Trend
    model.fit(years, df["Su (NDWI %)"])
    ndwi_slope = model.coef_[0]

    # NDVI Trend
    model.fit(years, df["Yeşil (NDVI %)"])
    ndvi_slope = model.coef_[0]

    # YOLO Trend
    model.fit(years, df["YOLO Su (%)"])
    yolo_slope = model.coef_[0]

    return ndwi_slope, ndvi_slope, yolo_slope

ndwi_slope, ndvi_slope, yolo_slope = compute_trends(df)

st.subheader("📈 Trend Değerleri (Yıllık Değişim)")
st.write(f"**NDWI Su Trend:** {ndwi_slope:.4f} % / yıl")
st.write(f"**NDVI Yeşil Trend:** {ndvi_slope:.4f} % / yıl")
st.write(f"**YOLO Su Trend:** {yolo_slope:.4f} % / yıl")
