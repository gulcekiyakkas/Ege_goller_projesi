# Ege Gölleri Zaman Serisi Analizi  
### YOLOv8 Segmentasyon ve NDVI/NDWI Tabanlı Uzaktan Algılama Projesi

Bu projede, Ege Bölgesi’nde yer alan **Burdur, Eber, Işıklı ve Salda Gölleri**nin  
**1990–2020** yılları arasındaki zamansal değişimi;

- **YOLOv8 Segmentasyon modeli**
- **NDVI / NDWI uzaktan algılama indeksleri**
- **Zaman serisi ve trend analizi**

kullanılarak incelenmiştir.

Proje, göl yüzey alanlarındaki değişimin **hem klasik indeks yöntemleri** hem de  
**derin öğrenme tabanlı segmentasyon** ile karşılaştırmalı olarak analiz edilmesini amaçlamaktadır.

---

## 🔧 Kullanılan Teknolojiler

- **Python**
- **YOLOv8-Segmentation (Ultralytics)**
- **OpenCV**
- **NumPy / Pandas**
- **Matplotlib**
- **Streamlit**
- **FPDF**
- **Scikit-learn (Linear Regression)**

---

## 📊 Veri Seti

### Uydu Görüntüleri
- 4 göl × 4 yıl (1990, 2000, 2010, 2020)
- Toplam **16 adet** zaman serisi görüntüsü
- Görüntüler RGB formatında kullanılmıştır

### YOLO Eğitim Verisi
- Toplam **136 adet** göl görüntüsü
- NDWI tabanlı otomatik maske üretimi ile segmentasyon etiketleri oluşturulmuştur
- Manuel etiketleme yapılmadan **yarı-otomatik dataset** hazırlanmıştır

---

## 🧠 Model Eğitimi (YOLOv8 Segmentasyon)

- **Model:** YOLOv8s-seg
- **Epoch:** 80
- **Image Size:** 512×512
- **Eğitim Türü:** Su alanı segmentasyonu
- **Donanım:** NVIDIA RTX 2050 (CUDA)

### Eğitim Performansı (Özet)
- **Mask mAP50:** ≈ 0.99  
- **Mask mAP50-95:** ≈ 0.85  
- Model, su alanlarını yüksek doğrulukla segment edebilmektedir.

---

## 🌿 NDVI & NDWI Analizi

Projede klasik uzaktan algılama yaklaşımları da kullanılmıştır:

- **NDVI (Normalized Difference Vegetation Index)**  
  → Bitki örtüsü yoğunluğunu analiz etmek için

- **NDWI (Normalized Difference Water Index)**  
  → Su yüzeylerini belirlemek için

Sabit eşik değerleri kullanılarak:
- Su alanı yüzdesi
- Yeşil alan yüzdesi  

yıllara göre hesaplanmıştır.

---

## 📈 Zaman Serisi ve Trend Analizi

- Her göl için:
  - NDWI su yüzdesi
  - NDVI yeşil alan yüzdesi
  - YOLO segmentasyon su yüzdesi

yıllara göre karşılaştırılmıştır.

- **Linear Regression** kullanılarak:
  - Su alanı trendi
  - Bitki örtüsü trendi

grafiksel olarak gösterilmiştir.

---

## 🖥️ Streamlit Arayüzü

Proje, kullanıcı dostu bir **Streamlit arayüzü** ile sunulmaktadır.

Arayüzde:
- Göl seçimi
- Yıllara göre tablo
- Zaman serisi grafikleri
- NDVI / NDWI haritaları
- YOLO segmentasyon sonuçları
- Otomatik **PDF rapor üretimi**

özellikleri bulunmaktadır.

---

## 📄 PDF Raporlama

Streamlit üzerinden tek tıkla:
- Tablo sonuçları
- Sayısal analizler

içeren **PDF rapor** üretilebilmektedir.  
Türkçe karakter uyumluluğu için özel düzeltme uygulanmıştır.

---

## 🎯 Projenin Katkıları

- Klasik NDWI yöntemi ile derin öğrenme tabanlı segmentasyonun karşılaştırılması
- Göl su seviyelerinin zamansal değişiminin görsel ve sayısal analizi
- Otomatik dataset üretimi ile etiketleme yükünün azaltılması
- Akademik çalışmalara ve çevresel izleme projelerine altyapı oluşturması

---

## 📌 Not

Bu proje:
- **Akademik amaçlı**
- **Çevresel izleme ve uzaktan algılama odaklı**
- **Geliştirilmeye açık** bir çalışmadır.

Yeni yıllar, farklı göller veya çok bantlı uydu verileri eklenerek genişletilebilir.

---

## 👤 Geliştirici

**Gülce KIYAKKAŞ**  
 Uzaktan Algılama / Yapay Zeka Odaklı Proje Çalışması
