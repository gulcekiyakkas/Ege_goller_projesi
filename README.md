🌍 Ege Gölleri Zaman Serisi Analizi
YOLOv8 Segmentasyon + NDWI / NDVI Tabanlı Uzaktan Algılama Projesi

Bu proje, Ege Bölgesi’nde yer alan Burdur, Eber, Işıklı ve Salda göllerinin yıllar içerisindeki su alanı ve bitki örtüsü değişimlerini analiz etmek amacıyla geliştirilmiştir. Çalışmada hem klasik uzaktan algılama indeksleri (NDVI, NDWI) hem de derin öğrenme tabanlı YOLOv8 segmentasyon modeli birlikte kullanılmıştır. Proje, elde edilen sonuçların etkileşimli biçimde incelenebilmesi için Streamlit tabanlı bir web arayüzü ile sunulmaktadır.

🎯 Projenin Amacı

Bu çalışmanın temel amacı, uydu görüntüleri üzerinden göllerin zamansal değişimini nicel olarak analiz etmek ve klasik indeks yöntemleri ile derin öğrenme temelli segmentasyon yaklaşımlarını karşılaştırmalı biçimde değerlendirmektir. Özellikle su alanı kayıplarının uzun vadede nasıl bir eğilim gösterdiği ortaya konulmakta ve geleceğe yönelik öngörüler üretilmektedir.

🛰️ Kullanılan Veri Seti

Projede her göl için aşağıdaki yıllara ait uydu görüntüleri kullanılmıştır:

1990

2000

2010

2020

Toplamda 4 göl × 4 yıl = 16 uydu görüntüsü analiz edilmiştir.
YOLOv8 segmentasyon modeli için ayrıca 25 görüntüden oluşan özel bir segmentasyon veri seti oluşturulmuş ve model bu veri seti üzerinde eğitilmiştir.

🧠 Kullanılan Yöntemler ve Teknolojiler
Uzaktan Algılama İndeksleri

NDVI (Normalized Difference Vegetation Index)
Bitki örtüsü yoğunluğunu belirlemek için kullanılmıştır.

NDWI (Normalized Difference Water Index)
Su alanlarının tespiti ve yüzdesel dağılımı için kullanılmıştır.

Derin Öğrenme

YOLOv8 Segmentasyon (YOLOv8s-seg)
Göl su alanlarının piksel bazlı olarak tespit edilmesi amacıyla eğitilmiştir.

Otomatik maske üretimi için NDWI tabanlı ön işlem uygulanmıştır.

Model 50 epoch boyunca eğitilmiş ve en iyi ağırlıklar best.pt dosyası olarak kaydedilmiştir.

Zaman Serisi ve Trend Analizi

Doğrusal regresyon kullanılarak:

NDWI su trendi

NDVI bitki trendi

YOLO tabanlı su alanı trendi
hesaplanmıştır.

2050 ve 2100 yılları için su alanı tahminleri üretilmiştir.

🖥️ Uygulama Arayüzü (Streamlit)

Proje, Streamlit kullanılarak geliştirilen etkileşimli bir arayüz üzerinden sunulmaktadır. Arayüzde aşağıdaki özellikler yer almaktadır:

Göl seçimi

Yıllara göre NDVI ve NDWI haritaları

YOLO segmentasyon sonuçlarına dayalı su yüzdesi hesapları

Zaman serisi grafikleri ve trend çizgileri

Geleceğe yönelik su alanı tahminleri

Otomatik PDF rapor oluşturma

📊 Çıktılar

NDVI / NDWI harita görselleştirmeleri

YOLO segmentasyon maskeleri

Su ve bitki değişim grafikleri

Yıllık trend değerleri (% / yıl)

2050 ve 2100 projeksiyonları

Akademik formatta PDF analiz raporu

📁 Proje Yapısı (Özet)
YOLO_Training/
│
├── dataset/
│   ├── images/
│   ├── labels/
│   └── masks/
│
├── runs/
│   └── segment/
│       └── seg_train/
│           └── weights/
│               └── best.pt
│
├── create_seg_dataset.py
├── train_segment.py
└── data.yaml


Streamlit uygulaması ana dizinde yer alan app.py dosyası üzerinden çalıştırılmaktadır.

⚙️ Kurulum ve Çalıştırma

Gerekli kütüphaneler:

pip install ultralytics streamlit opencv-python numpy pandas matplotlib scikit-learn fpdf pillow


Uygulamayı çalıştırmak için:

streamlit run app.py
