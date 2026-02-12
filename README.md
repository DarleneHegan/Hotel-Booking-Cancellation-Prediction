# 🏨 Hotel Booking Cancellation Prediction

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange)
![Status](https://img.shields.io/badge/Status-Completed-success)

> **Strategic Revenue Protection:** A machine learning project to predict hotel booking cancellations with **86% Accuracy** and **81% Recall**, enabling proactive revenue management.

---

## 📋 Table of Contents
1. [Latar Belakang & Masalah](#-latar-belakang--masalah)
2. [Target Project](#-target-project)
3. [End-to-End Workflow](#-end-to-end-workflow)
4. [Model Configuration](#-model-configuration)
5. [Cross-Validation Results](#-cross-validation-results)
6. [Model Evaluation: Original vs Optimized](#-model-evaluation-original-vs-optimized)
7. [Key Insights (SHAP Analysis)](#-key-insights-shap-analysis)
8. [Business Recommendations](#-business-recommendations)

---

## 📌 Latar Belakang & Masalah

Industri perhotelan sering menghadapi tantangan besar berupa pembatalan pesanan mendadak yang berdampak langsung pada pendapatan.

1.  **Tingginya Ketidakpastian Operasional:** Pembatalan pesanan (*booking cancellations*) secara mendadak menyebabkan kamar hotel seringkali kosong di saat-saat terakhir. Hal ini mengakibatkan kerugian pendapatan (*Revenue Leakage*) yang tidak dapat dipulihkan.
2.  **Inefisiensi Manajemen Stok Kamar:** Tanpa prediksi yang akurat, manajemen sulit menentukan kebijakan *overbooking* yang aman. Jika terlalu berani, hotel berisiko menolak tamu (*overbooked*); jika terlalu pasif, hotel kehilangan potensi keuntungan.
3.  **Kegagalan Strategi Manual:** Penilaian manual berdasarkan intuisi staf seringkali tidak mampu melihat pola kompleks dari ribuan data transaksi, seperti hubungan antara waktu tunggu (*lead time*), asal negara, dan tipe deposit.

---

## 🎯 Target Project

### 1. Target Teknis (Data Science)
* **Pengembangan Model Klasifikasi Robust:** Membangun model *machine learning* yang mampu mempelajari pola perilaku pembatalan tamu secara akurat dari data historis.
* **Optimasi Deteksi Kelas Minoritas (Recall):** Menitikberatkan pada kemampuan model untuk mengidentifikasi sebanyak mungkin potensi pembatalan, guna meminimalisir adanya tamu yang batal tanpa terdeteksi oleh sistem.
* **Keseimbangan Performa (Trade-off):** Mencari titik optimal antara ketepatan tebakan (*Precision*) dan sensitivitas deteksi (*Recall*) agar model dapat digunakan dalam operasional harian tanpa mengganggu tamu yang berniat menginap.

### 2. Target Bisnis (Strategic Impact)
* **Mitigasi Revenue Leakage:** Membantu pihak manajemen hotel dalam memprediksi potensi kerugian pendapatan akibat kamar kosong.
* **Efisiensi Manajemen Stok Kamar:** Menyediakan dasar pengambilan keputusan bagi tim reservasi dalam menentukan kebijakan *overbooking* yang aman dan terukur.
* **Intervensi Berbasis Data:** Mengidentifikasi faktor risiko utama agar hotel dapat menjalankan langkah preventif (misal: penyesuaian kebijakan deposit).

---

## ⚙️ End-to-End Workflow

Proyek ini dikerjakan melalui 11 tahapan teknis yang ketat:

### 1. Data Gathering
* Pemuatan dataset `hotel_bookings.csv` dan pemeriksaan distribusi target `is_canceled`.
* Identifikasi *missing values* signifikan pada kolom `children`, `country`, `agent`, dan `company`.
* Deteksi anomali data, *outlier*, dan duplikasi.

### 2. Data Preprocessing
* **Cleaning:** Menghapus baris *null*, membuang kolom `company`, hapus duplikat, dan konversi `reservation_status_date`.
* **Filtering Anomali:** Menghapus transaksi tidak wajar (0 tamu, tarif di luar range, dll).
* **Splitting:** Membagi data 70% Train : 30% Test dengan `stratify=y`.
* **Post-Split Cleaning:** Imputasi median untuk `agent`. Transformasi `log1p` pada `lead_time` dan `sqrt` pada `adr`.

### 3. Feature Engineering & Selection
* **New Features:** Membuat fitur `country_risk` & `agent_risk` (Target Encoding), `total_guests`, `is_family`, `room_type_changed`, dll.
* **Selection:** Menggunakan uji statistik (*T-Test* & *Chi-Square*) dan *Feature Importance* untuk membuang fitur *noise*.

### 4. Modeling & Tuning
* Melatih model **XGBoost Classifier**.
* Melakukan **Threshold Tuning** menggunakan *Precision-Recall Curve* untuk mencari titik potong optimal.

---

## 🛠️ Model Configuration

Model dibangun menggunakan algoritma **XGBoost** dengan parameter manual berikut untuk menangani karakteristik data:

| Parameter | Nilai | Deskripsi |
| :--- | :--- | :--- |
| **n_estimators** | 500 | Jumlah pohon keputusan yang dibangun dalam model. |
| **max_depth** | 10 | Kedalaman maksimal setiap pohon untuk menangkap pola data yang kompleks. |
| **learning_rate** | 0.03 | Kecepatan model dalam mempelajari pola (*shrinkage*) untuk mencegah overfitting. |
| **scale_pos_weight**| 2.0 | **Penting:** Memberikan bobot lebih pada kelas minoritas (pembatalan). |
| **min_child_weight**| 2 | Kontrol beban sampel minimal yang dibutuhkan pada setiap cabang pohon. |
| **gamma** | 0.1 | Parameter regularisasi untuk pemangkasan pohon (*pruning*). |
| **subsample** | 0.85 | Persentase sampel data yang digunakan untuk melatih setiap pohon. |
| **colsample_bytree**| 0.85 | Persentase fitur yang digunakan dalam pembangunan setiap pohon. |
| **eval_metric** | 'logloss' | Metrik evaluasi yang digunakan selama proses pelatihan model. |
| **random_state** | 42 | Menjamin hasil yang konsisten dan dapat direproduksi. |

---

## 📈 Cross-Validation Results

Evaluasi dilakukan menggunakan **5-Fold Stratified Cross-Validation** untuk menguji stabilitas model pada subset data yang berbeda.

* **Hasil tiap Fold:** `[0.8497, 0.8482, 0.8536, 0.8498, 0.8466]`
* **Rata-rata F1-Score CV:** `0.8496`
* **Standar Deviasi:** `0.0023`

> **Interpretasi Stabilitas:**
> Standar deviasi yang sangat rendah (**0.0023**) menunjukkan bahwa model sangat **stabil**. Ini berarti performa model tidak *overfit* pada satu subset data tertentu dan dapat diharapkan memberikan hasil yang konsisten saat digunakan di dunia nyata.

---

## 📊 Model Evaluation: Original vs Optimized

Berikut adalah perbandingan performa antara model dengan threshold standar (0.5) dan model yang telah dioptimalkan (*threshold tuning*).

### 1. Model XGBoost Original (Baseline - Threshold 0.5)
* **Akurasi (Accuracy): 85%**
    * Model mampu memprediksi secara benar apakah pesanan akan dibatalkan atau tidak sebanyak 85% dari seluruh kasus.
* **Presisi (Precision - Kelas 1): 70%**
    * Dari semua pesanan yang diprediksi akan batal, 70% di antaranya benar-benar batal. Masih ada 30% risiko *false alarm*.
* **Recall (Recall - Kelas 1): 81%**
    * Model berhasil menangkap 81% dari seluruh pesanan yang sebenarnya batal.
* **AUC Score: 0.9260**

### 2. Model XGBoost Optimized (Final - Threshold 0.5120)
Setelah optimasi threshold ke angka **0.5120**, performa meningkat:

* **Akurasi (Accuracy): 86%** (📈 Naik 1%)
    * Model menjadi lebih handal dengan tingkat kebenaran prediksi total mencapai 86%.
* **Presisi (Precision - Kelas 1): 71%** (🎯 Lebih Efisien)
    * Efisiensi meningkat. Dengan presisi 71%, hotel dapat lebih percaya diri dalam mengambil tindakan (seperti menagih deposit) karena tingkat kesalahan prediksi "Batal" berkurang.
* **Recall (Recall - Kelas 1): 81%** (✅ Stabil Tinggi)
    * Model mampu mengidentifikasi **81% dari semua pesanan yang sebenarnya dibatalkan**. Artinya, dari 100 pesanan yang akan dibatalkan, model berhasil menangkap 81 di antaranya. Ini sangat penting untuk mitigasi *revenue leakage*.
* **AUC Score: 0.9260** (💎 Excellent)
    * Nilai AUC yang sangat tinggi menunjukkan kemampuan diskriminasi kelas yang sangat baik.

> **Kesimpulan Optimasi:**
> Perubahan threshold ke **0.5120** terbukti memberikan performa yang lebih "cerdas". Model akhir ini dipilih karena mampu **meningkatkan akurasi dan presisi** tanpa mengorbankan nilai *recall*, sehingga meminimalisir kesalahan operasional sekaligus memaksimalkan penyelamatan pendapatan hotel.

---

## 🧠 Key Insights (SHAP Analysis)

Berdasarkan analisis **SHAP (SHapley Additive exPlanations)**, berikut adalah 10 faktor yang paling memengaruhi keputusan model:

1. `country_risk` & `agent_risk`: Dua fitur teratas yang memiliki nilai tertinggi. Titik merah di sisi kanan menunjukkan bahwa jika seorang tamu berasal dari negara dengan risiko pembatalan tinggi atau memesan melalui agen yang sering bermasalah, kemungkinan besar mereka akan batal.
2. `lead_time`: Titik merah cenderung berada di sisi kanan. Ini mengonfirmasi teori bisnis bahwa semakin lama jarak antara waktu pemesanan dan hari kedatangan, semakin besar risiko tamu tersebut berubah pikiran atau menemukan opsi lain.
3. `required_car_parking_spaces`: Titik merah pada fitur ini menumpuk jauh di sisi kiri. Tamu yang meminta lahan parkir hampir tidak pernah membatalkan pesanan. Secara bisnis, tamu yang membawa mobil biasanya adalah wisatawan domestik atau keluarga yang rencana perjalanannya sudah sangat matang.
4. `room_type_changed`: Jika tamu melakukan perubahan tipe kamar (nilai merah di sisi kiri), mereka cenderung tetap datang. Aktivitas mengubah pesanan menunjukkan interaksi dan niat yang kuat dari tamu terhadap reservasi mereka.
5. `total_of_special_requests`: Semakin banyak permintaan khusus (merah), semakin kecil kemungkinan mereka batal. Tamu yang "rewel" atau punya banyak keinginan biasanya adalah tamu yang sudah pasti ingin menginap di hotel Anda.
6. `booking_changes`: Mirip dengan perubahan tipe kamar, setiap perubahan pada detail pesanan merupakan sinyal bahwa tamu tersebut aktif mengelola rencana perjalanannya dan kemungkinan besar akan hadir.
7. `customer_type_Transient`: Tamu jenis Transient (perorangan/bukan grup) yang nilainya rendah (biru) cenderung memiliki risiko batal yang lebih tinggi dibandingkan tamu grup atau korporasi.
8. `arrival_date_year` &`arrival_date_week_number`: Menunjukkan adanya pengaruh musiman. Tahun atau minggu tertentu dalam setahun memiliki tren pembatalan yang berbeda, namun pengaruhnya tidak sekuat faktor risiko negara atau permintaan fasilitas (parkir/request).
---

## 🚀 Business Recommendations

Dengan kemampuan deteksi **81% Recall**, model ini memberikan landasan kuat untuk strategi bisnis berikut:

1.  **Mengurangi Revenue Leakage:**
    * Terapkan kebijakan **Deposit Wajib / Non-Refundable** untuk reservasi yang terdeteksi sebagai *High Risk* (terutama yang memiliki `lead_time` panjang dan berasal dari `country_risk` tinggi).
2.  **Optimasi Manajemen Stok:**
    * Lakukan strategi *overbooking* yang terukur pada tanggal-tanggal dengan prediksi pembatalan tinggi, sehingga tingkat hunian kamar bisa dimaksimalkan mendekati 100%.
3.  **Loyalty Program:**
    * Berikan layanan prioritas bagi tamu yang meminta lahan parkir atau memiliki banyak *special request*, karena mereka adalah segmen tamu yang paling loyal dan minim risiko batal.
