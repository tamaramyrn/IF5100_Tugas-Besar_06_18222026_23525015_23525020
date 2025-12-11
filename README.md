# 🚀 Prediksi Churn Pelanggan Telekomunikasi (Customer Churn Prediction)

### Tugas Besar IF5100 Pemrograman untuk Data Analitik - Kelompok 6

Proyek ini bertujuan untuk membangun model pembelajaran mesin yang akurat untuk memprediksi *churn* pelanggan dalam industri telekomunikasi, sebuah metrik penting untuk retensi dan strategi bisnis.

## 👥 Anggota Kelompok

* **Tamara Mayranda Lubis** (18222026)
* **Tantan Nugraha** (23525015)
* **Era Desti Ramayani** (23525020)

## 🎯 Tujuan Proyek

1.  Melakukan eksplorasi, pembersihan, dan pra-pemrosesan data (*Exploratory Data Analysis* & *Preprocessing*).
2.  Mengatasi masalah ketidakseimbangan kelas (*imbalanced data*) menggunakan teknik **SMOTE**.
3.  Mengevaluasi model klasifikasi dasar (LR, RF, GBC, XGBoost) dan menyempurnakannya melalui **Grid Search Hyperparameter Tuning**.
4.  Meningkatkan performa melalui teknik **Ensemble Learning** (Voting dan Stacking Classifier).
5.  Mengidentifikasi model paling optimal untuk prediksi *churn*.

## 🛠️ Metodologi dan Pendekatan

### 1. Pra-pemrosesan Data
* **Pengecekan dan Penanganan Data Awal:**
    * Mengidentifikasi kolom `TotalCharges` sebagai tipe data `object` yang seharusnya `float` dan memiliki nilai *whitespaces*.
    * Mengonversi `TotalCharges` ke tipe numerik dan mengatasi nilai yang tidak dapat dikonversi dengan `NaN`.
    * Mengatasi nilai yang hilang (`NaN`) pada kolom `TotalCharges` dengan nilai median.
* **Encoding Variabel Kategorikal:**
    * Melakukan *Binary Encoding* (Yes: 1, No: 0) pada kolom biner (`Partner`, `Dependents`, `PhoneService`, `PaperlessBilling`, `Churn`, dan `gender`).
    * Melakukan *One-Hot Encoding* untuk variabel multiclass (`MultipleLines`, `InternetService`, `Contract`, dll.).
* **Feature Engineering:**
    * Membuat fitur `ChargeDelta` (selisih antara `MonthlyCharges` saat ini dan `AvgMonthlyCharges` historis) untuk mendeteksi potensi *'Bill Shock'* yang dapat meningkatkan risiko *churn*.
    * Mengelompokkan `tenure` menjadi kategori biner (`New Customer`: 0-12 bulan, `Mid-Term`: 13-48 bulan, `Loyal/Established`: >48 bulan) untuk menangkap pola risiko *churn* yang non-linear.
* **Penskalaan Data:** Menggunakan `StandardScaler` (implisit melalui pipeline atau terpisah) untuk menormalkan fitur numerik.

### 2. Eksplorasi Data Analitik (EDA) dan Visualisasi

EDA dilakukan untuk memahami struktur data dan mengidentifikasi pendorong utama *churn*.

* **Distribusi Variabel Target:** Data menunjukkan **ketidakseimbangan kelas** yang signifikan, dengan 73% pelanggan *Non-Churn* (`No`) dan 26% pelanggan *Churn* (`Yes`).
* **Analisis Korelasi:**
    * **Pendorong *Churn* Terkuat (Korelasi Positif):** `Tenure_Group_New Customer` (r ≈ 0.32), `InternetService_Fiber optic` (r ≈ 0.31), dan `PaymentMethod_Electronic check` (r ≈ 0.30).
    * **Pencegah *Churn* Terkuat (Korelasi Negatif):** `tenure` (lama berlangganan) (r ≈ -0.35), `Contract_Two year` (r ≈ -0.30), dan `TotalCharges` (r ≈ -0.20).
* **Visualisasi Kunci:**
    * **Kontrak vs Churn (Stacked Bar Chart):** Pelanggan dengan tipe kontrak **Month-to-month** memiliki **proporsi *churn* tertinggi**, menunjukkan mereka lebih memilih untuk berhenti berlangganan dibandingkan yang memiliki kontrak tahunan atau dua tahunan.     * **Distribusi Biaya Bulanan vs Churn (Density Plot):** Pelanggan yang **Churn** cenderung memiliki **Biaya Bulanan (`MonthlyCharges`) yang tinggi** (sekitar $70-$110), sementara pelanggan yang tidak *churn* memiliki distribusi yang lebih merata.     * **Distribusi Tenure vs Churn (Density Plot):** Pelanggan baru dengan **Durasi Berlangganan (`Tenure`) yang sangat singkat** memiliki **probabilitas *churn* yang jauh lebih tinggi**. Pelanggan yang loyal/tidak *churn* umumnya memiliki *tenure* yang lebih lama.     * **Layanan Internet vs Churn (Pie Charts):** Pelanggan dengan layanan **Fiber optic** memiliki **tingkat *churn* tertinggi** (41.9%), sedangkan pelanggan yang **Tidak Ada** layanan internet menunjukkan **tingkat *churn* terendah** (7.4%).
* **Penanganan Imbalance Data:**
    * Teknik **SMOTE (Synthetic Minority Over-sampling Technique)** digunakan untuk menyeimbangkan data pelatihan, meningkatkan jumlah sampel kelas minoritas (`Churn=1`) menjadi setara dengan kelas mayoritas (`Churn=0`), yaitu **3614 sampel** untuk setiap kelas.

### 3. Pemodelan dan Tuning

Model-model klasifikasi dilatih menggunakan data yang telah di-*scaling* dan di-*resampling* (`X_train_smote`, `y_train_smote`).

* **Model Dasar:** `RandomForestClassifier` (RF), `LogisticRegression` (LR).
* **Hyperparameter Tuning (Grid Search):** Dilakukan pada model *boosting* (`GradientBoostingClassifier` dan `XGBClassifier`) untuk mencari kombinasi parameter terbaik berdasarkan metrik `scoring='roc_auc'`.

### 4. Ensemble Learning

* **Voting Classifier (Soft Voting):** Menggabungkan prediksi dari LR, RF, GBC (Tuned), dan XGBoost (Tuned). Prediksi akhir didasarkan pada rata-rata probabilitas kelas, memaksimalkan kekuatan kolektif model.
* **Stacking Classifier:** Menggunakan LR, RF, GBC (Tuned), dan XGBoost (Tuned) sebagai model dasar (Level 0), dengan `LogisticRegression` sebagai meta-model (Level 1) untuk mempelajari cara menggabungkan prediksi.

## 📊 Hasil Utama dan Evaluasi Model

Evaluasi model didasarkan pada metrik **ROC AUC Score** (kemampuan membedakan kelas) dan **Recall Class 1** (kemampuan mengidentifikasi *churn* sebenarnya).

| Model | ROC AUC Score | Accuracy | Precision (Class 1) | Recall (Class 1) | F1-Score (Class 1) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Voting Classifier** | **0.8167** | 0.8251 | 0.5370 | **0.6774** | **0.6017** |
| Stacking Classifier | 0.8160 | 0.8207 | 0.5263 | 0.6728 | 0.5902 |
| XGBoost (Tuned) | 0.8157 | 0.8220 | 0.5290 | 0.6613 | 0.5878 |
| Gradient Boosting (Tuned) | 0.8151 | 0.8171 | 0.5248 | 0.6684 | 0.5881 |
| Random Forest | 0.8143 | 0.8242 | 0.5401 | 0.6549 | 0.5922 |
| Logistic Regression | 0.7960 | 0.7674 | 0.4795 | 0.7250 | 0.5787 |

### 🏆 Kesimpulan Model Optimal

Model **Voting Classifier** menunjukkan **performa paling optimal** (AUC tertinggi dan Recall Class 1 tertinggi). Mengingat pentingnya metrik **Recall** dalam masalah *churn* (menghindari kerugian akibat gagal mengidentifikasi pelanggan yang akan *churn*), Voting Classifier adalah model yang paling direkomendasikan untuk implementasi akhir.

## 💻 Cara Menjalankan

### 1. Persyaratan (Prerequisites)

Pastikan Anda telah menginstal Python (disarankan versi 3.8+) dan paket-paket berikut:

```bash
pip install pandas numpy scikit-learn xgboost imbalanced-learn matplotlib seaborn