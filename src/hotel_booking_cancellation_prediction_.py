import os
import pickle
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report, 
                             roc_curve, auc, precision_recall_curve)
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier

# Konfigurasi Global
warnings.filterwarnings('ignore')
RANDOM_STATE = 42
DATA_URL = 'https://raw.githubusercontent.com/DarleneHegan/Hotel-Booking-Cancellation-Prediction/refs/heads/main/Data/hotel_bookings.csv'

# Buat folder untuk output jika belum ada
os.makedirs('models', exist_ok=True)
os.makedirs('docs/images', exist_ok=True)

def load_data(url):
    """Memuat dataset dari URL."""
    print("⏳ Loading dataset...")
    df = pd.read_csv(url)
    print(f"✅ Dataset dimuat: {df.shape[0]} baris, {df.shape[1]} kolom")
    return df

def initial_cleaning(df):
    """Membersihkan data awal sebelum split (menghapus anomali dan kolom tidak berguna)."""
    print("🧹 Melakukan Initial Cleaning...")
    
    # Drop missing values spesifik dan kolom company
    df.dropna(subset=['country', 'children'], inplace=True)
    if 'company' in df.columns:
        df.drop(columns=['company'], inplace=True)
    
    # Drop duplikat
    df.drop_duplicates(inplace=True)
    
    # Konversi tanggal
    df['reservation_status_date'] = pd.to_datetime(df['reservation_status_date'])
    
    # Filter data tamu yang valid
    filter_guests = (df['adults'] + df['children'] + df['babies']) > 0
    df = df[filter_guests]
    
    # Filter ADR yang wajar
    filter_adr = (df['adr'] >= 0) & (df['adr'] < 5000)
    df = df[filter_adr]
    
    # Filter Adults
    df = df[df['adults'] <= 50]
    
    return df

def feature_engineering(df):
    """Membuat fitur baru."""
    # Fitur Tamu
    df['total_guests'] = df['adults'] + df['children'] + df['babies']
    df['is_family'] = np.where((df['children'] > 0) | (df['babies'] > 0), 1, 0)
    
    # Fitur History
    df['total_previous_bookings'] = df['previous_cancellations'] + df['previous_bookings_not_canceled']
    # Tambah epsilon kecil untuk menghindari pembagian nol
    df['cancellation_rate_history'] = df['previous_cancellations'] / (df['total_previous_bookings'] + 0.01)
    
    # Fitur Perubahan
    df['room_type_changed'] = np.where(df['reserved_room_type'] != df['assigned_room_type'], 1, 0)
    
    # Fitur Agen
    df['has_agent'] = np.where(df['agent'].notnull() & (df['agent'] != 0), 1, 0)
    
    # Fitur Harga
    df['price_per_person'] = df['adr'] / (df['total_guests'] + 0.1)
    
    # Fitur Musim (Peak Season: July, August)
    df['is_peak_season'] = df['arrival_date_month'].apply(lambda x: 1 if x in ['July', 'August'] else 0)
    
    # Fitur Total Nights (dengan Capping)
    df['total_nights'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']
    df.loc[df['total_nights'] > 50, 'total_nights'] = 50
    
    return df

def calc_smooth_mean(df, by, target, m=10):
    """Menghitung Target Encoding dengan Smoothing."""
    agg = df.groupby(by)[target].agg(['count', 'mean'])
    counts = agg['count']
    means = agg['mean']
    global_mean = df[target].mean()
    smooth = (counts * means + m * global_mean) / (counts + m)
    return smooth

def preprocess_split(df):
    """Melakukan splitting, imputasi, dan encoding."""
    print("⚙️ Preprocessing dan Splitting Data...")
    
    X = df.drop(columns=['is_canceled'])
    y = df['is_canceled']
    
    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
    )
    
    # 1. Imputasi Agent (Median)
    imputer_agent = SimpleImputer(strategy='median')
    X_train['agent'] = imputer_agent.fit_transform(X_train[['agent']])
    X_test['agent'] = imputer_agent.transform(X_test[['agent']])
    
    # 2. Transformasi Log & Sqrt
    X_train['lead_time'] = np.log1p(X_train['lead_time'])
    X_test['lead_time'] = np.log1p(X_test['lead_time'])
    
    X_train['adr'] = np.sqrt(X_train['adr'])
    X_test['adr'] = np.sqrt(X_test['adr'])
    
    # 3. Handling Outlier (Capping Manual)
    for col in ['babies', 'required_car_parking_spaces']:
        X_train.loc[X_train[col] > 3, col] = 3
        X_test.loc[X_test[col] > 3, col] = 3
        
    # 4. Feature Engineering (diterapkan ke X_train dan X_test)
    X_train = feature_engineering(X_train)
    X_test = feature_engineering(X_test)
    
    # 5. Target Encoding (Smoothing) untuk Country & Agent
    # Perlu gabung X dan y sementara untuk hitung mean
    train_temp = X_train.copy()
    train_temp['target'] = y_train
    
    country_map = calc_smooth_mean(train_temp, by='country', target='target', m=10)
    agent_map = calc_smooth_mean(train_temp, by='agent', target='target', m=10)
    
    X_train['country_risk'] = X_train['country'].map(country_map)
    X_test['country_risk'] = X_test['country'].map(country_map)
    
    X_train['agent_risk'] = X_train['agent'].map(agent_map)
    X_test['agent_risk'] = X_test['agent'].map(agent_map)
    
    # Isi missing value di test set akibat kategori baru dengan rata-rata global
    global_mean = y_train.mean()
    X_test['country_risk'] = X_test['country_risk'].fillna(global_mean)
    X_test['agent_risk'] = X_test['agent_risk'].fillna(global_mean)
    
    # 6. Drop Kolom yang sudah di-encode atau tidak perlu
    cols_to_drop = ['assigned_room_type', 'reservation_status', 'reservation_status_date', 
                    'country', 'agent']
    X_train.drop(columns=cols_to_drop, errors='ignore', inplace=True)
    X_test.drop(columns=cols_to_drop, errors='ignore', inplace=True)
    
    # 7. Map Bulan
    month_map = {
        'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
        'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12
    }
    X_train['arrival_date_month'] = X_train['arrival_date_month'].map(month_map)
    X_test['arrival_date_month'] = X_test['arrival_date_month'].map(month_map)
    
    # 8. One-Hot Encoding
    X_train = pd.get_dummies(X_train, drop_first=True)
    X_test = pd.get_dummies(X_test, drop_first=True)
    
    # Align Columns (Pastikan kolom train dan test sama)
    X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)
    
    return X_train, X_test, y_train, y_test

def train_and_evaluate(X_train, X_test, y_train, y_test):
    """Melatih model, seleksi fitur, dan evaluasi."""
    
    # --- Tahap 1: Initial Model untuk Feature Importance ---
    print("🚀 Training Initial Model untuk Feature Selection...")
    initial_model = XGBClassifier(
        n_estimators=300, max_depth=9, learning_rate=0.05,
        scale_pos_weight=1.5, use_label_encoder=False,
        eval_metric='logloss', random_state=RANDOM_STATE
    )
    initial_model.fit(X_train, y_train)
    
    # Seleksi Fitur (> 0.005)
    importances = pd.Series(initial_model.feature_importances_, index=X_train.columns)
    selected_features = importances[importances > 0.005].index.tolist()
    print(f"✅ Fitur terpilih: {len(selected_features)} dari {len(X_train.columns)}")
    
    X_train_sel = X_train[selected_features]
    X_test_sel = X_test[selected_features]
    
    # --- Tahap 2: Final Model Training ---
    print("🚀 Training Final XGBoost Model...")
    final_model = XGBClassifier(
        n_estimators=500,
        max_depth=10,
        learning_rate=0.03,
        scale_pos_weight=2.0,
        min_child_weight=2,
        gamma=0.1,
        subsample=0.85,
        colsample_bytree=0.85,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=RANDOM_STATE
    )
    final_model.fit(X_train_sel, y_train)
    
    # --- Tahap 3: Threshold Tuning ---
    print("🔧 Melakukan Threshold Tuning...")
    y_probs = final_model.predict_proba(X_test_sel)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_probs)
    
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-5)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    
    print(f"🏆 Optimal Threshold: {best_threshold:.4f}")
    
    # Prediksi Final dengan Threshold Baru
    y_pred_tuned = (y_probs >= best_threshold).astype(int)
    
    # Report
    print("\n======= FINAL CLASSIFICATION REPORT =======")
    print(classification_report(y_test, y_pred_tuned))
    
    # --- Tahap 4: Visualisasi & Saving Plot ---
    save_evaluation_plots(y_test, y_pred_tuned, y_probs)
    save_shap_plot(final_model, X_test_sel)
    
    return final_model, X_train_sel

def save_evaluation_plots(y_test, y_pred, y_probs):
    """Menyimpan Confusion Matrix dan ROC Curve."""
    cm = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    
    # Confusion Matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax[0],
                xticklabels=['Not Canceled', 'Canceled'],
                yticklabels=['Not Canceled', 'Canceled'])
    ax[0].set_title('Confusion Matrix (Tuned)')
    ax[0].set_ylabel('Actual')
    ax[0].set_xlabel('Predicted')
    
    # ROC Curve
    ax[1].plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.4f}')
    ax[1].plot([0, 1], [0, 1], color='navy', linestyle='--')
    ax[1].set_title('ROC Curve')
    ax[1].legend(loc="lower right")
    
    plt.tight_layout()
    plt.savefig('docs/images/evaluation_plots.png')
    print("💾 Plot evaluasi disimpan di docs/images/evaluation_plots.png")
    plt.close()

def save_shap_plot(model, X_test):
    """Menyimpan SHAP Summary Plot."""
    print("🎨 Generating SHAP Values...")
    # Gunakan sampel kecil agar cepat
    X_sample = X_test.sample(n=1000, random_state=RANDOM_STATE)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, max_display=10, show=False)
    plt.title("SHAP Summary Plot")
    plt.tight_layout()
    plt.savefig('docs/images/shap_summary.png')
    print("💾 SHAP plot disimpan di docs/images/shap_summary.png")
    plt.close()

def run_cross_validation(model, X, y):
    """Menjalankan Cross Validation untuk cek stabilitas."""
    print("🔄 Menjalankan 5-Fold Cross Validation...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_results = cross_val_score(model, X, y, cv=skf, scoring='f1_weighted')
    
    print(f"📊 CV Mean F1-Score: {cv_results.mean():.4f}")
    print(f"📊 CV Std Dev: {cv_results.std():.4f}")

if __name__ == "__main__":
    print("=== MULAI PROSES TRAINING MODEL ===")
    
    # 1. Load
    df = load_data(DATA_URL)
    
    # 2. Initial Cleaning
    df = initial_cleaning(df)
    
    # 3. Preprocessing & Splitting
    X_train, X_test, y_train, y_test = preprocess_split(df)
    
    # 4. Training & Evaluation
    final_model, X_train_final = train_and_evaluate(X_train, X_test, y_train, y_test)
    
    # 5. Cross Validation (pada data train yg sudah diseleksi)
    run_cross_validation(final_model, X_train_final, y_train)
    
    # 6. Save Model
    model_path = 'models/hotel_cancellation_prediction_model.pkl'
    pickle.dump(final_model, open(model_path, 'wb'))
    print(f"💾 Model berhasil disimpan di {model_path}")
    
    print("=== PROSES SELESAI ===")