import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import shap
import joblib
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_percentage_error, r2_score, mean_absolute_error
from sklearn.inspection import partial_dependence

# ==========================================
# 1. KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(
    page_title="Dashboard Ketahanan Pangan",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS Kustom
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    h1 { color: #2c3e50; font-family: 'Helvetica Neue', sans-serif; font-weight: 700; }
    .stDataFrame { border: 1px solid #e0e0e0; border-radius: 5px; }
    div[data-testid="stMetricValue"] { font-size: 1.4rem; }
    .stAlert { border-radius: 8px; }
    </style>
    """, unsafe_allow_html=True)

st.title("🌾 Analisis Faktor Indeks Ketahanan Pangan")
st.markdown("Dashboard interaktif berbasis **Gradient Boosting Regressor**.")
st.markdown("---")

# ==========================================
# 2. FUNGSI UTAMA
# ==========================================

@st.cache_data
def load_data():
    try:
        df = pd.read_csv('IKP 2024.csv') 
        cols_to_drop = ["Unnamed: 13", "IKP Ranking", "Komposit", "Wilayah"]
        df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
        return df
    except FileNotFoundError:
        return pd.DataFrame()

@st.cache_data
def preprocess_data(df):
    if df.empty: return None, None, None, None, None, None, None

    # 1. Pisahkan Fitur & Target
    X = df.drop(columns=['IKP'])
    y = df['IKP']
    
    # Tambahkan Intercept 
    X['Intercept'] = 1.0
        
    X = X.select_dtypes(include=[np.number])
    
    # 2. Split Data 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42
    )
    
    # 3. Scaling 
    scaler = RobustScaler()
    
    # Identifikasi kolom yang perlu di-scale 
    cols_to_scale = [c for c in X_train.columns if c != 'Intercept']
    
    # Copy data agar aman
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    # Scale hanya kolom terpilih
    X_train_scaled[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
    X_test_scaled[cols_to_scale] = scaler.transform(X_test[cols_to_scale])
    
    return X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler

@st.cache_resource
def get_models(X_train, y_train):
    # A. Baseline (Default Parameters)
    model_baseline = GradientBoostingRegressor(random_state=42)
    model_baseline.fit(X_train, y_train)
    
    # B. Tuned Model
    best_params = {
        'learning_rate': 0.11852892694151533,
        'max_depth': 2,
        'max_features': None,
        'min_samples_leaf': 2,
        'min_samples_split': 7,
        'n_estimators': 404,
        'subsample': 0.8074696377498429,
        'random_state': 42
    }
    
    model_tuned = GradientBoostingRegressor(**best_params)
    model_tuned.fit(X_train, y_train)
    
    return model_baseline, model_tuned

# ==========================================
# 3. EKSEKUSI
# ==========================================

df = load_data()

if not df.empty:
    X_train_raw, X_test_raw, X_train_final, X_test_final, y_train, y_test, scaler = preprocess_data(df)
    
    # Training
    model_baseline, model_tuned = get_models(X_train_final, y_train)

    # Sidebar
    st.sidebar.header("Navigasi")
    menu = st.sidebar.radio(
        "Menu:",
        ["1. Overview Data", "2. Preprocessing", "3. Performa & Learning Curve", "4. Feature Importance (SHAP)", "5. Partial Dependence (PDP)"]
    )
    st.sidebar.markdown("---")
    st.sidebar.caption("Oleh: M. Bagus Prayogi")

    # --- 1. OVERVIEW ---
    if menu == "1. Overview Data":
        st.header("📂 1. Dataset Overview")
        st.info("""
        **📝 Alur Analisis:**
        Tahap awal adalah memahami struktur data. Dataset ini mencakup 9 variabel pembangun IKP. 
        Statistik deskriptif di bawah digunakan untuk melihat sebaran awal sebelum pemodelan.
        """)
        st.dataframe(df.head(), use_container_width=True)
        st.markdown("### 📊 Statistik Deskriptif")
        st.dataframe(df.describe().T[['mean', 'std', 'min', 'max']].style.format("{:.2f}"), use_container_width=True)

    # --- 2. PREPROCESSING ---
    elif menu == "2. Preprocessing":
        st.header("🔍 2. Preprocessing")
        st.info("""
        **📝 Mengapa harus di preprocessing?**
        Data sosial-ekonomi seringkali memiliki *outliers*. Digunakan **Robust Scaler** agar model tidak bias terhadap nilai ekstrem.
        Grafik membandingkan distribusi data sebelum dan sesudah scaling.
        """)
        
        # Filter kolom Intercept agar tidak muncul di pilihan dropdwan
        cols_view = [c for c in X_train_raw.columns if c != 'Intercept']
        col_select = st.selectbox("Pilih Variabel:", cols_view)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Original Data**")
            st.plotly_chart(px.box(X_train_raw, y=col_select, color_discrete_sequence=['#FF6B6B']), use_container_width=True)
        with col2:
            st.markdown("**After Scaling**")
            st.plotly_chart(px.box(X_train_final, y=col_select, color_discrete_sequence=['#4ECDC4']), use_container_width=True)

    # --- 3. PERFORMA ---
    elif menu == "3. Performa & Learning Curve":
        st.header("🚀 3. Performa Model & Learning Curve")
        st.subheader("A. Perbandingan Model: Default vs Tuned")
        st.info("""
        **📝 Evaluasi Model:**
        Membandingkan model Baseline (Default) vs Tuned (Optimasi).
        """)
        
        def get_metrics(model, X, y):
            pred = model.predict(X)
            return {
                "R2": r2_score(y, pred),
                "MAE": mean_absolute_error(y, pred),
                "MAPE": mean_absolute_percentage_error(y, pred)
            }
            
        res_base = get_metrics(model_baseline, X_test_final, y_test)
        res_tuned = get_metrics(model_tuned, X_test_final, y_test)
        
        data_compare = {
            "R2": [res_base['R2'], res_tuned['R2']],
            "MAE": [res_base['MAE'], res_tuned['MAE']],
            "MAPE": [res_base['MAPE'], res_tuned['MAPE']]
        }
        df_compare = pd.DataFrame(data_compare, index=["Gradient Boosting (BHT)", "Gradient Boosting (AHT)"])
        
        c1, c2, c3 = st.columns(3)
        c1.metric("R2 Score (Tuned)", f"{res_tuned['R2']:.4f}", delta=f"{res_tuned['R2']-res_base['R2']:.4f}")
        c2.metric("MAE (Tuned)", f"{res_tuned['MAE']:.4f}", delta=f"{res_tuned['MAE']-res_base['MAE']:.4f}", delta_color="inverse")
        c3.metric("MAPE (Tuned)", f"{res_tuned['MAPE']*100:.2f}%", delta=f"{(res_tuned['MAPE']-res_base['MAPE'])*100:.2f}%", delta_color="inverse")
        
        st.write("Tabel Perbandingan:")
        st.dataframe(df_compare.style.format("{:.6f}"), use_container_width=True)
        
        st.markdown("---")
        st.subheader("B. Learning Curve")
        st.info("""
        **📝 Diagnosa Overfitting:**
        Grafik ini menunjukkan apakah model mengalami Overfitting (Gap besar) atau Good Fit (Garis berdekatan).
        """)
        
        with st.spinner("Menghitung Learning Curve..."):
            train_sizes, train_scores, test_scores = learning_curve(
                estimator=model_tuned,
                X=X_train_final,
                y=y_train,
                cv=5,
                scoring="neg_mean_absolute_percentage_error",
                train_sizes=np.linspace(0.1, 1.0, 10),
                n_jobs=-1,
                shuffle=True,
                random_state=42
            )

            # Kali 100 untuk Persen
            train_mape = -np.mean(train_scores, axis=1) * 100
            test_mape = -np.mean(test_scores, axis=1) * 100
            train_std = np.std(train_scores, axis=1) * 100
            test_std = np.std(test_scores, axis=1)

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(train_sizes, train_mape, 'o-', color="blue", label="MAPE Pelatihan")
            ax.plot(train_sizes, test_mape, 'o-', color="red", label="MAPE Validasi")
            ax.fill_between(train_sizes, train_mape - train_std, train_mape + train_std, alpha=0.15, color="blue")
            ax.fill_between(train_sizes, test_mape - test_std, test_mape + test_std, alpha=0.15, color="red")
            
            ax.text(train_sizes[-1], test_mape[-1], f"{test_mape[-1]:.2f}%", fontsize=10, color='red', ha="left", va="bottom")
            ax.text(train_sizes[-1], train_mape[-1], f"{train_mape[-1]:.2f}%", fontsize=10, color='blue', ha="left", va="top")
            
            ax.set_title("Kurva Pembelajaran (MAPE %)", fontsize=14)
            ax.set_xlabel("Jumlah Data Latih")
            ax.set_ylabel("MAPE (%)")
            ax.legend(loc="best")
            ax.grid(True, linestyle='--', alpha=0.7)
            st.pyplot(fig)

    # --- 4. SHAP IMPORTANCE ---
    elif menu == "4. Feature Importance (SHAP)":
        st.header("⭐ 4. Feature Importance (SHAP)")
        st.info("""
        **📝 Interpretasi Fitur:**
        Menggunakan **SHAP Value** untuk melihat kontribusi setiap fitur.
        """)
        
        with st.spinner("Menghitung SHAP Values (Menggunakan X_train sebagai background)..."):
            # PERBAIKAN: Masukkan X_train_final ke TreeExplainer (Sama persis notebook)
            explainer = shap.TreeExplainer(model_tuned, X_train_final)
            shap_values = explainer.shap_values(X_test_final)
            
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            
            df_imp = pd.DataFrame({
                "Feature": X_test_final.columns,
                "Mean(|SHAP|)": mean_abs_shap
            })
            
            # FILTER INTERCEPT 
            df_imp = df_imp[~df_imp["Feature"].str.lower().isin(["intercept", "base value", "expected value"])]
            df_imp = df_imp.sort_values(by="Mean(|SHAP|)", ascending=False)
            
            col_g, col_t = st.columns([2, 1])
            with col_g:
                fig = px.bar(
                    df_imp.head(10).sort_values(by="Mean(|SHAP|)", ascending=True), 
                    x="Mean(|SHAP|)", y="Feature", orientation='h', 
                    title="Top 10 Feature Importance (SHAP)",
                    color="Mean(|SHAP|)", color_continuous_scale="Blues"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col_t:
                st.dataframe(df_imp.style.format({"Mean(|SHAP|)": "{:.4f}"}), use_container_width=True, height=400)
                
            if 'top_3_features' not in st.session_state:
                st.session_state['top_3_features'] = df_imp['Feature'].head(3).tolist()
            
   # --- 5. PDP ---
    elif menu == "5. Partial Dependence (PDP)":
        st.header("📈 5. Partial Dependence Plot (PDP)")
        st.info("""
        **📝 Analisis Kausalitas:**
        Melihat arah pengaruh (Positif/Negatif) fitur terhadap IKP dalam **Skala Asli**.
        """)
        
        # Ambil fitur selain Intercept untuk diplot
        # Karena Intercept tidak punya makna fisik untuk di-plot PDP-nya
        valid_features = [c for c in X_train_final.columns if c != 'Intercept']
        
        if 'top_3_features' in st.session_state:
            # Filter agar top 3 tidak memuat Intercept (jika ada)
            top_features = [f for f in st.session_state['top_3_features'] if f in valid_features]
            if not top_features: top_features = valid_features[:3]
        else:
            top_features = valid_features[:3]
            
        feature_to_plot = st.selectbox("Pilih Fitur:", top_features)
        
        if st.button(f"Generate PDP untuk {feature_to_plot}"):
            with st.spinner("Mengkalkulasi..."):
                grid_resolution = 20
                
                # 1. Hitung PDP 
                pd_results = partial_dependence(
                    model_tuned, X_train_final, [feature_to_plot], 
                    grid_resolution=grid_resolution, kind="average"
                )
                
                pdp_values = pd_results["average"][0]
                x_scaled = pd_results["grid_values"][0]
                
                # 2. INVERSE TRANSFORM LOGIC
                cols_used_in_scaler = [c for c in X_train_final.columns if c != 'Intercept']
                
                if feature_to_plot in cols_used_in_scaler:
                    # Cari urutan index fitur ini di dalam list Scaler (0 sampai 8)
                    idx_in_scaler = cols_used_in_scaler.index(feature_to_plot)
                    
                    # Buat dummy array dengan ukuran yang PAS dengan Scaler (misal 9 kolom)
                    dummy = np.zeros((len(x_scaled), len(cols_used_in_scaler)))
                    dummy[:, idx_in_scaler] = x_scaled
                    
                    # Inverse Transform aman karena bentuknya cocok
                    x_unscaled = scaler.inverse_transform(dummy)[:, idx_in_scaler]
                else:
                    # Jika fitur tidak di-scale (misal kategorikal/intercept), pakai nilai asli
                    x_unscaled = x_scaled
                
                # 3. Plotting
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.plot(x_unscaled, pdp_values, marker="o", linewidth=2, color="#2c3e50", zorder=2)
                
                # Anotasi
                idxs = [0, len(x_unscaled)//2, -1]
                for i in idxs:
                    val_x = x_unscaled[i]
                    val_y = pdp_values[i]
                    ax.annotate(
                        f"{val_x:.1f}\n{val_y:.2f}", xy=(val_x, val_y), xytext=(0, 15),
                        textcoords="offset points", ha='center',
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8)
                    )
                    ax.scatter([val_x], [val_y], color='red', zorder=3)

                ax.set_xlabel(f"{feature_to_plot} (Skala Asli)"); ax.set_ylabel("Prediksi IKP")
                ax.set_title(f"Pengaruh {feature_to_plot} terhadap IKP")
                ax.grid(True, linestyle="--", alpha=0.5)
                st.pyplot(fig)

else:
    st.warning("Data belum dimuat.")