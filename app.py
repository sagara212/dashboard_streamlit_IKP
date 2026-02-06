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

# 1. KONFIGURASI HALAMAN
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

# 2. FUNGSI UTAMA
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
    
    cols_to_scale = [c for c in X_train.columns if c != 'Intercept']
    
    # Copy data
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

# 3. EKSEKUSI
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
    if "1." in menu:
        st.subheader("📂 Overview Dataset")
        st.markdown("Ringkasan statistik data IKP sebelum dilakukan pemodelan.")
        
        # Metric Cards Layout
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f'<div class="metric-card"><div class="metric-label">Total Data</div><div class="metric-value">{df.shape[0]}</div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="metric-card"><div class="metric-label">Jumlah Fitur</div><div class="metric-value">{df.shape[1]-1}</div></div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="metric-card"><div class="metric-label">Rata-rata IKP</div><div class="metric-value">{df["IKP"].mean():.2f}</div></div>', unsafe_allow_html=True)
        c4.markdown(f'<div class="metric-card"><div class="metric-label">Rentang IKP</div><div class="metric-value">{df["IKP"].min():.0f}-{df["IKP"].max():.0f}</div></div>', unsafe_allow_html=True)
        
        st.write("")
        col_tab1, col_tab2 = st.tabs(["📊 Statistik Deskriptif", "📋 Tabel Data"])
        
        with col_tab1:
            # FIX: Rename '50%' menjadi 'median' agar bisa dipanggil
            stats = df.describe().T
            stats = stats.rename(columns={'50%': 'median'})
            
            # Tampilkan Tabel
            st.dataframe(stats[['mean', 'median', 'std', 'min', 'max']].style.background_gradient(cmap="Greens", subset=['mean', 'max']), use_container_width=True)
            
            # Insight Box (Dark Mode)
            st.markdown("""
            <div style="background-color: #1f2937; padding: 20px; border-radius: 12px; border: 1px solid #374151; border-left: 5px solid #3b82f6; margin-top: 15px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);">
                <h4 style="color: #60a5fa; margin-top: 0; margin-bottom: 12px;">💡 Insight: Statistik Deskriptif</h4>
                <p style="color: #cbd5e1; font-size: 15px; line-height: 1.6; text-align: justify; margin-bottom: 15px;">
                    Data tahun 2024 ini menyingkap <b style="color: #f8fafc;">kesenjangan tajam</b> antar wilayah di Indonesia. 
                    Meskipun rata-rata nasional terlihat stabil, terdapat anomali ekstrem di daerah tertentu:
                </p>
                <ul style="color: #cbd5e1; font-size: 15px; line-height: 1.6; padding-left: 20px;">
                    <li style="margin-bottom: 8px;">
                        <b style="color: #93c5fd;">Ketersediaan Pangan (NPCR):</b> Rata-rata nasional <b>1,33</b>, namun median <b>0,67</b> menunjukkan data miring ke nilai rendah. 
                        Max <b>5,00</b> mengindikasikan adanya wilayah dengan <b>defisit pangan ekstrem</b> (konsumsi > produksi).
                    </li>
                    <li style="margin-bottom: 8px;">
                        <b style="color: #93c5fd;">Akses & Kerentanan Ekonomi:</b> Tantangan serius terlihat pada tingkat kemiskinan (Max <b>40,01%</b>) dan proporsi pengeluaran pangan >65% yang mencapai angka ekstrem <b>93,04%</b>. 
                        Ini sinyal kuat adanya wilayah dengan daya beli sangat rendah.
                    </li>
                    <li style="margin-bottom: 8px;">
                        <b style="color: #93c5fd;">Infrastruktur (Gap Paling Mencolok):</b> Akses listrik mayoritas baik (median rumah tangga tanpa listrik hanya 0,11%), namun ada wilayah tertinggal dengan angka <b>70,03%</b>. 
                        Lebih kritis lagi, wilayah tanpa akses air bersih mencapai <b>99,62%</b>.
                    </li>
                    <li>
                        <b style="color: #93c5fd;">Kesehatan vs Pendidikan:</b> Stunting masih krusial (Rata-rata <b>22,34%</b>, Max <b>50,20%</b>) diperburuk oleh distribusi Nakes yang timpang (Std Dev <b>8,74</b>). 
                        Sebaliknya, <b>Lama Sekolah Perempuan</b> adalah indikator paling merata (Std Dev terendah <b>1,54</b>).
                    </li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            # -----------------------------------------------------

        with col_tab2:
            st.dataframe(df, use_container_width=True)

        with col_tab2:
            st.dataframe(df, use_container_width=True)

    # --- 2. PREPROCESSING ---
    elif menu == "2. Preprocessing":
        st.header("🔍 2. Preprocessing & EDA")
        
        # Pilihan Variabel
        cols_view = [c for c in X_train_raw.columns if c != 'Intercept']
        col_select = st.selectbox("Pilih Variabel untuk Analisis:", cols_view)
        
        # Visualisasi Side-by-Side
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 1. Data Asli (Raw)")
            fig_raw = px.box(X_train_raw, y=col_select, color_discrete_sequence=['#ef4444'], template="plotly_dark")
            fig_raw.update_layout(height=400, margin=dict(l=20, r=20, t=20, b=20), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_raw, use_container_width=True)
            
        with col2:
            st.markdown("#### 2. Setelah Scaling (Robust)")
            fig_scaled = px.box(X_train_final, y=col_select, color_discrete_sequence=['#3b82f6'], template="plotly_dark")
            fig_scaled.update_layout(height=400, margin=dict(l=20, r=20, t=20, b=20), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_scaled, use_container_width=True)

        st.markdown("""
<div style="background-color: #1f2937; padding: 20px; border-radius: 12px; border: 1px solid #374151; border-left: 5px solid #10b981; margin-top: 20px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);">
    <h4 style="color: #34d399; margin-top: 0; margin-bottom: 12px;">💡 Insight: Analisis Sebaran & Strategi Scaling</h4>
    <p style="color: #cbd5e1; font-size: 15px; line-height: 1.6; text-align: justify; margin-bottom: 15px;">
        Visualisasi Boxplot di atas digunakan untuk mendeteksi karakteristik data. Area kotak menunjukkan rentang nilai normal, sedangkan 
        titik-titik di luar garis adalah <b style="color: #fca5a5;">pencilan (outlier)</b>, yang mengindikasikan adanya wilayah dengan kondisi ekstrem.
    </p>
    <ul style="color: #cbd5e1; font-size: 15px; line-height: 1.6; padding-left: 20px;">
        <li style="margin-bottom: 8px;">
            <b style="color: #fca5a5;">Masalah (Bias Skala):</b> Terindikasi perbedaan skala antar variabel yang tajam dan banyaknya outlier. 
            Jika langsung digunakan, model Gradient Boosting akan cenderung <b>bias</b> terhadap variabel berskala besar.
        </li>
        <li style="margin-bottom: 8px;">
            <b style="color: #60a5fa;">Solusi (Robust Scaler):</b> Dilakukan standarisasi menggunakan metode <b>Robust Scaler</b> (berbasis Median & IQR). 
            Metode ini efektif menyamakan skala tanpa membuang informasi penting pada data ekstrem tersebut.
        </li>
        <li>
            <b style="color: #34d399;">Hasil Akhir:</b> Seperti terlihat pada Gambar 2, seluruh variabel kini berada dalam rentang skala yang relatif sama dan terpusat di sekitar median (0). 
            Hal ini memastikan setiap variabel memiliki <b style="color: #f8fafc;">bobot yang adil</b> dalam proses pelatihan model.
        </li>
    </ul>
</div>
""", unsafe_allow_html=True)
        # -----------------------------------------------------

# --- 3. PERFORMA ---
    elif menu == "3. Performa & Learning Curve":
        st.header("🚀 3. Performa Model & Learning Curve")
        
        # --- SUB A: PERBANDINGAN MODEL ---
        st.subheader("A. Perbandingan Model: Default vs Tuned")
        st.info("""
        **📝 Evaluasi Model:**
        Membandingkan model Baseline (Default) vs Tuned (Optimasi).
        """)
        
        # Fungsi Hitung Metrik
        def get_metrics(model, X, y):
            pred = model.predict(X)
            return {
                "R2": r2_score(y, pred),
                "MAE": mean_absolute_error(y, pred),
                "MAPE": mean_absolute_percentage_error(y, pred)
            }
            
        res_base = get_metrics(model_baseline, X_test_final, y_test)
        res_tuned = get_metrics(model_tuned, X_test_final, y_test)
        
        df_compare = pd.DataFrame({
            "R2": [res_base['R2'], res_tuned['R2']],
            "MAE": [res_base['MAE'], res_tuned['MAE']],
            "MAPE": [res_base['MAPE'], res_tuned['MAPE']]
        }, index=["Gradient Boosting (Baseline)", "Gradient Boosting (Tuned)"])
        
        # 1. TAMPILAN KARTU METRIK
        c1, c2, c3 = st.columns(3)
        c1.metric("R2 Score (Tuned)", f"{res_tuned['R2']:.4f}", delta=f"{res_tuned['R2']-res_base['R2']:.4f}")
        c2.metric("MAE (Tuned)", f"{res_tuned['MAE']:.4f}", delta=f"{res_tuned['MAE']-res_base['MAE']:.4f}", delta_color="inverse")
        c3.metric("MAPE (Tuned)", f"{res_tuned['MAPE']*100:.2f}%", delta=f"{(res_tuned['MAPE']-res_base['MAPE'])*100:.2f}%", delta_color="inverse")
        
        st.write("Tabel Perbandingan:")
        st.dataframe(df_compare.style.format("{:.6f}"), use_container_width=True)

        # --- BAGIAN DESKRIPSI TABEL (INSIGHT) ---
        st.markdown("""
        <div style="background-color: #1f2937; padding: 20px; border-radius: 12px; border: 1px solid #374151; border-left: 5px solid #3b82f6; margin-top: 20px; margin-bottom: 20px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);">
            <h4 style="color: #60a5fa; margin-top: 0; margin-bottom: 12px;">💡 Insight: Hyperparameter Tuning</h4>
            <p style="color: #cbd5e1; font-size: 15px; line-height: 1.6; text-align: justify; margin-bottom: 15px;">
                Berdasarkan tabel di atas, proses <i>hyperparameter tuning</i> memberikan dampak positif signifikan, menyempurnakan performa model baseline yang sebelumnya sudah tergolong tinggi. 
                Berikut detail peningkatannya:
            </p>
            <ul style="color: #cbd5e1; font-size: 15px; line-height: 1.6; padding-left: 20px;">
                <li style="margin-bottom: 8px;">
                    <b style="color: #60a5fa;">Akurasi Tinggi (R²):</b> Model final mencatat nilai <b>0.981</b> (naik dari 0.975). 
                    Artinya, model mampu menjelaskan <b>98,0%</b> variasi pada skor IKP. Peningkatan ini menegaskan presisi yang sangat tinggi dalam menangkap pola data ketahanan pangan.
                </li>
                <li style="margin-bottom: 8px;">
                    <b style="color: #34d399;">Penurunan Error (MAE & MAPE):</b> Nilai MAE turun dari 1.724 menjadi <b>1.213</b>, mengindikasikan penyimpangan prediksi terhadap data aktual semakin kecil. 
                    Selain itu, MAPE membaik dari 0.029 menjadi <b>0.020</b>, menandakan tingkat kesalahan relatif sangat rendah dan model lebih stabil.
                </li>
                <li>
                    <b style="color: #fbbf24;">Parameter Terbaik:</b> Diperoleh konfigurasi optimal: <code>n_estimators=404</code>, <code>learning_rate=0.118</code>, 
                    <code>max_depth=2</code>, <code>min_samples_split=7</code>, <code>min_samples_leaf=2</code>, dan <code>subsample=0.807</code>.
                </li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        # -----------------------------------------------------
        
        st.markdown("---")
        
        # --- SUB B: LEARNING CURVE ---
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

            train_mape = -np.mean(train_scores, axis=1) * 100
            test_mape = -np.mean(test_scores, axis=1) * 100
            train_std = np.std(train_scores, axis=1) * 100
            test_std = np.std(test_scores, axis=1)

            fig, ax = plt.subplots(figsize=(10, 6))
            
            fig.patch.set_facecolor('#0e1117')
            ax.set_facecolor('#0e1117')
            ax.grid(True, linestyle='--', alpha=0.1, color='white')
            ax.spines['bottom'].set_color('#334155')
            ax.spines['left'].set_color('#334155')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.tick_params(colors='#94a3b8')
            
            ax.plot(train_sizes, train_mape, 'o-', color="#3b82f6", label="MAPE Pelatihan")
            ax.plot(train_sizes, test_mape, 'o-', color="#ef4444", label="MAPE Validasi")
            ax.fill_between(train_sizes, train_mape - train_std, train_mape + train_std, alpha=0.1, color="#3b82f6")
            ax.fill_between(train_sizes, test_mape - test_std, test_mape + test_std, alpha=0.1, color="#ef4444")
            
            ax.text(train_sizes[-1], test_mape[-1], f"{test_mape[-1]:.2f}%", fontsize=10, color='#ef4444', ha="left", va="bottom", weight='bold')
            ax.text(train_sizes[-1], train_mape[-1], f"{train_mape[-1]:.2f}%", fontsize=10, color='#3b82f6', ha="left", va="top", weight='bold')
            
            ax.set_title("Kurva Pembelajaran (MAPE %)", fontsize=14, color='#f8fafc', pad=15)
            ax.set_xlabel("Jumlah Data Latih", fontsize=11, color='#94a3b8')
            ax.set_ylabel("MAPE (%)", fontsize=11, color='#94a3b8')
            ax.legend(loc="best", frameon=False, labelcolor='#e2e8f0')
            
            st.pyplot(fig)
            plt.close(fig)

        st.markdown("""
        <div style="background-color: #1f2937; padding: 20px; border-radius: 12px; border: 1px solid #374151; border-left: 5px solid #ef4444; margin-top: 15px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);">
            <h4 style="color: #fca5a5; margin-top: 0; margin-bottom: 12px;">💡 Insight: Learning Curve</h4>
            <p style="color: #cbd5e1; font-size: 15px; line-height: 1.6; text-align: justify; margin-bottom: 15px;">
                Berdasarkan gambar di atas, terlihat pola konvergensi yang positif antara performa pelatihan dan validasi:
            </p>
            <ul style="color: #cbd5e1; font-size: 15px; line-height: 1.6; padding-left: 20px;">
                <li style="margin-bottom: 8px;">
                    <b style="color: #60a5fa;">Garis Biru (Training):</b> Berada sangat rendah di angka <b>0.44%</b>. 
                    Ini mengindikasikan model mampu mempelajari pola data latih dengan sangat baik.
                </li>
                <li style="margin-bottom: 8px;">
                    <b style="color: #f87171;">Garis Merah (Validation):</b> Awalnya memiliki error tinggi saat data sedikit, namun secara konsisten menurun tajam seiring bertambahnya data, 
                    hingga mencapai titik stabil di angka <b>2.19%</b>.
                </li>
                <li>
                    <b style="color: #34d399;">Analisis Celah (Gap):</b> Celah antara garis pelatihan dan validasi semakin menyempit seiring bertambahnya data. 
                    Penurunan error yang konsisten menunjukkan bahwa model memiliki <b>generalisasi yang baik</b> dan tidak mengalami <i>overfitting</i> yang parah, berkat hyperparameter tuning dan rasio pembagian data yang tepat.
                </li>
            </ul>
            <p style="color: #cbd5e1; font-size: 15px; margin-top: 15px; border-top: 1px solid #374151; padding-top: 10px;">
                <b>Kesimpulan:</b> Kurva pembelajaran ini mengonfirmasi bahwa model GBR yang dibangun sudah <b>robust (kokoh)</b> dan siap digunakan untuk analisis interpretasi faktor dominan pada tahap selanjutnya.
            </p>
        </div>
        """, unsafe_allow_html=True)
        # -----------------------------------------------------

# --- 4. SHAP IMPORTANCE ---
    elif menu == "4. Feature Importance (SHAP)":
        st.header("⭐ 4. Feature Importance (SHAP)")
        st.info("""
        **📝 Interpretasi Fitur:**
        Menggunakan **SHAP Value** untuk melihat kontribusi setiap fitur terhadap prediksi IKP.
        """)
        
        with st.spinner("Menghitung SHAP Values (Menggunakan X_train sebagai background)..."):
            # Hitung SHAP
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
            
            # Layout Grafik & Tabel
            col_g, col_t = st.columns([2, 1])
            with col_g:
                fig = px.bar(
                    df_imp.head(10).sort_values(by="Mean(|SHAP|)", ascending=True), 
                    x="Mean(|SHAP|)", y="Feature", orientation='h', 
                    title="Top 10 Feature Importance (SHAP)",
                    color="Mean(|SHAP|)", color_continuous_scale="Tealgrn", template="plotly_dark"
                )
                fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True)
            
            with col_t:
                st.markdown("##### 📋 Detail Nilai")
                st.dataframe(df_imp.style.format({"Mean(|SHAP|)": "{:.4f}"}).background_gradient(cmap="Greens"), use_container_width=True, height=400)
                
            if 'top_3_features' not in st.session_state:
                st.session_state['top_3_features'] = df_imp['Feature'].head(3).tolist()


            st.markdown("""
<div style="background-color: #1f2937; padding: 20px; border-radius: 12px; border: 1px solid #374151; border-left: 5px solid #3b82f6; margin-top: 20px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);">
    <h4 style="color: #60a5fa; margin-top: 0; margin-bottom: 12px;">💡 Insight: Faktor Dominan</h4>
    <p style="color: #cbd5e1; font-size: 15px; line-height: 1.6; text-align: justify; margin-bottom: 15px;">
        Berdasarkan visualisasi SHAP Value di atas, teridentifikasi urutan prioritas faktor-faktor yang mempengaruhi IKP tahun 2024 sebagai berikut:
    </p>
    <ul style="color: #cbd5e1; font-size: 15px; line-height: 1.6; padding-left: 20px;">
        <li style="margin-bottom: 10px;">
            <b style="color: #34d399;">1. Rasio Konsumsi Normatif (NPCR) - Score: 7.4752</b><br>
            Konsisten menempati peringkat pertama sebagai faktor paling dominan. Nilainya mencolok lebih dari dua kali lipat dibanding faktor kedua. 
            Ini mengidentifikasi bahwa aspek <b>ketersediaan pangan</b> (keseimbangan produksi vs konsumsi) menjadi determinan utama status ketahanan pangan wilayah.
        </li>
        <li style="margin-bottom: 10px;">
            <b style="color: #34d399;">2. Tanpa Akses Air Bersih - Score: 3.1510</b><br>
            Menempati posisi kedua, temuan ini menyoroti krusialnya peran akses air bersih (aspek <b>pemanfaatan</b>) dalam menunjang ketahanan pangan yang layak.
        </li>
        <li style="margin-bottom: 10px;">
            <b style="color: #34d399;">3. Persentase Penduduk Miskin - Score: 2.3290</b><br>
            Berada di urutan ketiga, merefleksikan pentingnya aspek <b>akses pangan</b>. Tingkat kemiskinan secara langsung membatasi kemampuan rumah tangga menjangkau pangan di pasar.
        </li>
    </ul>
    <hr style="border-top: 1px solid #374151; margin: 15px 0;">
    <p style="color: #94a3b8; font-size: 14px; line-height: 1.6;">
        <b>Faktor Lainnya (Kontribusi Moderat/Kecil):</b><br>
        Variabel lain memiliki pengaruh relatif lebih rendah (SHAP < 1.5). Angka Harapan Hidup (1.42) dan Stunting (0.82) berkontribusi moderat, 
        sedangkan Pengeluaran Pangan, Lama Sekolah Perempuan, Rasio Nakes, dan Listrik memiliki dampak lebih kecil terhadap variasi model global.
    </p>
    <p style="color: #cbd5e1; font-size: 15px; margin-top: 10px; font-weight: 500;">
        <b>Kesimpulan:</b> NPCR, Akses Air Bersih, dan Kemiskinan adalah tiga faktor dominan yang menjadi penggerak utama dinamika IKP 2024, mewakili kombinasi vital dari pilar ketersediaan, akses, dan pemanfaatan pangan.
    </p>
</div>
""", unsafe_allow_html=True)
            # -----------------------------------------------------

# --- 5. PDP ---
    elif menu == "5. Partial Dependence (PDP)":
        st.header("📈 5. Partial Dependence Plot (PDP)")
        st.info("""
        **📝 Analisis Kausalitas:**
        Melihat arah pengaruh (Positif/Negatif) fitur terhadap IKP dalam **Skala Asli**.
        """)
        
        # --- 1.PILIHAN FITUR ---
        valid_features = [c for c in X_train_final.columns if c != 'Intercept']
        
        if 'top_3_features' in st.session_state:
            # Ambil Top 3 dari SHAP
            top_features = [f for f in st.session_state['top_3_features'] if f in valid_features]
            # Fallback jika kosong
            if not top_features: top_features = valid_features[:3]
        else:
            # Default jika belum ke menu SHAP 
            top_features = valid_features[:3] 
            
        col_sel, col_btn = st.columns([3, 1])
        with col_sel:
            feature_to_plot = st.selectbox("Pilih Fitur Dominan (Top 3):", top_features)
        with col_btn:
            st.write(""); st.write("")
            btn_gen = st.button("Generate Grafik ✨")
        
        if btn_gen:
            with st.spinner(f"Menganalisis hubungan {feature_to_plot} terhadap IKP..."):
                # Hitung PDP
                pd_results = partial_dependence(
                    model_tuned, X_train_final, [feature_to_plot], 
                    grid_resolution=20, kind="average"
                )
                
                pdp_values = pd_results["average"][0]
                x_scaled = pd_results["grid_values"][0]
                
                # Inverse Transform Logic
                cols_used_in_scaler = [c for c in X_train_final.columns if c != 'Intercept']
                
                if feature_to_plot in cols_used_in_scaler:
                    idx_in_scaler = cols_used_in_scaler.index(feature_to_plot)
                    dummy = np.zeros((len(x_scaled), len(cols_used_in_scaler)))
                    dummy[:, idx_in_scaler] = x_scaled
                    x_unscaled = scaler.inverse_transform(dummy)[:, idx_in_scaler]
                else:
                    x_unscaled = x_scaled
                
                # --- 2. PLOTTING
                fig, ax = plt.subplots(figsize=(9, 5))
                fig.patch.set_facecolor('#0e1117')
                ax.set_facecolor('#0e1117')
                
                # Garis Utama
                ax.plot(x_unscaled, pdp_values, color='#10b981', linewidth=3, marker='o', markersize=4, label="Prediksi IKP")
                ax.fill_between(x_unscaled, min(pdp_values), pdp_values, color='#10b981', alpha=0.1)
                
                # Anotasi: Awal, Tengah, Akhir
                idxs = [0, len(x_unscaled)//2, -1] 
                
                for i in idxs:
                    val_x = x_unscaled[i]
                    val_y = pdp_values[i]
                    
                    # Titik Penanda
                    ax.scatter(val_x, val_y, color='#f8fafc', s=60, zorder=5, edgecolors='#10b981', linewidth=2)
                    
                    # Kotak Anotasi 
                    ax.annotate(
                        f"{val_x:.1f}\n{val_y:.2f}", 
                        xy=(val_x, val_y), 
                        xytext=(0, 25), # Geser ke atas
                        textcoords="offset points", 
                        ha='center', va='bottom',
                        fontsize=10, fontweight='bold', color='#10b981',
                        bbox=dict(boxstyle="round,pad=0.4", fc="#0e1117", alpha=0.8, edgecolor='#10b981')
                    )

                ax.set_xlabel(f"{feature_to_plot} (Skala Asli)", fontsize=11, color='#94a3b8')
                ax.set_ylabel("Perubahan Prediksi IKP", fontsize=11, color='#94a3b8')
                ax.set_title(f"Pengaruh {feature_to_plot} terhadap IKP", fontsize=14, fontweight='bold', color='#f8fafc', pad=25)
                
                ax.grid(True, linestyle='--', alpha=0.1, color='white')
                ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
                ax.spines['left'].set_color('#334155'); ax.spines['bottom'].set_color('#334155')
                ax.tick_params(colors='#94a3b8')
                
                st.pyplot(fig)
                plt.close(fig)

                # --- 3. DESKRIPSI DINAMIS ---
                if "NCPR" in feature_to_plot or "Konsumsi" in feature_to_plot:
                    insight_title = "💡 Insight: Rasio Konsumsi Normatif (NPCR)"
                    
                    insight_text = """
<p style="color: #cbd5e1; font-size: 15px; line-height: 1.6; text-align: justify;">
    Grafik di atas memperlihatkan <b>pola hubungan negatif non-linear yang sangat tajam/ekstrem</b>.
</p>
<ul style="color: #cbd5e1; font-size: 15px; line-height: 1.6; padding-left: 20px;">
    <li>Pada rentang awal (0 - 0.5), grafik menunjukkan lonjakan ke titik puncak dampak positif tertinggi.</li>
    <li>Namun, setelahnya garis menurun drastis tanpa henti hingga angka 5.0. Penurunan ini sangat tajam, bergerak dari dampak positif <b>(+5)</b> hingga dampak negatif terendah <b>(-24.3)</b>.</li>
</ul>
<p style="color: #cbd5e1; font-size: 15px; line-height: 1.6; text-align: justify;">
    <b>Kesimpulan:</b> Mengingat semakin tinggi NPCR berarti semakin besar defisit pangan (konsumsi > produksi), grafik ini mengonfirmasi bahwa <b>ketersediaan pangan mandiri adalah fondasi utama</b>. 
    Wilayah yang gagal memenuhi kebutuhan pangannya sendiri akan mengalami penurunan skor IKP paling parah dibandingkan faktor manapun.
</p>
"""
                
                elif "Air Bersih" in feature_to_plot:
                    insight_title = "💡 Insight: Akses Air Bersih"
                    insight_text = """
<p style="color: #cbd5e1; font-size: 15px; line-height: 1.6; text-align: justify;">
    Grafik ini menunjukkan <b>hubungan negatif linier</b>. Terlihat tren penurunan yang konsisten; semakin tinggi persentase rumah tangga tanpa air bersih (bergerak ke kanan), semakin rendah skor IKP.
</p>
<ul style="color: #cbd5e1; font-size: 15px; line-height: 1.6; padding-left: 20px;">
    <li>Rentang dampak bergerak dari posisi positif <b>3.69</b> (saat persentase mendekati 0%) hingga turun mencapai dampak negatif <b>-6.40</b> (saat persentase tinggi di 65.3%).</li>
</ul>
<p style="color: #cbd5e1; font-size: 15px; line-height: 1.6; text-align: justify;">
    <b>Kesimpulan:</b> Infrastruktur akses air bersih memiliki korelasi langsung dengan pemanfaatan pangan. Ketiadaan air bersih menghambat penyerapan nutrisi dan menurunkan kualitas kesehatan, yang secara langsung menggerus skor IKP.
</p>
"""
                
                elif "Miskin" in feature_to_plot or "Kemiskinan" in feature_to_plot:
                    insight_title = "💡 Insight: Persentase Penduduk Miskin"
                    insight_text = """
<p style="color: #cbd5e1; font-size: 15px; line-height: 1.6; text-align: justify;">
    Grafik memperlihatkan <b>hubungan negatif non-linear</b>. Dampak positif tertinggi (2.93) terjadi pada tingkat kemiskinan terendah (4.1%).
</p>
<ul style="color: #cbd5e1; font-size: 15px; line-height: 1.6; padding-left: 20px;">
    <li><b>Fase Shock:</b> Penurunan tajam terjadi pada rentang awal hingga tingkat kemiskinan mencapai 16.6% (dampak turun ke -2.09).</li>
    <li><b>Fase Melandai:</b> Setelah melewati titik tersebut, penurunan skor IKP mulai melandai hingga mencapai dampak negatif -6.79 pada tingkat kemiskinan 27.9%.</li>
</ul>
<p style="color: #cbd5e1; font-size: 15px; line-height: 1.6; text-align: justify;">
    <b>Kesimpulan:</b> Grafik ini mengisyaratkan bahwa peningkatan kemiskinan pada fase awal memiliki dampak guncangan yang lebih besar terhadap penurunan ketahanan pangan dibandingkan peningkatan pada fase lanjut.
</p>
"""
                else:
                    insight_title = "💡 Interpretasi Grafik"
                    insight_text = f"""
<p style="color: #cbd5e1; font-size: 15px; line-height: 1.6; text-align: justify;">
    Grafik Partial Dependence Plot (PDP) di atas menunjukkan hubungan marjinal antara variabel <b>{feature_to_plot}</b> dengan prediksi skor IKP.
    Sumbu X merepresentasikan nilai fitur dalam skala aslinya, sedangkan sumbu Y menunjukkan perubahan prediksi skor IKP. 
    Perhatikan tren garis untuk melihat apakah hubungan bersifat positif, negatif, atau non-linear.
</p>
"""

                # Render Insight Box
                st.markdown(f"""
<div style="background-color: #1f2937; padding: 20px; border-radius: 12px; border: 1px solid #374151; border-left: 5px solid #10b981; margin-top: 20px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);">
    <h4 style="color: #34d399; margin-top: 0; margin-bottom: 12px;">{insight_title}</h4>
    {insight_text}
<hr style="border-top: 1px solid #374151; margin: 15px 0;">
<p style="color: #94a3b8; font-size: 13px; margin-top: 5px;">
    <i>Catatan: Sumbu X = Nilai Fitur (Skala Asli), Sumbu Y = Perubahan Marjinal Prediksi IKP.</i>
</p>
</div>
""", unsafe_allow_html=True)
                
                # Rekomendasi Strategis (Static)
                st.markdown("""
<div style="margin-top: 15px; padding: 15px; background-color: rgba(59, 130, 246, 0.1); border-radius: 8px; border: 1px solid #1e3a8a;">
    <p style="color: #bfdbfe; font-size: 14px; margin: 0; text-align: center;">
        <b>Rekomendasi Strategis:</b> Peningkatan IKP harus difokuskan pada tiga intervensi utama: 
        Swasembada pangan (menekan NPCR), memperluas infrastruktur air bersih, dan mitigasi kemiskinan awal.
    </p>
</div>
""", unsafe_allow_html=True)
                
else:
    st.warning("Data belum dimuat.")