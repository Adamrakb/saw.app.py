import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Fungsi pagination sederhana
def paginate_dataframe(data, page_size=10, key=None):
    total_pages = (len(data) - 1) // page_size + 1
    page = st.number_input("Halaman", min_value=1, max_value=total_pages, value=1, step=1, key=key)
    start = (page - 1) * page_size
    end = start + page_size
    return data.iloc[start:end]

# Pengaturan Halaman
st.set_page_config(page_title="Sistem Pendukung Keputusan", layout="wide")
st.title("📊 Sistem Pendukung Keputusan Adaptif dengan Analisis Sensitivitas")

# Upload CSV
uploaded_file = st.file_uploader("📁 Upload file CSV Anda", type=["csv"])
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        df.columns = df.columns.str.strip()  # hilangkan spasi di nama kolom
        st.success("File berhasil dibaca!")

        # Konversi kolom tanggal otomatis jika ada
        for col in df.columns:
            if 'date' in col.lower():
                try:
                    df[col] = pd.to_datetime(df[col])
                except:
                    pass

        numeric_cols = df.select_dtypes(include=['int', 'float']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

        if len(numeric_cols) < 2:
            st.warning("Minimal harus ada dua kolom numerik untuk SPK.")
        else:
            with st.sidebar:
                st.header("🔍 Filter dan Pengaturan")

                if categorical_cols:
                    selected_cat = st.selectbox("Pilih kolom kategori", categorical_cols)
                else:
                    selected_cat = df.columns[0]

                selected_criteria = st.multiselect(
                    "Pilih kriteria numerik untuk analisis",
                    numeric_cols,
                    default=numeric_cols[:2]
                )
                jenis_kriteria = {}
                for crit in selected_criteria:
                    jenis_kriteria[crit] = st.selectbox(f"Jenis kriteria untuk {crit}", ["Benefit", "Cost"])

                if selected_criteria:
                    st.subheader("⚖️ Atur Bobot Tiap Kriteria (%)")
                    weights_raw = []
                    for crit in selected_criteria:
                        w = st.number_input(f"Bobot untuk {crit}", min_value=0, max_value=100,
                                            value=int(100 / len(selected_criteria)))
                        weights_raw.append(w)
                    weight_total = sum(weights_raw)
                    if weight_total == 0:
                        weights = np.ones(len(weights_raw)) / len(weights_raw)
                    else:
                        weights = np.array(weights_raw) / weight_total

            # Normalisasi manual sesuai cost/benefit
            norm_df = pd.DataFrame()
            for crit in selected_criteria:
                if jenis_kriteria[crit] == "Benefit":
                    norm_df[crit] = df[crit] / df[crit].max()
                else:  # Cost
                    norm_df[crit] = df[crit].min() / df[crit]

            # SAW
            df['SAW_Score'] = norm_df.dot(weights)

            # WP
            wp_scores = np.prod(np.power(norm_df, weights), axis=1)
            df['WP_Score'] = wp_scores / np.max(wp_scores)

            # TOPSIS
            ideal_pos = norm_df.max()
            ideal_neg = norm_df.min()
            d_pos = np.sqrt(((norm_df - ideal_pos) ** 2 * weights).sum(axis=1))
            d_neg = np.sqrt(((norm_df - ideal_neg) ** 2 * weights).sum(axis=1))
            df['TOPSIS_Score'] = d_neg / (d_pos + d_neg)

            # Label kolom untuk identifikasi
            label_col = selected_cat if selected_cat in df.columns else df.index.astype(str)
            display_df = df[[*selected_criteria, 'SAW_Score', 'WP_Score', 'TOPSIS_Score']].copy()
            display_df[label_col] = df[label_col]
            sorted_df = display_df.sort_values(by='SAW_Score', ascending=False)

            # === TAB LAYOUT ===
            tab1, tab2, tab3 = st.tabs(["📄 Pratinjau Data", "📊 Hasil Skor", "📉 Grafik Skor"])

            with tab1:
                st.write("### 🔍 Pratinjau Data")
                paginated_df = paginate_dataframe(df, page_size=10, key="preview")
                st.dataframe(paginated_df, use_container_width=True)

            with tab2:
                st.write("### 📊 Tabel Skor & Ranking")
                paginated_result = paginate_dataframe(sorted_df, page_size=10, key="ranking")
                st.dataframe(paginated_result, use_container_width=True)

            with tab3:
                st.write("### 📉 Visualisasi Skor Keputusan")
                for method in ['SAW_Score', 'WP_Score', 'TOPSIS_Score']:
                    st.markdown(f"**Metode {method.replace('_Score', '')}**")
                    bar = alt.Chart(df).mark_bar().encode(
                        x=alt.X(f'{method}:Q', title='Skor'),
                        y=alt.Y(f'{label_col}:N', sort='-x'),
                        tooltip=[label_col, method]
                    ).properties(width=700, height=400)
                    st.altair_chart(bar, use_container_width=True)

    except Exception as e:
        st.error(f"Gagal membaca file: {e}")
else:
    st.info("Silakan unggah file CSV terlebih dahulu untuk mulai.")
