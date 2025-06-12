import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
from sklearn.preprocessing import MinMaxScaler

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
        st.write("### 🔍 Pratinjau Data")
        st.dataframe(df, use_container_width=True)

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

                if selected_criteria:
                    st.subheader("⚖️ Atur Bobot Tiap Kriteria (%)")
                    weights_raw = []
                    for crit in selected_criteria:
                        w = st.number_input(f"Bobot untuk {crit}", min_value=0, max_value=100, value=int(100/len(selected_criteria)))
                        weights_raw.append(w)
                    weight_total = sum(weights_raw)
                    if weight_total == 0:
                        weights = np.ones(len(weights_raw)) / len(weights_raw)
                    else:
                        weights = np.array(weights_raw) / weight_total

            st.subheader("📈 Visualisasi Total berdasarkan Kategori")
            selected_num = selected_criteria[0] if selected_criteria else numeric_cols[0]
            chart = alt.Chart(df).mark_bar().encode(
                x=selected_cat,
                y=f'sum({selected_num})',
                color=selected_cat,
                tooltip=[selected_cat, selected_num]
            ).properties(width=700, height=400)
            st.altair_chart(chart, use_container_width=True)

            scaler = MinMaxScaler()
            norm_df = pd.DataFrame(scaler.fit_transform(df[selected_criteria]), columns=selected_criteria)

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

            st.subheader("📊 Hasil Skor dan Ranking")
            label_col = selected_cat if selected_cat in df.columns else df.index.astype(str)
            display_df = df[[*selected_criteria, 'SAW_Score', 'WP_Score', 'TOPSIS_Score']].copy()
            display_df[label_col] = df[label_col]
            st.dataframe(display_df.sort_values(by='SAW_Score', ascending=False), use_container_width=True)

            st.subheader("📉 Visualisasi Skor Keputusan")
            for method in ['SAW_Score', 'WP_Score', 'TOPSIS_Score']:
                st.markdown(f"**Metode {method.replace('_Score','')}**")
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
