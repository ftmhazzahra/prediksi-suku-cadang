import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

from prophet import Prophet  # tetap dibutuhkan untuk load model pickle
import gspread
from google.oauth2.service_account import Credentials

# ================== CONFIG / STYLE ==================
st.set_page_config("Forecast Suku Cadang Bengkel", layout="wide")
plt.rcParams["figure.figsize"] = (9, 4)
plt.rcParams["figure.dpi"] = 120


# ================== LOAD MODELS & METRICS ==================
@st.cache_resource
def load_models():
    """
    Load semua model Prophet yang sudah di-train di Colab
    dari file models_prophet.pkl

    Struktur yang diasumsikan:
    models_dict = {
        nama_clean: {
            "model": Prophet object,
            "history": DataFrame dengan kolom [ds, y],
            "last_date": timestamp terakhir di history
        },
        ...
    }
    """
    with open("models_prophet.pkl", "rb") as f:
        models_dict = pickle.load(f)
    return models_dict


@st.cache_data
def load_metrics_and_mapping():
    """
    Load hasil evaluasi (MAPE, RMSE) dari Colab
    dan mapping nama barang.
    """
    hasil = pd.read_csv("hasil_metrics.csv")

    # pastikan kolom dsb sesuai dengan file hasil_metrics.csv
    # diasumsikan ada kolom: ["nama_clean", "Nama Barang", "MAPE_Prophet", "RMSE_Prophet"]
    mapping_nama = (
        hasil[["nama_clean", "Nama Barang"]]
        .drop_duplicates()
        .set_index("nama_clean")["Nama Barang"]
        .to_dict()
    )
    return hasil, mapping_nama


def get_worksheet():
    """
    Membuka worksheet Google Sheets untuk input transaksi baru.
    Service account dibaca dari st.secrets["gcp_service_account"]
    dan info sheet dari st.secrets["sheets"].
    """
    scope = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]

    creds = Credentials.from_service_account_info(
        st.secrets["gcp_service_account"],
        scopes=scope
    )
    client = gspread.authorize(creds)

    sh = client.open_by_key(st.secrets["sheets"]["sheet_id"])
    ws = sh.worksheet(st.secrets["sheets"]["data"])
    return ws


# ================== MAIN APP ==================
def main():
    st.title("📈 Peramalan Permintaan Suku Cadang Bengkel (Prophet)")

    # --- Load model & metrics (sekali saja, via cache) ---
    models_dict = load_models()
    hasil_metrics, mapping_nama = load_metrics_and_mapping()

    # --- Sidebar: pilih barang & horizon ---
    list_barang = (
        hasil_metrics.sort_values("Nama Barang")["Nama Barang"].tolist()
    )
    barang_selected = st.sidebar.selectbox("Pilih Barang", list_barang)

    # cari nama_clean dari mapping
    nama_clean_selected = None
    for nama_clean, nama_asli in mapping_nama.items():
        if nama_asli == barang_selected:
            nama_clean_selected = nama_clean
            break

    if nama_clean_selected is None:
        st.error("Nama barang tidak ditemukan di mapping. Cek kembali hasil_metrics.csv.")
        st.stop()

    horizon = st.sidebar.slider("Horizon Forecast (hari)", 7, 60, 30)

    # --- Info MAPE & RMSE (PAKAI HASIL COLAB, TIDAK DIHITUNG ULANG) ---
    met_row = hasil_metrics[hasil_metrics["Nama Barang"] == barang_selected]

    col1, col2 = st.columns(2)
    with col1:
        if not met_row["MAPE_Prophet"].isna().iloc[0]:
            st.metric(
                "MAPE (test) – Prophet (dari Colab)",
                f"{met_row['MAPE_Prophet'].iloc[0]:.2f}%"
            )
        else:
            st.metric("MAPE (test) – Prophet (dari Colab)", "-")

    with col2:
        if not met_row["RMSE_Prophet"].isna().iloc[0]:
            st.metric(
                "RMSE (test) – Prophet (dari Colab)",
                f"{met_row['RMSE_Prophet'].iloc[0]:.2f}"
            )
        else:
            st.metric("RMSE (test) – Prophet (dari Colab)", "-")

    st.caption("📌 Nilai MAPE & RMSE di atas adalah hasil evaluasi model di Colab, tidak dihitung ulang di aplikasi ini.")
    st.markdown("---")

    # --- Ambil model pre-trained untuk barang terpilih ---
    if nama_clean_selected not in models_dict:
        st.error(f"Model untuk '{nama_clean_selected}' tidak ditemukan di models_prophet.pkl.")
        st.stop()

    model_info = models_dict[nama_clean_selected]
    m_full: Prophet = model_info["model"]
    data_hist: pd.DataFrame = model_info["history"]
    last_date = model_info["last_date"]

    # --- Forecast ke depan menggunakan model pre-trained ---
    future = m_full.make_future_dataframe(periods=horizon)
    fc = m_full.predict(future)
    fc["yhat_real"] = np.expm1(fc["yhat"])

    fc_future = fc[fc["ds"] > pd.to_datetime(last_date)].copy()

    # --- Plot history + forecast ---
    st.subheader(f"Grafik Forecast {barang_selected} ({horizon} hari ke depan)")

    fig, ax = plt.subplots()
    ax.plot(data_hist["ds"], data_hist["y"], label="Aktual (historis)")
    ax.plot(fc_future["ds"], fc_future["yhat_real"], "--", label="Forecast")
    ax.set_xlabel("Tanggal")
    ax.set_ylabel("Jumlah")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

    # --- Komponen Prophet (trend & musiman) ---
    with st.expander("Lihat Komponen Prophet (trend & musiman)"):
        fig_comp = m_full.plot_components(
            m_full.predict(m_full.make_future_dataframe(periods=horizon))
        )
        st.pyplot(fig_comp)

    # --- Tabel forecast harian ---
    st.subheader("Tabel Forecast Harian")
    fc_table = fc_future[["ds", "yhat_real"]].copy()
    fc_table["Nama Barang"] = barang_selected
    fc_table = fc_table[["Nama Barang", "ds", "yhat_real"]]
    fc_table = fc_table.rename(columns={
        "ds": "Tanggal",
        "yhat_real": "Prediksi Jumlah"
    })
    st.dataframe(fc_table, use_container_width=True)

    # --- Ringkasan forecast ---
    total_horizon = fc_table["Prediksi Jumlah"].sum()
    avg_per_day = fc_table["Prediksi Jumlah"].mean()
    max_row = fc_table.loc[fc_table["Prediksi Jumlah"].idxmax()]

    st.info(
        f"Perkiraan total kebutuhan {barang_selected} selama {horizon} hari ke depan: "
        f"**{total_horizon:.0f} unit**.\n\n"
        f"Rata-rata per hari: **{avg_per_day:.0f} unit**.\n"
        f"Hari tersibuk diprediksi pada **{max_row['Tanggal'].date()}** "
        f"dengan **{max_row['Prediksi Jumlah']:.0f} unit**."
    )

    st.markdown("---")

    # ================== INPUT TRANSAKSI BARU ==================
    st.subheader("➕ Input Transaksi Baru")

    st.caption(
        "Transaksi baru akan disimpan ke Google Sheets. "
        "Model dan nilai MAPE/RMSE **tidak langsung berubah**. "
        "Untuk memperbarui model, lakukan retraining di Colab dan upload ulang file models_prohpet.pkl & hasil_metrics.csv."
    )

    with st.form("form_transaksi"):
        tgl_baru = st.date_input("Tanggal Transaksi")
        barang_baru = st.selectbox("Nama Barang", list_barang, key="barang_form")
        jumlah_baru = st.number_input("Jumlah", min_value=1, step=1, value=1)
        submitted = st.form_submit_button("Simpan Transaksi")

    if submitted:
        ws = get_worksheet()
        new_row = [
            str(tgl_baru),
            barang_baru,
            int(jumlah_baru)
        ]
        ws.append_row(new_row)
        st.success("Transaksi baru berhasil disimpan ke Google Sheets.")
        st.caption("Untuk memasukkan data ini ke dalam model, lakukan retraining di Colab dan update file model di aplikasi.")


if __name__ == "__main__":
    main()
