import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

import gspread
from google.oauth2.service_account import Credentials

# =============== CONFIG & STYLE ===============
st.set_page_config("Forecast Suku Cadang Bengkel", layout="wide")
plt.rcParams["figure.figsize"] = (9, 4)
plt.rcParams["figure.dpi"] = 120


# =============== DATA LOADER (GOOGLE SHEETS) ===============
@st.cache_data(ttl=60)  # refresh data paling lama tiap 60 detik
def load_raw_data_from_sheet():
    """
    Baca data transaksi dari Google Sheets dan ubah ke dataframe standar:
    kolom: ds (tanggal), nama_barang, y (jumlah)
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

    data = ws.get_all_records()
    df = pd.DataFrame(data)

    # samakan nama kolom dengan yang dipakai di Colab
    df = df.rename(columns={
        "Tanggal": "ds",
        "Nama Barang": "nama_barang",
        "Jumlah": "y"
    })
    df["ds"] = pd.to_datetime(df["ds"], errors="coerce", dayfirst=True)
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df = df.dropna(subset=["ds", "nama_barang", "y"])


    return df


def get_worksheet():
    """
    Buka worksheet untuk append transaksi baru.
    Jangan di-cache karena ini koneksi live.
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


def build_full_timeseries(df):
    """
    Dari data mentah (ds, nama_barang, y) buat timeseries harian lengkap
    dengan nama_clean sebagai key.
    """
    df = df.copy()
    df["nama_clean"] = (
        df["nama_barang"]
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "", regex=True)
    )

    # agregasi per hari per nama_clean
    df_daily = (
        df.groupby(["nama_clean", "ds"], as_index=False)["y"]
          .sum()
    )

    def _expand(group):
        full_date = pd.date_range(group["ds"].min(),
                                  group["ds"].max(),
                                  freq="D")
        g = group.set_index("ds").reindex(full_date)
        g.index.name = "ds"
        g["nama_clean"] = g["nama_clean"].ffill().bfill()
        g["y"] = g["y"].fillna(0)
        return g.reset_index()

    df_full = (
        df_daily.groupby("nama_clean", group_keys=False)
                .apply(_expand)
    )

    # mapping nama_clean -> nama asli (dari data terbaru)
    mapping_nama_live = (
        df[["nama_clean", "nama_barang"]]
        .drop_duplicates()
        .set_index("nama_clean")["nama_barang"]
        .to_dict()
    )

    return df_full, mapping_nama_live


# =============== METRICS DARI COLAB ===============
@st.cache_data
def load_metrics_from_colab():
    """
    Baca hasil evaluasi (MAPE, RMSE) yang sudah dihitung di Colab.
    File ini TIDAK diubah di app, hanya dibaca saja.
    """
    hasil = pd.read_csv("hasil_metrics.csv")

    # diasumsikan kolom: ["nama_clean", "Nama Barang", "MAPE_Prophet", "RMSE_Prophet"]
    return hasil


# =============== MODEL UTILS (TRAINING LIVE UNTUK FORECAST) ===============
def train_full_and_forecast(df_brand, periods=30):
    """
    Melatih Prophet dari data terbaru (df_brand) untuk SATU barang,
    lalu membuat forecast ke depan 'periods' hari,
    DIMULAI dari H+1 setelah tanggal terakhir data.
    """
    # 1. Siapkan data per hari
    data = (
        df_brand[["ds", "y"]]
        .groupby("ds", as_index=False)["y"]
        .sum()
        .sort_values("ds")
        .reset_index(drop=True)
    )

    if len(data) < 5:
        return None, data, pd.DataFrame()

    # 2. Log transform
    data["y_log"] = np.log1p(data["y"])
    df_prophet = data[["ds", "y_log"]].rename(columns={"y_log": "y"})

    # 3. Train Prophet
    m = Prophet(
        seasonality_mode="multiplicative",
        yearly_seasonality=False,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.8,
        seasonality_prior_scale=10
    )
    m.add_country_holidays(country_name="ID")
    m.fit(df_prophet)

    # 4. Tentukan tanggal terakhir di data
    last_date = data["ds"].max()

    # (opsional debug di Streamlit)
    # st.write("Last date untuk barang ini:", last_date)

    # 5. Buat future dates MANUAL: mulai H+1 dari last_date
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=periods,
        freq="D"
    )
    future = pd.DataFrame({"ds": future_dates})

    # 6. Prediksi hanya untuk future (tanpa history)
    fc = m.predict(future)
    fc["yhat_real"] = np.expm1(fc["yhat"])

    # 7. fc_future langsung = fc (karena isinya cuma tanggal ke depan)
    fc_future = fc[["ds", "yhat_real"]].copy()

    return m, data, fc_future


# =============== MAIN APP ===============
def main():
    st.title("📈 Peramalan Permintaan Suku Cadang Bengkel (Prophet) – Hybrid")

    # ---- Load data terbaru dari Google Sheets ----
    df_raw = load_raw_data_from_sheet()
    if df_raw.empty:
        st.error("Data dari Google Sheets kosong atau tidak terbaca. Cek isi sheet dan nama kolom.")
        st.stop()

    df_full, mapping_nama_live = build_full_timeseries(df_raw)

    # ---- Load metrics dari Colab ----
    hasil_metrics = load_metrics_from_colab()

    # ---- Sidebar: pilih barang & horizon ----
    # List barang ambil dari metrics Colab supaya MAPE ada
    list_barang = (
        hasil_metrics.sort_values("Nama Barang")["Nama Barang"].tolist()
    )
    barang_selected = st.sidebar.selectbox("Pilih Barang", list_barang)

    # Ambil nama_clean dari metrics
    met_row = hasil_metrics[hasil_metrics["Nama Barang"] == barang_selected]
    if met_row.empty:
        st.error("Barang tidak ditemukan di hasil_metrics.csv. Cek kembali file metrics.")
        st.stop()

    nama_clean_selected = met_row["nama_clean"].iloc[0]

    horizon = st.sidebar.slider("Horizon Forecast (hari)", 7, 60, 30)

    # ---- Info MAPE & RMSE (FIX dari Colab) ----
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

    st.caption("📌 Nilai MAPE & RMSE di atas adalah hasil evaluasi model di Colab (snapshot), tidak dihitung ulang di aplikasi ini.")
    st.markdown("---")

    # ---- Filter data untuk barang terpilih & training model live ----
    df_b = df_full[df_full["nama_clean"] == nama_clean_selected].copy()
    if df_b.empty:
        st.warning("Tidak ada data historis untuk barang ini di Google Sheets.")
        st.stop()

    m_full, data_hist, fc_future = train_full_and_forecast(df_b, periods=horizon)
    if m_full is None or fc_future.empty:
        st.warning("Data terlalu sedikit untuk dibuat forecast.")
        st.stop()

    # ---- Plot history + forecast ----
    st.subheader(f"Grafik Forecast {barang_selected} ({horizon} hari ke depan)")

    fig, ax = plt.subplots()
    ax.plot(data_hist["ds"], data_hist["y"], label="Aktual (historis)")
    ax.plot(fc_future["ds"], fc_future["yhat_real"], "--", label="Forecast")
    ax.set_xlabel("Tanggal")
    ax.set_ylabel("Jumlah")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

    # ---- Komponen Prophet (trend & musiman) ----
    with st.expander("Lihat Komponen Prophet (trend & musiman)"):
        fig_comp = m_full.plot_components(
            m_full.predict(m_full.make_future_dataframe(periods=horizon))
        )
        st.pyplot(fig_comp)

    # ---- Tabel forecast harian ----
    st.subheader("Tabel Forecast Harian")
    fc_table = fc_future[["ds", "yhat_real"]].copy()
    fc_table["Nama Barang"] = barang_selected
    fc_table = fc_table[["Nama Barang", "ds", "yhat_real"]]
    fc_table = fc_table.rename(columns={
        "ds": "Tanggal",
        "yhat_real": "Prediksi Jumlah"
    })
    st.dataframe(fc_table, use_container_width=True)

    # ---- Ringkasan forecast ----
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

    # =============== INPUT TRANSAKSI BARU ===============
    st.subheader("➕ Input Transaksi Baru")

    st.caption(
        "Transaksi baru akan disimpan ke Google Sheets dan akan ikut dipakai saat model dilatih ulang "
        "setiap kali aplikasi di-refresh. Nilai MAPE/RMSE tetap mengacu pada hasil evaluasi di Colab."
    )

    with st.form("form_transaksi"):
        tgl_baru = st.date_input("Tanggal Transaksi")
        # Nama barang pakai list_barang dari metrics, biar konsisten
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
        st.caption("Refresh / buka ulang aplikasi untuk memasukkan data baru ini ke dalam training forecast.")


if __name__ == "__main__":
    main()


