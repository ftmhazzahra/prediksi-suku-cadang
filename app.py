# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

st.set_page_config("Forecast Suku Cadang Bengkel", layout="wide")
plt.rcParams["figure.figsize"] = (9, 4)
plt.rcParams["figure.dpi"] = 120


# =============== CONFIG ===============
DATA_PATH = "dataset.xlsx"   # ganti dengan file kamu
SHEET_NAME = 0                      # atau "Sheet1"


# =============== DATA LOADER ===============
@st.cache_data
def load_processed_data():
    df_full = pd.read_csv("df_full_clean.csv")
    df_full["ds"] = pd.to_datetime(df_full["ds"])

    hasil = pd.read_csv("hasil_metrics.csv")

    # bikin mapping nama
    mapping_nama = (
        hasil[["nama_clean", "Nama Barang"]]
        .drop_duplicates()
        .set_index("nama_clean")["Nama Barang"]
        .to_dict()
    )
    return df_full, hasil, mapping_nama


def build_full_timeseries(df):
    # nama_clean untuk konsistensi
    df["nama_clean"] = (
        df["nama_barang"]
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "", regex=True)
    )

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

    # mapping nama asli
    mapping_nama = (
        df[["nama_clean", "nama_barang"]]
        .drop_duplicates()
        .set_index("nama_clean")["nama_barang"]
        .to_dict()
    )

    return df_full, mapping_nama


# =============== MODEL UTILS ===============
def train_evaluate_prophet(df_brand):
    data = (
        df_brand[["ds", "y"]]
        .groupby("ds", as_index=False)["y"]
        .sum()
        .sort_values("ds")
        .reset_index(drop=True)
    )

    if len(data) < 10:
        return np.nan, np.nan

    data["y_log"] = np.log1p(data["y"])

    split = int(len(data) * 0.8)
    train = data.iloc[:split].copy()
    test = data.iloc[split:].copy()

    train_prophet = train[["ds", "y_log"]].rename(columns={"y_log": "y"})
    test_dates = test[["ds"]]

    m = Prophet(
        seasonality_mode="multiplicative",
        yearly_seasonality=False,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.8,
        seasonality_prior_scale=10
    )
    m.add_country_holidays(country_name="ID")
    m.fit(train_prophet)

    fc = m.predict(test_dates)
    y_true = test["y"].values
    y_pred = np.expm1(fc["yhat"].values)

    mask = y_true != 0
    if mask.any():
        mape = mean_absolute_percentage_error(y_true[mask], y_pred[mask]) * 100
    else:
        mape = np.nan
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    return mape, rmse


@st.cache_data
def compute_metrics(df_full, mapping_nama):
    rows = []
    for nama_clean in df_full["nama_clean"].unique():
        df_b = df_full[df_full["nama_clean"] == nama_clean]
        mape, rmse = train_evaluate_prophet(df_b)
        rows.append({
            "nama_clean": nama_clean,
            "Nama Barang": mapping_nama.get(nama_clean, nama_clean),
            "MAPE_Prophet": mape,
            "RMSE_Prophet": rmse
        })
    hasil = pd.DataFrame(rows)
    return hasil


def train_full_and_forecast(df_brand, periods=30):
    data = (
        df_brand[["ds", "y"]]
        .groupby("ds", as_index=False)["y"]
        .sum()
        .sort_values("ds")
        .reset_index(drop=True)
    )

    data["y_log"] = np.log1p(data["y"])
    df_prophet = data[["ds", "y_log"]].rename(columns={"y_log": "y"})

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

    future = m.make_future_dataframe(periods=periods)
    fc = m.predict(future)
    fc["yhat_real"] = np.expm1(fc["yhat"])

    last_date = data["ds"].max()
    fc_future = fc[fc["ds"] > last_date].copy()

    return m, data, fc_future


# =============== MAIN APP ===============
def main():
    st.title("📈 Peramalan Permintaan Suku Cadang Bengkel (Prophet)")

    # # ---- Load & preprocess data ----
    # df_raw = load_raw_data()
    # df_full, mapping_nama = build_full_timeseries(df_raw)
    # hasil_metrics = compute_metrics(df_full, mapping_nama)

    df_full, hasil_metrics, mapping_nama = load_processed_data()

    # ---- Sidebar: pilih barang & horizon ----
    list_barang = (
        hasil_metrics.sort_values("Nama Barang")["Nama Barang"].tolist()
    )
    barang_selected = st.sidebar.selectbox("Pilih Barang", list_barang)

    nama_clean_selected = None
    for k, v in mapping_nama.items():
        if v == barang_selected:
            nama_clean_selected = k
            break

    horizon = st.sidebar.slider("Horizon Forecast (hari)", 7, 60, 30)

    # ---- Info MAPE & RMSE ----
    met_row = hasil_metrics[hasil_metrics["Nama Barang"] == barang_selected]
    col1, col2 = st.columns(2)
    with col1:
        st.metric("MAPE (test) – Prophet",
                  f"{met_row['MAPE_Prophet'].iloc[0]:.2f}%" if not met_row['MAPE_Prophet'].isna().iloc[0] else "-")
    with col2:
        st.metric("RMSE (test) – Prophet",
                  f"{met_row['RMSE_Prophet'].iloc[0]:.2f}" if not met_row['RMSE_Prophet'].isna().iloc[0] else "-")

    st.markdown("---")

    # ---- Forecast untuk barang terpilih ----
    df_b = df_full[df_full["nama_clean"] == nama_clean_selected].copy()
    m_full, data_hist, fc_future = train_full_and_forecast(df_b, periods=horizon)

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

    # ---- Komponen Prophet (opsional) ----
    with st.expander("Lihat Komponen Prophet (trend & musiman)"):
        from prophet.plot import plot_components_plotly
        # pakai matplotlib biar aman
        fig_comp = m_full.plot_components(
            m_full.predict(m_full.make_future_dataframe(periods=horizon))
        )
        st.pyplot(fig_comp)

    # ---- Tabel forecast 30 hari ke depan ----
    st.subheader("Tabel Forecast Harian")
    fc_table = fc_future[["ds", "yhat_real"]].copy()
    fc_table["Nama Barang"] = barang_selected
    fc_table = fc_table[["Nama Barang", "ds", "yhat_real"]]
    fc_table = fc_table.rename(columns={
        "ds": "Tanggal",
        "yhat_real": "Prediksi Jumlah"
    })
    st.dataframe(fc_table, use_container_width=True)

    avg_per_day = fc_table["Prediksi Jumlah"].mean()
    max_row = fc_table.loc[fc_table["Prediksi Jumlah"].idxmax()]

    st.write(f"- Rata-rata per hari: **{avg_per_day:.0f} unit**")
    st.write(f"- Perkiraan hari tersibuk: **{max_row['Tanggal'].date()}** dengan **{max_row['Prediksi Jumlah']:.0f} unit**")

    # Total 30 hari ke depan
    total_30 = fc_table["Prediksi Jumlah"].sum()
    st.info(f"Perkiraan total kebutuhan {barang_selected} selama {horizon} hari ke depan: **{total_30:.0f} unit**")

    st.markdown("---")

    # ---- Input transaksi baru ----
    st.subheader("➕ Input Transaksi Baru")

    with st.form("form_transaksi"):
        tgl_baru = st.date_input("Tanggal Transaksi")
        barang_baru = st.selectbox("Nama Barang", list_barang, key="barang_form")
        jumlah_baru = st.number_input("Jumlah", min_value=1, step=1, value=1)
        submitted = st.form_submit_button("Simpan Transaksi")

    if submitted:
        # mapping balik ke nama_clean / nama_barang asli
        # ambil nama_barang persis seperti di file
        nama_barang_asli = barang_baru

        new_row = {
            "Tanggal": pd.to_datetime(tgl_baru),
            "Nama Barang": nama_barang_asli,
            "Jumlah": jumlah_baru
        }

        # baca file asli, append, lalu simpan sebagai file baru
        df_now = pd.read_excel(DATA_PATH, sheet_name=SHEET_NAME)
        df_now = pd.concat([df_now, pd.DataFrame([new_row])], ignore_index=True)
        df_now.to_excel("data_transaksi_updated.xlsx", index=False)

        st.success("Transaksi baru berhasil disimpan ke file data_transaksi_updated.xlsx")
        st.caption("Untuk memperbarui model, jalankan ulang aplikasi dengan file terbaru.")

if __name__ == "__main__":
    main()
