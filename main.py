
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.signal import find_peaks
from astropy.io import fits

# -----------------------------
# 페이지 설정
# -----------------------------
st.set_page_config(
    page_title="Astronomy Spectrum Analyzer",
    page_icon="🔭",
    layout="wide"
)

st.title("🔭 Astronomy Spectrum Analyzer")
st.markdown("### 천체 스펙트럼 분석 프로그램")

st.write("""
이 앱은 천체의 스펙트럼 데이터를 분석하여

- 흡수선 및 방출선 확인
- 최고 세기 파장 찾기
- 별의 온도 추정
- 분광형 추정

을 수행합니다.
""")

st.divider()

# -----------------------------
# 파일 업로드
# -----------------------------

uploaded_file = st.file_uploader(
    "CSV 또는 FITS 파일을 업로드하세요.",
    type=["csv", "fits", "fit"]
)

# -----------------------------
# CSV 읽기
# -----------------------------

def load_csv(file):

    df = pd.read_csv(file)

    if len(df.columns) < 2:
        st.error("CSV는 최소 2개의 열(파장, 세기)이 필요합니다.")
        return None

    wavelength = df.iloc[:,0]
    intensity = df.iloc[:,1]

    return wavelength, intensity


# -----------------------------
# FITS 읽기
# -----------------------------

def load_fits(file):

    hdul = fits.open(file)

    data = hdul[0].data

    hdul.close()

    intensity = np.array(data).flatten()

    wavelength = np.arange(len(intensity))

    return wavelength, intensity

# -----------------------------
# 데이터 불러오기
# -----------------------------

if uploaded_file:

    if uploaded_file.name.endswith(".csv"):

        result = load_csv(uploaded_file)

    else:

        result = load_fits(uploaded_file)

    if result is not None:

        wavelength, intensity = result

        st.success("데이터를 성공적으로 불러왔습니다.")

        st.write("데이터 개수 :", len(wavelength))

        st.dataframe(
            pd.DataFrame({
                "Wavelength": wavelength,
                "Intensity": intensity
            }).head()
        )

        # -----------------------------
        # 그래프
        # -----------------------------

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=wavelength,
                y=intensity,
                mode="lines",
                name="Spectrum"
            )
        )

        fig.update_layout(
            title="Spectrum",
            xaxis_title="Wavelength",
            yaxis_title="Intensity"
        )

        st.plotly_chart(fig, use_container_width=True)

else:

    st.info("왼쪽에서 CSV 또는 FITS 파일을 업로드하세요.")
