
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.signal import find_peaks, savgol_filter
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

        # -----------------------------
        # 데이터 분석
        # -----------------------------
        
        smooth = savgol_filter(intensity, 15, 3)
        
        inverse = np.max(smooth) - smooth
        
        peaks, _ = find_peaks(
            inverse,
            prominence=np.max(inverse) * 0.05,
            distance=20
        )
        
        max_index = np.argmax(smooth)
        
        peak_wave = wavelength.iloc[max_index] if hasattr(wavelength, "iloc") else wavelength[max_index]
        peak_flux = smooth[max_index]
        
        temperature = 2.897e6 / peak_wave
        
        if temperature >= 30000:
            star = "O형"
        elif temperature >= 10000:
            star = "B형"
        elif temperature >= 7500:
            star = "A형"
        elif temperature >= 6000:
            star = "F형"
        elif temperature >= 5200:
            star = "G형"
        elif temperature >= 3700:
            star = "K형"
        else:
            star = "M형"
        
        col1, col2, col3 = st.columns(3)
        
        col1.metric("최고 파장", f"{peak_wave:.1f} nm")
        col2.metric("표면온도", f"{temperature:.0f} K")
        col3.metric("분광형", star)
        
        fig = go.Figure()
        
        fig.add_trace(
            go.Scatter(
                x=wavelength,
                y=smooth,
                mode="lines",
                name="Spectrum"
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=[peak_wave],
                y=[peak_flux],
                mode="markers",
                name="Maximum"
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=wavelength.iloc[peaks] if hasattr(wavelength, "iloc") else wavelength[peaks],
                y=smooth[peaks],
                mode="markers",
                name="Absorption"
            )
        )
        
        fig.add_vline(x=656.3, line_dash="dash", annotation_text="Hα")
        fig.add_vline(x=486.1, line_dash="dash", annotation_text="Hβ")
        
        fig.update_layout(
            title="Spectrum Analysis",
            xaxis_title="Wavelength (nm)",
            yaxis_title="Intensity",
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("검출된 흡수선")
        
        line_df = pd.DataFrame({
            "파장(nm)": wavelength.iloc[peaks] if hasattr(wavelength, "iloc") else wavelength[peaks],
            "세기": smooth[peaks]
        })
        
        st.dataframe(line_df, use_container_width=True)

else:

    st.info("왼쪽에서 CSV 또는 FITS 파일을 업로드하세요.")
