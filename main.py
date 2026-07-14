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
# 데이터 전처리
# -----------------------------

# 노이즈 제거
smooth = savgol_filter(intensity, 15, 3)

# 최고 세기
max_index = np.argmax(smooth)
max_wave = wavelength[max_index]
max_intensity = smooth[max_index]

# 흡수선 탐색
inverse = np.max(smooth) - smooth

peaks, _ = find_peaks(
    inverse,
    prominence=np.max(inverse) * 0.05,
    distance=20
)

# -----------------------------
# 결과 출력
# -----------------------------

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "최고 세기 파장",
        f"{max_wave:.2f} nm"
    )

with col2:
    st.metric(
        "검출된 흡수선",
        len(peaks)
    )

# -----------------------------
# 그래프
# -----------------------------

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=wavelength,
        y=smooth,
        mode="lines",
        name="Spectrum"
    )
)

# 최고점
fig.add_trace(
    go.Scatter(
        x=[max_wave],
        y=[max_intensity],
        mode="markers",
        marker=dict(size=10),
        name="Maximum"
    )
)

# 흡수선
fig.add_trace(
    go.Scatter(
        x=wavelength[peaks],
        y=smooth[peaks],
        mode="markers",
        marker=dict(size=8),
        name="Absorption Lines"
    )
)

# Hα
fig.add_vline(
    x=656.3,
    line_dash="dash",
    annotation_text="Hα"
)

# Hβ
fig.add_vline(
    x=486.1,
    line_dash="dash",
    annotation_text="Hβ"
)

fig.update_layout(
    title="Spectrum Analysis",
    xaxis_title="Wavelength (nm)",
    yaxis_title="Intensity",
    height=600
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# -----------------------------
# 흡수선 목록
# -----------------------------

st.subheader("검출된 흡수선")

line_df = pd.DataFrame({
    "번호": np.arange(1, len(peaks)+1),
    "파장(nm)": wavelength[peaks],
    "세기": smooth[peaks]
})

st.dataframe(
    line_df,
    use_container_width=True
)
else:

    st.info("왼쪽에서 CSV 또는 FITS 파일을 업로드하세요.")
