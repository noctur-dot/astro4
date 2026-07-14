import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from astropy.io import fits
from scipy.signal import find_peaks, savgol_filter

# ==========================
# 페이지 설정
# ==========================

st.set_page_config(
    page_title="Astronomy Spectrum Analyzer",
    page_icon="🔭",
    layout="wide"
)

st.title("🔭 Astronomy Spectrum Analyzer")
st.markdown("### 천체 스펙트럼 분석 프로그램")

st.markdown("""
이 프로그램은 천체의 스펙트럼 데이터를 분석하여

- 스펙트럼 그래프 생성
- 흡수선 자동 검출
- 주요 원소선 표시
- 별의 표면온도 추정
- 분광형 추정

을 수행합니다.
""")

st.divider()

# ==========================
# 사이드바
# ==========================

st.sidebar.header("설정")

smooth_window = st.sidebar.slider(
    "평활화(Window)",
    5,
    41,
    15,
    step=2
)

peak_prominence = st.sidebar.slider(
    "흡수선 민감도",
    0.01,
    0.30,
    0.05,
    step=0.01
)

# ==========================
# 파일 업로드
# ==========================

uploaded_file = st.file_uploader(
    "CSV 또는 FITS 파일을 업로드하세요.",
    type=["csv", "fits", "fit"]
)

# ==========================
# 데이터 읽기
# ==========================

def load_csv(file):

    df = pd.read_csv(file)

    if len(df.columns) < 2:
        st.error("CSV에는 최소 2개의 열이 필요합니다.")
        return None

    wavelength = df.iloc[:, 0].astype(float).to_numpy()
    intensity = df.iloc[:, 1].astype(float).to_numpy()

    return wavelength, intensity


def load_fits(file):

    hdul = fits.open(file)

    data = hdul[0].data

    hdul.close()

    intensity = np.array(data).flatten()

    wavelength = np.arange(len(intensity))

    return wavelength, intensity

# ==========================
# 별의 분광형
# ==========================

def spectral_type(temp):

    if temp >= 30000:
        return "O형"

    elif temp >= 10000:
        return "B형"

    elif temp >= 7500:
        return "A형"

    elif temp >= 6000:
        return "F형"

    elif temp >= 5200:
        return "G형"

    elif temp >= 3700:
        return "K형"

    else:
        return "M형"

# ==========================
# 원소선 정보
# ==========================

spectral_lines = {
    "Hα":656.3,
    "Hβ":486.1,
    "Na":589.0,
    "Ca K":393.4,
    "Ca H":396.8,
    "Mg":517.0,
    "He":587.6
}

# ==========================
# 데이터 처리
# ==========================

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

        # --------------------------

        smooth = savgol_filter(
            intensity,
            smooth_window,
            3
        )

        inverse = np.max(smooth) - smooth

        peaks, _ = find_peaks(
            inverse,
            prominence=np.max(inverse)*peak_prominence,
            distance=20
        )

        max_index = np.argmax(smooth)

        peak_wave = wavelength[max_index]

        peak_flux = smooth[max_index]
                # ==========================
        # 온도 계산 (빈의 변위 법칙)
        # ==========================

        if peak_wave > 0:

            temperature = 2.897e6 / peak_wave

        else:

            temperature = 0

        star_type = spectral_type(temperature)

        # ==========================
        # 정보 출력
        # ==========================

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "최고 세기 파장",
                f"{peak_wave:.2f} nm"
            )

        with col2:
            st.metric(
                "표면온도",
                f"{temperature:,.0f} K"
            )

        with col3:
            st.metric(
                "분광형",
                star_type
            )

        st.divider()

        # ==========================
        # 그래프
        # ==========================

        fig = go.Figure()

        fig.add_trace(

            go.Scatter(

                x=wavelength,

                y=smooth,

                mode="lines",

                name="Spectrum"

            )

        )

        # 최고점 표시

        fig.add_trace(

            go.Scatter(

                x=[peak_wave],

                y=[peak_flux],

                mode="markers",

                marker=dict(size=12),

                name="Maximum"

            )

        )

        # 흡수선 표시

        fig.add_trace(

            go.Scatter(

                x=wavelength[peaks],

                y=smooth[peaks],

                mode="markers",

                marker=dict(size=8),

                name="Absorption"

            )

        )

        # 대표 원소선 표시

        for name, wave in spectral_lines.items():

            fig.add_vline(

                x=wave,

                line_dash="dash",

                annotation_text=name

            )

        fig.update_layout(

            title="Spectrum Analysis",

            xaxis_title="Wavelength (nm)",

            yaxis_title="Intensity",

            height=650,

            hovermode="x unified"

        )

        st.plotly_chart(

            fig,

            use_container_width=True

        )

        st.divider()
                # ==========================
        # 검출된 흡수선 목록
        # ==========================

        st.subheader("🔎 검출된 흡수선")

        line_df = pd.DataFrame({
            "번호": np.arange(1, len(peaks)+1),
            "파장 (nm)": wavelength[peaks],
            "세기": smooth[peaks]
        })

        st.dataframe(
            line_df,
            use_container_width=True
        )

        st.divider()

        # ==========================
        # 원소 후보 분석
        # ==========================

        st.subheader("🧪 예상되는 원소")

        detected_elements = []

        tolerance = 2.0

        for element, line in spectral_lines.items():

            for w in wavelength[peaks]:

                if abs(w - line) <= tolerance:
                    detected_elements.append(element)
                    break

        if len(detected_elements) == 0:

            st.warning("일치하는 대표 원소선을 찾지 못했습니다.")

        else:

            detected_elements = sorted(list(set(detected_elements)))

            result_df = pd.DataFrame({

                "검출 원소": detected_elements

            })

            st.dataframe(
                result_df,
                use_container_width=True
            )

        st.divider()

        # ==========================
        # 분석 결과 요약
        # ==========================

        st.subheader("📋 분석 결과")

        summary = pd.DataFrame({

            "항목":[
                "최고 세기 파장",
                "추정 표면온도",
                "분광형",
                "검출된 흡수선 개수"
            ],

            "값":[
                f"{peak_wave:.2f} nm",
                f"{temperature:.0f} K",
                star_type,
                len(peaks)
            ]

        })

        st.table(summary)

        st.divider()

        # ==========================
        # CSV 다운로드
        # ==========================

        csv = summary.to_csv(index=False).encode("utf-8-sig")

        st.download_button(

            label="📥 분석 결과 다운로드 (CSV)",

            data=csv,

            file_name="spectrum_result.csv",

            mime="text/csv"

        )
    else:

    st.info("👈 왼쪽에서 CSV 또는 FITS 파일을 업로드하세요.")

    st.markdown("---")

    st.subheader("📖 사용 방법")

    st.markdown("""
### 1️⃣ CSV 파일

CSV는 아래와 같은 형식이어야 합니다.

| Wavelength | Intensity |
|------------|-----------|
|380|124|
|381|126|
|382|128|

첫 번째 열 : 파장(nm)

두 번째 열 : 세기(Intensity)

---

### 2️⃣ FITS 파일

천문관측에서 사용하는 FITS 파일도 업로드할 수 있습니다.

---

### 3️⃣ 분석 기능

업로드 후 자동으로

- 스펙트럼 그래프 생성
- 흡수선 검출
- 최고 세기 파장 계산
- 표면온도 계산
- 분광형 추정
- 원소 후보 분석

을 수행합니다.
""")

    st.markdown("---")

    st.subheader("⭐ 분광형 기준")

    spectral_df = pd.DataFrame({

        "분광형":[
            "O",
            "B",
            "A",
            "F",
            "G",
            "K",
            "M"
        ],

        "표면온도(K)":[
            ">30000",
            "10000~30000",
            "7500~10000",
            "6000~7500",
            "5200~6000",
            "3700~5200",
            "<3700"
        ],

        "대표 색":[
            "청색",
            "청백색",
            "백색",
            "황백색",
            "황색",
            "주황색",
            "적색"
        ]

    })

    st.dataframe(
        spectral_df,
        use_container_width=True
    )

    st.markdown("---")

    st.caption("Astronomy Spectrum Analyzer | Streamlit Project")
