import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)
LAST_YEAR = 2025      # 수업의 기준 기간 끝
MIN_DAYS = 300        # 관측일이 이보다 적은 해는 제외
SLIDER_MIN, SLIDER_MAX = 1900, 2100

st.set_page_config(page_title="기온 예측기", page_icon="🌡️", layout="wide")
st.title("🌡️ 기온 예측기")
st.caption("서울의 연평균기온으로 회귀 직선을 만들고, 고른 해의 기온을 예상해 봅니다.")


@st.cache_data(show_spinner="데이터를 불러오는 중...")
def load_yearly() -> pd.DataFrame:
    df = pd.read_csv(DATA_URL, encoding="utf-8")
    df.columns = [str(c).strip() for c in df.columns]
    df["날짜"] = pd.to_datetime(
        df["날짜"].astype(str).str.strip(), errors="coerce"
    )
    df["평균기온"] = pd.to_numeric(df["평균기온"], errors="coerce")
    df = df.dropna(subset=["날짜", "평균기온"])
    df["연도"] = df["날짜"].dt.year
    df = df[df["연도"] <= LAST_YEAR]

    yearly = (
        df.groupby("연도")["평균기온"]
        .agg(연평균기온="mean", 관측일수="count")
        .reset_index()
    )
    yearly = yearly[yearly["관측일수"] >= MIN_DAYS].reset_index(drop=True)
    return yearly


try:
    yearly = load_yearly()
except Exception as e:
    st.error(f"데이터를 불러오지 못했어요: {e}")
    st.stop()

if len(yearly) < 3:
    st.error("조건에 맞는 연도가 너무 적어서 회귀 직선을 만들 수 없어요.")
    st.stop()

# ---- 회귀 직선 ----
x = yearly["연도"].to_numpy(dtype=float)
y = yearly["연평균기온"].to_numpy(dtype=float)
slope, intercept = np.polyfit(x, y, 1)
r = float(np.corrcoef(x, y)[0, 1])

n_years = len(yearly)
start_year = int(yearly["연도"].min())
end_year = int(yearly["연도"].max())

# ---- 슬라이더와 예상 기온 ----
year = st.slider(
    "예상할 연도를 고르세요",
    min_value=SLIDER_MIN,
    max_value=SLIDER_MAX,
    value=min(max(LAST_YEAR, SLIDER_MIN), SLIDER_MAX),
    step=1,
)
pred = slope * year + intercept

st.markdown(
    f"""
    <div style="text-align:center; padding:1.2rem 0;">
        <div style="font-size:1.2rem; color:gray;">{year}년의 예상 연평균기온</div>
        <div style="font-size:4.5rem; font-weight:800; line-height:1.2;">{pred:.2f} ℃</div>
    </div>
    """,
    unsafe_allow_html=True,
)
if year < start_year or year > end_year:
    st.caption(
        f"※ {year}년은 직선을 만든 기간({start_year}~{end_year}년) 밖이라, "
        "직선을 그대로 늘려서 구한 값이에요."
    )

# ---- 요약 수치 ----
c1, c2, c3, c4 = st.columns(4)
c1.metric("직선을 만든 해의 개수", f"{n_years}개")
c2.metric("시작 연도", f"{start_year}년")
c3.metric("끝 연도", f"{end_year}년")
c4.metric("상관계수 (r)", f"{r:.3f}")
st.write(f"회귀 직선: **기온 = {slope:.4f} × 연도 + {intercept:.2f}**")

# ---- 기울기: 100년에 몇 도 오르는가 ----
st.markdown(
    f"""
    <div style="text-align:center; padding:1rem 0;">
        <div style="font-size:1.2rem; color:gray;">전체 기간 직선의 기울기 (100년 기준)</div>
        <div style="font-size:3.5rem; font-weight:800; line-height:1.2;">
            100년에 {slope * 100:+.2f} ℃
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

span_years = end_year - start_year + 1
recent_n = st.slider(
    "최근 몇 년으로 기울기를 구할까요?",
    min_value=5,
    max_value=max(span_years, 6),
    value=min(20, max(span_years, 6)),
    step=1,
    help="끝 연도를 포함해서 이 햇수만큼의 기간(달력 기준)으로 직선을 새로 구해요.",
)
recent = yearly[yearly["연도"] > end_year - recent_n]
has_recent = len(recent) >= 3
if has_recent:
    xr = recent["연도"].to_numpy(dtype=float)
    yr = recent["연평균기온"].to_numpy(dtype=float)
    slope_recent, intercept_recent = np.polyfit(xr, yr, 1)
    r_recent = float(np.corrcoef(xr, yr)[0, 1])

st.subheader(f"기울기 비교: 전체 기간 vs 최근 {recent_n}년")
col_all, col_recent = st.columns(2)
with col_all:
    st.markdown("#### 전체 기간")
    st.caption(f"{start_year}~{end_year}년, {n_years}개 해")
    st.markdown(
        f"<div style='font-size:2.6rem; font-weight:800;'>100년에 {slope * 100:+.2f} ℃</div>",
        unsafe_allow_html=True,
    )
    st.write(f"기울기 {slope:.4f} ℃/년 · 상관계수 r = {r:.3f}")
with col_recent:
    st.markdown(f"#### 최근 {recent_n}년")
    if has_recent:
        st.caption(
            f"{int(recent['연도'].min())}~{int(recent['연도'].max())}년, {len(recent)}개 해"
        )
        st.markdown(
            f"<div style='font-size:2.6rem; font-weight:800;'>100년에 {slope_recent * 100:+.2f} ℃</div>",
            unsafe_allow_html=True,
        )
        st.write(f"기울기 {slope_recent:.4f} ℃/년 · 상관계수 r = {r_recent:.3f}")
    else:
        st.write(f"최근 {recent_n}년 구간에 쓸 수 있는 해가 3개 미만이라 계산하지 않았어요.")

if has_recent:
    ratio_text = (
        f"최근 {recent_n}년의 상승 속도는 전체 기간의 약 {slope_recent / slope:.1f}배예요."
        if slope > 0 and slope_recent > 0
        else "두 기간의 기울기 방향이 달라요."
        if slope * slope_recent < 0
        else ""
    )
    st.info(
        f"100년 기준으로 환산하면 전체 기간은 {slope * 100:+.2f} ℃, "
        f"최근 {recent_n}년은 {slope_recent * 100:+.2f} ℃예요. {ratio_text} "
        "다만 기간이 짧을수록(점이 적을수록) 기울기가 크게 흔들릴 수 있어요."
    )

# ---- 그래프 ----
line_x = np.array([SLIDER_MIN, SLIDER_MAX], dtype=float)
line_y = slope * line_x + intercept

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=yearly["연도"],
        y=yearly["연평균기온"],
        mode="markers",
        name="연평균기온",
        marker=dict(size=8, opacity=0.8),
        customdata=yearly["관측일수"],
        hovertemplate="%{x}년<br>%{y:.2f} ℃<br>관측일수 %{customdata}일<extra></extra>",
    )
)
fig.add_trace(
    go.Scatter(
        x=line_x,
        y=line_y,
        mode="lines",
        name="회귀 직선",
        line=dict(width=3, color="crimson"),
    )
)
if has_recent:
    rx = np.array([recent["연도"].min(), recent["연도"].max()], dtype=float)
    fig.add_trace(
        go.Scatter(
            x=rx,
            y=slope_recent * rx + intercept_recent,
            mode="lines",
            name=f"최근 {recent_n}년 직선",
            line=dict(width=3, color="royalblue", dash="dash"),
        )
    )
fig.add_trace(
    go.Scatter(
        x=[year],
        y=[pred],
        mode="markers",
        name=f"{year}년 예상",
        marker=dict(size=16, color="gold", symbol="star", line=dict(width=1, color="black")),
        hovertemplate=f"{year}년 예상<br>{pred:.2f} ℃<extra></extra>",
    )
)
fig.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균기온 (℃)",
    xaxis=dict(range=[SLIDER_MIN - 2, SLIDER_MAX + 2]),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    height=520,
    margin=dict(l=20, r=20, t=40, b=20),
)
st.plotly_chart(fig, use_container_width=True)

with st.expander("사용한 연도별 데이터 보기"):
    st.dataframe(
        yearly.rename(columns={"연평균기온": "연평균기온(℃)"}),
        use_container_width=True,
        hide_index=True,
    )
