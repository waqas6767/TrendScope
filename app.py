"""
TrendScope — Earth System Trend Detective
NASA Space Apps Challenge 2026 — Team TrendScope | Challengers

Detects, quantifies and tests the statistical significance of trends in
NASA Earth data (NASA POWER: MERRA-2 / CERES-derived monthly time series).

Pipeline per location:
  1. Download monthly data from the NASA POWER API (no login needed)
  2. Remove the seasonal cycle (monthly anomalies)
  3. Trend magnitude: Sen's slope
  4. Significance: Mann-Kendall with Hamed-Rao autocorrelation correction
  5. Change point: Pettitt test
  6. Multiple locations: Benjamini-Hochberg False Discovery Rate control
  7. Plain-language summary

Run:  pip install -r requirements.txt  &&  streamlit run app.py
"""

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pymannkendall as mk

POWER_URL = "https://power.larc.nasa.gov/api/temporal/monthly/point"

VARIABLES = {
    "T2M": ("Air temperature at 2 m", "°C"),
    "PRECTOTCORR": ("Precipitation", "mm/day"),
    "RH2M": ("Relative humidity at 2 m", "%"),
    "ALLSKY_SFC_SW_DWN": ("Surface solar radiation (all-sky)", "kWh/m²/day"),
    "WS2M": ("Wind speed at 2 m", "m/s"),
    "T2M_MAX": ("Maximum air temperature at 2 m", "°C"),
}

PRESET_LOCATIONS = {
    "Bandung, Indonesia": (-6.91, 107.61),
    "Jakarta, Indonesia": (-6.21, 106.85),
    "Surabaya, Indonesia": (-7.25, 112.75),
    "Kuala Lumpur, Malaysia": (3.14, 101.69),
    "Lahore, Pakistan": (31.55, 74.34),
    "Bahawalpur, Pakistan": (29.40, 71.68),
    "Delhi, India": (28.61, 77.21),
    "Cairo, Egypt": (30.04, 31.24),
    "Phoenix, USA": (33.45, -112.07),
    "Madrid, Spain": (40.42, -3.70),
    "Sydney, Australia": (-33.87, 151.21),
    "Nairobi, Kenya": (-1.29, 36.82),
    "São Paulo, Brazil": (-23.55, -46.63),
    "Reykjavik, Iceland": (64.15, -21.94),
}


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def fetch_power_monthly(param, lat, lon, start, end):
    """Return a monthly pandas Series from NASA POWER (fill values removed)."""
    params = {
        "parameters": param,
        "community": "AG",
        "longitude": lon,
        "latitude": lat,
        "start": start,
        "end": end,
        "format": "JSON",
    }
    r = requests.get(POWER_URL, params=params, timeout=60)
    r.raise_for_status()
    raw = r.json()["properties"]["parameter"][param]
    recs = {}
    for key, val in raw.items():
        month = int(key[4:])
        if month == 13:  # annual mean entry
            continue
        if val is None or val <= -998:
            continue
        recs[pd.Timestamp(int(key[:4]), month, 1)] = val
    return pd.Series(recs).sort_index()


# ----------------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------------
def deseasonalize(s):
    """Monthly anomalies: subtract each calendar month's long-term mean."""
    clim = s.groupby(s.index.month).transform("mean")
    return s - clim


def pettitt_test(x):
    """Pettitt (1979) non-parametric change-point test. Returns (index, p)."""
    x = np.asarray(x)
    n = len(x)
    sgn = np.sign(x[:, None] - x[None, :])
    U = np.array([sgn[: k + 1, k + 1:].sum() for k in range(n - 1)])
    K = np.max(np.abs(U))
    idx = int(np.argmax(np.abs(U)))
    p = 2.0 * np.exp(-6.0 * K**2 / (n**3 + n**2))
    return idx, min(p, 1.0)


def benjamini_hochberg(pvals):
    """Return BH-adjusted q-values."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(q, 1.0)
    return out


def analyze(s):
    """Full trend analysis of one monthly series."""
    anom = deseasonalize(s)
    res = mk.hamed_rao_modification_test(anom.values)
    naive = mk.original_test(anom.values)
    slope_per_month = res.slope
    slope_decade = slope_per_month * 120
    total_change = slope_per_month * (len(anom) - 1)
    cp_idx, cp_p = pettitt_test(anom.values)
    return {
        "anom": anom,
        "trend": res.trend,
        "p": res.p,
        "p_naive": naive.p,
        "slope_decade": slope_decade,
        "intercept": res.intercept,
        "slope_month": slope_per_month,
        "total_change": total_change,
        "cp_date": anom.index[cp_idx + 1] if cp_p < 0.05 else None,
        "cp_p": cp_p,
        "n": len(anom),
    }


def plain_language(name, unit, place, r, alpha):
    sig = r["p"] < alpha
    direction = "increased" if r["slope_decade"] > 0 else "decreased"
    txt = (
        f"**{name}** in **{place}** has {direction} by "
        f"**{abs(r['slope_decade']):.3f} {unit} per decade** "
        f"(≈ {abs(r['total_change']):.2f} {unit} over the period). "
    )
    if sig:
        txt += (
            f"This trend is **statistically significant** (p = {r['p']:.4f}, "
            f"Mann-Kendall with autocorrelation correction)."
        )
    else:
        txt += (
            f"This trend is **not statistically significant** (p = {r['p']:.3f}); "
            f"it cannot be distinguished from natural variability."
        )
    if r["p_naive"] < alpha <= r["p"]:
        txt += (
            " ⚠️ A naive test (ignoring autocorrelation) would have wrongly "
            f"called this significant (p = {r['p_naive']:.4f})."
        )
    if r["cp_date"] is not None:
        txt += (
            f" A shift in the series is detected around "
            f"**{r['cp_date']:%B %Y}** (Pettitt p = {r['cp_p']:.3f})."
        )
    return txt


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
st.set_page_config(page_title="TrendScope", page_icon="🌍", layout="wide")
st.title("🌍 TrendScope — Earth System Trend Detective")
st.caption(
    "What is changing, where, by how much, and is it significant? "
    "Data: NASA POWER (monthly, MERRA-2 / CERES based)."
)

with st.sidebar:
    st.header("Settings")
    var = st.selectbox(
        "Variable", list(VARIABLES), format_func=lambda k: VARIABLES[k][0]
    )
    start, end = st.slider("Years", 1981, 2025, (1985, 2024))
    alpha = st.select_slider("Significance level α", [0.01, 0.05, 0.10], 0.05)
    mode = st.radio("Mode", ["Single location", "Compare locations (map)"])

name, unit = VARIABLES[var]

if mode == "Single location":
    c1, c2 = st.columns([2, 1])
    with c1:
        place = st.selectbox("Location", list(PRESET_LOCATIONS) + ["Custom"])
    if place == "Custom":
        lat = c2.number_input("Latitude", -90.0, 90.0, 0.0)
        lon = c2.number_input("Longitude", -180.0, 180.0, 0.0)
        place = f"({lat:.2f}, {lon:.2f})"
    else:
        lat, lon = PRESET_LOCATIONS[place]

    try:
        with st.spinner("Downloading NASA POWER data..."):
            s = fetch_power_monthly(var, lat, lon, start, end)
    except Exception as e:
        st.error(f"Could not fetch data: {e}")
        st.stop()

    r = analyze(s)
    st.markdown(plain_language(name, unit, place, r, alpha))

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Trend", f"{r['slope_decade']:+.3f} {unit}/decade")
    m2.metric("p-value (corrected)", f"{r['p']:.4f}")
    m3.metric("p-value (naive)", f"{r['p_naive']:.4f}")
    m4.metric("Significant?", "Yes ✅" if r["p"] < alpha else "No ❌")

    t = np.arange(r["n"])
    fit = r["intercept"] + r["slope_month"] * t
    fig = go.Figure()
    fig.add_scatter(x=s.index, y=s.values, name="Monthly values",
                    line=dict(color="lightgray"))
    fig.update_layout(title=f"{name} — raw monthly data", yaxis_title=unit,
                      height=320, margin=dict(t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)

    fig2 = go.Figure()
    fig2.add_scatter(x=r["anom"].index, y=r["anom"].values,
                     name="Anomaly (seasonal cycle removed)",
                     line=dict(color="steelblue"))
    fig2.add_scatter(x=r["anom"].index,
                     y=r["anom"].rolling(12, center=True).mean(),
                     name="12-month mean", line=dict(color="navy", width=2))
    fig2.add_scatter(x=r["anom"].index, y=fit, name="Sen's slope trend",
                     line=dict(color="crimson", width=3, dash="dash"))
    if r["cp_date"] is not None:
        fig2.add_vline(x=r["cp_date"], line_dash="dot", line_color="orange")
    fig2.update_layout(title="Anomalies, trend and change point",
                       yaxis_title=f"Anomaly ({unit})", height=380,
                       margin=dict(t=40, b=20))
    st.plotly_chart(fig2, use_container_width=True)

else:
    chosen = st.multiselect(
        "Locations", list(PRESET_LOCATIONS), default=list(PRESET_LOCATIONS)[:8]
    )
    if not chosen:
        st.stop()
    rows = []
    prog = st.progress(0.0, "Downloading NASA POWER data...")
    for i, p in enumerate(chosen):
        lat, lon = PRESET_LOCATIONS[p]
        try:
            r = analyze(fetch_power_monthly(var, lat, lon, start, end))
            rows.append({"Location": p, "lat": lat, "lon": lon,
                         "Trend per decade": r["slope_decade"],
                         "p (corrected)": r["p"], "p (naive)": r["p_naive"]})
        except Exception as e:
            st.warning(f"{p}: {e}")
        prog.progress((i + 1) / len(chosen))
    prog.empty()
    df = pd.DataFrame(rows)
    df["q (FDR)"] = benjamini_hochberg(df["p (corrected)"])
    df["Significant (FDR)"] = df["q (FDR)"] < alpha
    df["Naively significant"] = df["p (naive)"] < alpha

    n_sig = int(df["Significant (FDR)"].sum())
    n_naive = int(df["Naively significant"].sum())
    st.markdown(
        f"**{n_sig} of {len(df)}** locations show a significant trend in "
        f"**{name}** after autocorrelation correction and False Discovery "
        f"Rate control (α = {alpha}). A naive analysis would have flagged "
        f"**{n_naive}**."
    )

    df["Status"] = np.where(df["Significant (FDR)"], "Significant",
                            "Not significant")
    lim = df["Trend per decade"].abs().max()
    fig = px.scatter_geo(
        df, lat="lat", lon="lon", color="Trend per decade",
        symbol="Status", hover_name="Location",
        hover_data={"Trend per decade": ":.3f", "q (FDR)": ":.4f",
                    "lat": False, "lon": False},
        color_continuous_scale="RdBu_r", range_color=(-lim, lim),
        projection="natural earth",
    )
    fig.update_traces(marker=dict(size=14, line=dict(width=1, color="black")))
    fig.update_layout(height=500, margin=dict(t=20, b=0),
                      coloraxis_colorbar_title=f"{unit}/decade")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(
        df.drop(columns=["lat", "lon", "Status"]).round(4),
        use_container_width=True, hide_index=True,
    )

with st.expander("Methods"):
    st.markdown(
        """
- **Seasonality removal:** monthly anomalies relative to each calendar month's mean.
- **Trend magnitude:** Sen's slope (robust to outliers).
- **Significance:** Mann-Kendall test with the Hamed-Rao (1998) correction for
  serial autocorrelation. Earth data are autocorrelated; ignoring this inflates
  false positives, so the naive p-value is shown for comparison.
- **Change point:** Pettitt (1979) test.
- **Multiple testing:** Benjamini-Hochberg False Discovery Rate across locations.
- **Data:** NASA POWER monthly point data (MERRA-2 meteorology, CERES/GEWEX radiation).
"""
    )
