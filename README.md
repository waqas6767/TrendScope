# 🌍 TrendScope — Earth System Trend Detective

**NASA Space Apps Challenge 2026 · Challenge: Be An Earth System Trend Detective! · Team: TrendScope | Challengers**

**🚀 Live app:** https://km2sqxzt4mdntwcidhwfvj.streamlit.app *(deployed after the Space Apps submission; app code unchanged)*

TrendScope answers four questions about NASA Earth data: **what** is changing, **where**, **by how much**, and **is it statistically significant?**

## Features
- Monthly NASA POWER data (MERRA-2 meteorology, CERES radiation) for any location, 1981 onward — no login needed
- Seasonal cycle removed via monthly anomalies
- Trend magnitude with **Sen's slope**
- Significance with the **Mann-Kendall test + Hamed-Rao autocorrelation correction** (naive p-value shown for comparison)
- **Pettitt** change-point detection
- Multi-location map with **Benjamini-Hochberg False Discovery Rate** control
- Plain-language summary of every result

## Demo
[TrendScope_demo.mp4](TrendScope_demo.mp4) (29 s)

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Example result
Air temperature in Bandung, Indonesia (1985–2024): **+0.13 °C per decade, statistically significant**. Across 8 Asian/African cities, 6 show significant trends after correction, versus 7 with a naive test.

## Data
NASA POWER API — https://power.larc.nasa.gov

## References
Mann (1945); Kendall (1975); Sen (1968); Hamed & Rao (1998); Pettitt (1979); Benjamini & Hochberg (1995); Wilks (2016).

## Team
Muhammad Waqas · Shaista Saddique
