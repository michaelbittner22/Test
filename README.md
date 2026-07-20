# 🍫 Forecast Bias Detector

A Streamlit app for demand planners to identify SKU/market combinations where
**Lag-0 Forecast** has been persistently biased vs **Shipped Units**.

You choose:
- **X** — the bias threshold (%)
- **Y** — the number of consecutive months the bias must persist
- The **direction** (over-forecast, under-forecast, or both)
- The **grain** to analyze at (SKU, Brand, Customer Group, Region, Cluster, Sub-Cluster — any combination)

The app then flags every line whose longest streak of months breaching the
threshold is `>= Y`, and lets you drill into trend charts and export results.

---

## 1. Project structure

```
chocolate-bias-app/
├── app.py                      # Streamlit app
├── requirements.txt            # Python dependencies
├── data/
│   └── sample_forecast_shipments.csv   # 60-SKU synthetic demo dataset (12 months)
├── utils/
│   └── generate_data.py        # Script used to (re)generate the sample dataset
└── README.md
```

## 2. Data format

Whether you use the bundled sample data or upload your own CSV, it must have
these columns (case-sensitive):

| Column                | Type          | Description                                  |
|------------------------|---------------|-----------------------------------------------|
| `Date`                 | date          | Month, e.g. `2026-06-01`                      |
| `Region`               | text          | e.g. `EMEA`                                    |
| `Cluster`               | text          | e.g. `EMEA North`, `EMEA South`, `EMEA Central`, `EMEA East` |
| `Sub_Cluster`           | text          | e.g. `Nordics`, `DACH`, `Adria`, `Iberia`      |
| `Customer_Group`        | text          | e.g. `Grocery Retail`, `Discounter`, `E-commerce` |
| `Brand`                 | text          | e.g. `Choco Supreme`, `Velvet Cacao`           |
| `SKU_ID`                | text          | Unique SKU code                                |
| `SKU_Name`              | text          | SKU description                                |
| `Forecast_Lag0_Units`   | number        | Lag-0 forecast, in units                       |
| `Shipped_Units`         | number        | Actual shipped units                           |

One row = one SKU × market combination × month. You don't need to pre-aggregate —
the app aggregates to whatever grain you select in the sidebar.

The included `data/sample_forecast_shipments.csv` has **60 SKUs**, **6 brands**,
**7 customer groups**, **4 clusters / 11 sub-clusters**, and **12 months** of
history (~3,200 rows), with a mix of accurate, chronically-biased, improving,
deteriorating, volatile, and spike-pattern SKUs so the flagging logic has
realistic cases to surface.

## 3. Run locally

```bash
git clone <your-repo-url>
cd chocolate-bias-app
pip install -r requirements.txt
streamlit run app.py
```

The app opens at `http://localhost:8501`. Use the sidebar to switch from the
sample dataset to your own CSV upload at any time.

## 4. Deploy on Streamlit Community Cloud (via GitHub)

1. **Create a GitHub repo** and push this folder's contents to it:
   ```bash
   cd chocolate-bias-app
   git init
   git add .
   git commit -m "Initial commit: forecast bias detector"
   git branch -M main
   git remote add origin https://github.com/<your-username>/<your-repo>.git
   git push -u origin main
   ```
2. Go to **[share.streamlit.io](https://share.streamlit.io)** and sign in with GitHub.
3. Click **"New app"**, select your repository, branch (`main`), and set the
   main file path to `app.py`.
4. Click **Deploy**. Streamlit Cloud will install `requirements.txt` automatically
   and give you a public URL (`https://<your-app>.streamlit.app`).
5. Any future `git push` to `main` will auto-redeploy the app.

> No secrets or API keys are required for this app — it runs entirely on the
> uploaded/sample CSV data in-session.

## 5. Regenerating the sample dataset

If you want a different random sample (different seed, more SKUs, etc.), edit
`utils/generate_data.py` and re-run:

```bash
cd utils
python generate_data.py
```

This overwrites `data/sample_forecast_shipments.csv`.

## 6. How bias & flagging are calculated

For each analyzed line (at your chosen grain) and month:

```
Bias % = (Forecast_Lag0_Units − Shipped_Units) / Shipped_Units × 100
```

- **Positive** = over-forecast (forecast exceeded what shipped)
- **Negative** = under-forecast (forecast fell short of what shipped)

A line is **flagged** if its longest run of *consecutive* months breaching the
chosen `X%` threshold (in the chosen direction) is `>= Y` months.

## 7. License / usage

Sample data is entirely synthetic and randomly generated — safe to use for
demos, testing, and internal training. Replace with real forecast/shipment
extracts for production use.
