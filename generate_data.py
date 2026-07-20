"""
Generates a realistic, 12-month synthetic Lag-0 Forecast vs Shipped Units
dataset for a chocolate company demand-planning bias app.

Hierarchy:
  Region -> Cluster -> Sub-Cluster -> Customer Group
  Brand -> SKU

Run:  python generate_data.py
Output: ../data/sample_forecast_shipments.csv
"""

import numpy as np
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta

rng = np.random.default_rng(42)

# ---------------------------------------------------------------------
# 1. Reference hierarchies
# ---------------------------------------------------------------------

REGION = "EMEA"

CLUSTERS = {
    "EMEA North": ["Nordics", "UK & Ireland", "Baltics"],
    "EMEA South": ["Iberia", "Italy", "Adria"],
    "EMEA Central": ["DACH", "Benelux", "France"],
    "EMEA East": ["Poland", "CEE", "Turkey"],
}

CUSTOMER_GROUPS = [
    "Grocery Retail",
    "Discounter",
    "Wholesale / Cash & Carry",
    "E-commerce",
    "Convenience",
    "Travel Retail",
    "Foodservice",
]

BRANDS = {
    "Choco Supreme": "Premium boxed chocolates",
    "Velvet Cacao": "Dark chocolate specialist",
    "Nutty Delight": "Nut & praline range",
    "Pure Origin": "Single-origin / ethical cacao",
    "Milko Classic": "Mainstream milk chocolate",
    "Snap & Share": "Impulse / sharing bars",
}

PRODUCT_WORDS = {
    "Choco Supreme": ["Gift Box", "Truffle Collection", "Pralines Assortment", "Signature Bar", "Selection Tin"],
    "Velvet Cacao": ["70% Dark Bar", "85% Dark Bar", "Dark Truffles", "Cacao Nibs Bar", "Espresso Dark Bar"],
    "Nutty Delight": ["Hazelnut Bar", "Almond Praline", "Peanut Crunch", "Pistachio Bar", "Walnut Delight"],
    "Pure Origin": ["Ecuador 75%", "Madagascar 68%", "Peru Single Origin", "Ghana Reserve", "Vietnam Cacao Bar"],
    "Milko Classic": ["Milk Bar", "Milk & Caramel", "Milk & Hazelnut", "Family Block", "Milk Buttons"],
    "Snap & Share": ["Sharing Bag", "Mini Bars Pouch", "Multipack 6x", "Snack Size 4x", "Duo Bar"],
}

PACK_SIZES = ["100g", "150g", "200g", "250g", "300g"]

BIAS_PROFILES = [
    "accurate",            # small random noise, no persistent bias
    "chronic_over",        # persistently over-forecast (forecast > shipped)
    "chronic_under",       # persistently under-forecast (forecast < shipped)
    "improving",           # starts biased, trends toward accurate
    "deteriorating",       # starts accurate, drifts into bias
    "volatile",            # large swings both directions, no consistent streak
    "seasonal_spike",      # one bad patch (e.g. promo/launch) mid-year, else fine
]

PROFILE_WEIGHTS = [0.28, 0.20, 0.20, 0.10, 0.10, 0.07, 0.05]

# ---------------------------------------------------------------------
# 2. Build SKU master list (50+ SKUs)
# ---------------------------------------------------------------------

skus = []
sku_id = 1
for brand, words in PRODUCT_WORDS.items():
    for w in words:
        for pack in rng.choice(PACK_SIZES, size=rng.integers(2, 3), replace=False):
            skus.append({
                "SKU_ID": f"CHO-{sku_id:04d}",
                "SKU_Name": f"{brand} {w} {pack}",
                "Brand": brand,
                "Pack_Size": pack,
            })
            sku_id += 1

sku_df = pd.DataFrame(skus)
print(f"Generated {len(sku_df)} SKUs")

# ---------------------------------------------------------------------
# 3. Assign each SKU a set of distribution combos (cluster/sub-cluster/customer group)
#    and a bias profile per combo, plus a base monthly volume.
# ---------------------------------------------------------------------

all_subclusters = [(c, s) for c, subs in CLUSTERS.items() for s in subs]

records = []
today = datetime(2026, 7, 1)
months = [today - relativedelta(months=i) for i in range(12, 0, -1)]  # last 12 full months

for _, sku in sku_df.iterrows():
    n_combos = rng.integers(3, 7)  # each SKU distributed to 3-6 market combos
    combo_idx = rng.choice(len(all_subclusters), size=n_combos, replace=False)

    for ci in combo_idx:
        cluster, subcluster = all_subclusters[ci]
        cust_group = rng.choice(CUSTOMER_GROUPS)
        profile = rng.choice(BIAS_PROFILES, p=PROFILE_WEIGHTS)

        base_volume = rng.integers(800, 15000)  # baseline monthly shipped units
        trend = rng.uniform(-0.01, 0.02)  # slight organic monthly growth/decline
        noise_sd = rng.uniform(0.04, 0.10)  # natural demand noise

        # bias trajectory (% forecast bias vs shipped) per month, depends on profile
        if profile == "accurate":
            bias_path = rng.normal(0, 4, size=12)
        elif profile == "chronic_over":
            bias_path = rng.normal(rng.uniform(15, 35), 5, size=12)
        elif profile == "chronic_under":
            bias_path = rng.normal(-rng.uniform(15, 35), 5, size=12)
        elif profile == "improving":
            start = rng.uniform(25, 45) * rng.choice([-1, 1])
            bias_path = np.linspace(start, start * 0.1, 12) + rng.normal(0, 4, size=12)
        elif profile == "deteriorating":
            end = rng.uniform(25, 45) * rng.choice([-1, 1])
            bias_path = np.linspace(end * 0.1, end, 12) + rng.normal(0, 4, size=12)
        elif profile == "volatile":
            bias_path = rng.normal(0, 22, size=12)
        elif profile == "seasonal_spike":
            bias_path = rng.normal(0, 4, size=12)
            spike_start = rng.integers(2, 8)
            spike_len = rng.integers(2, 4)
            spike_sign = rng.choice([-1, 1])
            bias_path[spike_start:spike_start + spike_len] += spike_sign * rng.uniform(30, 55)

        for m_idx, month in enumerate(months):
            seasonal = 1 + 0.15 * np.sin(2 * np.pi * (month.month / 12))  # mild seasonality
            shipped = max(50, base_volume * seasonal * (1 + trend * m_idx) * (1 + rng.normal(0, noise_sd)))
            shipped = round(shipped)

            bias_pct = bias_path[m_idx]
            forecast = max(0, round(shipped * (1 + bias_pct / 100)))

            records.append({
                "Date": month.strftime("%Y-%m-01"),
                "Region": REGION,
                "Cluster": cluster,
                "Sub_Cluster": subcluster,
                "Customer_Group": cust_group,
                "Brand": sku["Brand"],
                "SKU_ID": sku["SKU_ID"],
                "SKU_Name": sku["SKU_Name"],
                "Forecast_Lag0_Units": int(forecast),
                "Shipped_Units": int(shipped),
            })

df = pd.DataFrame(records)
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(["SKU_ID", "Customer_Group", "Sub_Cluster", "Date"]).reset_index(drop=True)

out_path = "/home/claude/chocolate-bias-app/data/sample_forecast_shipments.csv"
df.to_csv(out_path, index=False)
print(f"Wrote {len(df):,} rows to {out_path}")
print(df.head(10))
