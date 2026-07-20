"""
Chocolate Co. — Forecast Bias Detector
Streamlit app to identify SKU/market combinations with persistent
Lag-0 forecast bias over a user-defined threshold (X%) and duration (Y months).
"""

import io
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

APP_DIR = Path(__file__).parent
SAMPLE_DATA_PATH = APP_DIR / "data" / "sample_forecast_shipments.csv"

st.set_page_config(
    page_title="Forecast Bias Detector | Chocolate Co.",
    page_icon="🍫",
    layout="wide",
)

REQUIRED_COLS = [
    "Date", "Region", "Cluster", "Sub_Cluster", "Customer_Group",
    "Brand", "SKU_ID", "SKU_Name", "Forecast_Lag0_Units", "Shipped_Units",
]

DIM_OPTIONS = ["SKU_ID", "SKU_Name", "Brand", "Customer_Group", "Region", "Cluster", "Sub_Cluster"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@st.cache_data
def load_sample_data():
    if not SAMPLE_DATA_PATH.exists():
        st.error(
            f"Sample data file not found at `{SAMPLE_DATA_PATH}`.\n\n"
            "This usually means the `data/` folder wasn't pushed to your GitHub repo, "
            "or `.gitignore` is excluding it. Run `git status` / `git ls-files` locally "
            "to confirm `data/sample_forecast_shipments.csv` is tracked and committed, "
            "then push again — or switch to 'Upload my own CSV' in the sidebar."
        )
        st.stop()
    return pd.read_csv(SAMPLE_DATA_PATH, parse_dates=["Date"])


def validate_columns(df: pd.DataFrame):
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    return missing


def compute_bias(df: pd.DataFrame, group_cols: list) -> pd.DataFrame:
    """Aggregate to the chosen grain and compute monthly bias %."""
    g = (
        df.groupby(group_cols + ["Date"], as_index=False)
        .agg(Forecast_Lag0_Units=("Forecast_Lag0_Units", "sum"),
             Shipped_Units=("Shipped_Units", "sum"))
    )
    g["Bias_Units"] = g["Forecast_Lag0_Units"] - g["Shipped_Units"]
    g["Bias_Pct"] = np.where(
        g["Shipped_Units"] > 0,
        g["Bias_Units"] / g["Shipped_Units"] * 100,
        np.nan,
    )
    g = g.sort_values(group_cols + ["Date"])
    return g


def flag_streaks(g: pd.DataFrame, group_cols: list, x_threshold: float, y_months: int, direction: str):
    """
    For each group, find the longest run of consecutive months where the bias
    condition (direction-aware) holds, and flag groups whose longest run >= y_months.
    """
    def exceeds(row_bias):
        if direction == "Over-forecast (Forecast > Shipped)":
            return row_bias > x_threshold
        elif direction == "Under-forecast (Forecast < Shipped)":
            return row_bias < -x_threshold
        else:  # Both directions (absolute bias)
            return abs(row_bias) > x_threshold

    results = []
    for keys, sub in g.groupby(group_cols):
        sub = sub.sort_values("Date").reset_index(drop=True)
        cond = sub["Bias_Pct"].apply(exceeds).fillna(False).to_numpy()

        best_len, best_start = 0, None
        cur_len, cur_start = 0, None
        for i, c in enumerate(cond):
            if c:
                if cur_len == 0:
                    cur_start = i
                cur_len += 1
                if cur_len > best_len:
                    best_len, best_start = cur_len, cur_start
            else:
                cur_len = 0

        flagged = best_len >= y_months
        if best_len > 0:
            streak_slice = sub.iloc[best_start: best_start + best_len]
            avg_bias = streak_slice["Bias_Pct"].mean()
            streak_start_date = streak_slice["Date"].min()
            streak_end_date = streak_slice["Date"].max()
        else:
            avg_bias = np.nan
            streak_start_date = pd.NaT
            streak_end_date = pd.NaT

        row = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
        row.update({
            "Max_Consecutive_Months": best_len,
            "Flagged": flagged,
            "Streak_Avg_Bias_Pct": round(avg_bias, 1) if pd.notna(avg_bias) else np.nan,
            "Streak_Start": streak_start_date,
            "Streak_End": streak_end_date,
            "Months_of_Data": len(sub),
            "Overall_Avg_Bias_Pct": round(sub["Bias_Pct"].mean(), 1),
            "Total_Shipped_Units": int(sub["Shipped_Units"].sum()),
            "Total_Forecast_Units": int(sub["Forecast_Lag0_Units"].sum()),
        })
        results.append(row)

    return pd.DataFrame(results)


def bias_color(val):
    if pd.isna(val):
        return ""
    if val > 0:
        return "color: #b23b3b"
    elif val < 0:
        return "color: #2f6f4f"
    return ""


def style_apply_elementwise(styler, func, subset=None):
    """Styler.applymap was removed in newer pandas in favor of Styler.map.
    This helper works across pandas versions."""
    if hasattr(styler, "map"):
        return styler.map(func, subset=subset)
    return styler.applymap(func, subset=subset)


# ---------------------------------------------------------------------------
# Sidebar — data source & parameters
# ---------------------------------------------------------------------------

st.sidebar.title("🍫 Forecast Bias Detector")
st.sidebar.caption("Lag-0 Forecast vs Shipped Units — bias & consistency analysis")

st.sidebar.header("1. Data source")
data_source = st.sidebar.radio("Choose data", ["Use sample dataset", "Upload my own CSV"], index=0)

if data_source == "Upload my own CSV":
    uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if uploaded is not None:
        raw_df = pd.read_csv(uploaded)
        missing = validate_columns(raw_df)
        if missing:
            st.sidebar.error(f"Missing required column(s): {', '.join(missing)}")
            st.stop()
        raw_df["Date"] = pd.to_datetime(raw_df["Date"])
    else:
        st.sidebar.info("Upload a file, or switch back to the sample dataset.")
        st.stop()
else:
    raw_df = load_sample_data()
    with st.sidebar.expander("Expected CSV format"):
        st.code(", ".join(REQUIRED_COLS), language="text")
        st.caption("Date should be month-level (YYYY-MM-01). One row per SKU / market combo / month.")

st.sidebar.header("2. Bias rule")
x_threshold = st.sidebar.slider("Bias threshold — X (%)", min_value=5, max_value=100, value=20, step=1)
y_months = st.sidebar.slider("Consecutive months — Y", min_value=1, max_value=12, value=3, step=1)
direction = st.sidebar.selectbox(
    "Bias direction",
    ["Both directions (absolute bias)", "Over-forecast (Forecast > Shipped)", "Under-forecast (Forecast < Shipped)"],
)

st.sidebar.header("3. Analysis grain")
group_cols = st.sidebar.multiselect(
    "Group by (defines what counts as one 'SKU line')",
    DIM_OPTIONS,
    default=["SKU_ID", "SKU_Name", "Customer_Group"],
)
if not group_cols:
    st.sidebar.warning("Select at least one grouping dimension.")
    st.stop()

st.sidebar.header("4. Filters")
f_brand = st.sidebar.multiselect("Brand", sorted(raw_df["Brand"].unique()))
f_cust = st.sidebar.multiselect("Customer Group", sorted(raw_df["Customer_Group"].unique()))
f_cluster = st.sidebar.multiselect("Cluster", sorted(raw_df["Cluster"].unique()))
f_subcluster = st.sidebar.multiselect("Sub-Cluster", sorted(raw_df["Sub_Cluster"].unique()))

df = raw_df.copy()
if f_brand:
    df = df[df["Brand"].isin(f_brand)]
if f_cust:
    df = df[df["Customer_Group"].isin(f_cust)]
if f_cluster:
    df = df[df["Cluster"].isin(f_cluster)]
if f_subcluster:
    df = df[df["Sub_Cluster"].isin(f_subcluster)]

if df.empty:
    st.warning("No data matches the current filters.")
    st.stop()

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

st.title("Forecast Bias Detector")
st.caption(
    f"Flagging lines where forecast bias exceeds **{x_threshold}%** "
    f"for **{y_months}+ consecutive months** ({direction.split(' (')[0].lower()})."
)

monthly = compute_bias(df, group_cols)
streaks = flag_streaks(monthly, group_cols, x_threshold, y_months, direction)

n_total = len(streaks)
n_flagged = int(streaks["Flagged"].sum())
pct_flagged = (n_flagged / n_total * 100) if n_total else 0

k1, k2, k3, k4 = st.columns(4)
k1.metric("Lines analyzed", f"{n_total:,}")
k2.metric("Flagged lines", f"{n_flagged:,}")
k3.metric("% of portfolio flagged", f"{pct_flagged:.1f}%")
k4.metric("Months in dataset", f"{df['Date'].nunique()}")

st.divider()

tab1, tab2, tab3 = st.tabs(["🚩 Flagged SKUs", "📈 Trend explorer", "📄 Raw / aggregated data"])

# --- Tab 1: Flagged table -----------------------------------------------
with tab1:
    st.subheader("Flagged lines")
    flagged_df = streaks[streaks["Flagged"]].sort_values(
        "Max_Consecutive_Months", ascending=False
    )

    if flagged_df.empty:
        st.success("No lines meet the flagging criteria at the current threshold. Try lowering X or Y in the sidebar.")
    else:
        display_cols = group_cols + [
            "Max_Consecutive_Months", "Streak_Avg_Bias_Pct", "Streak_Start", "Streak_End",
            "Overall_Avg_Bias_Pct", "Total_Shipped_Units", "Total_Forecast_Units",
        ]
        styled = style_apply_elementwise(
            flagged_df[display_cols].style, bias_color,
            subset=["Streak_Avg_Bias_Pct", "Overall_Avg_Bias_Pct"]
        ).format({
            "Streak_Avg_Bias_Pct": "{:+.1f}%",
            "Overall_Avg_Bias_Pct": "{:+.1f}%",
            "Total_Shipped_Units": "{:,.0f}",
            "Total_Forecast_Units": "{:,.0f}",
            "Streak_Start": lambda d: d.strftime("%b %Y") if pd.notna(d) else "",
            "Streak_End": lambda d: d.strftime("%b %Y") if pd.notna(d) else "",
        })
        st.dataframe(styled, use_container_width=True, height=420)

        csv_buf = io.StringIO()
        flagged_df[display_cols].to_csv(csv_buf, index=False)
        st.download_button(
            "⬇ Download flagged lines (CSV)",
            data=csv_buf.getvalue(),
            file_name=f"flagged_bias_X{x_threshold}pct_Y{y_months}mo.csv",
            mime="text/csv",
        )

        # Distribution chart
        st.markdown("##### Where is the bias concentrated?")
        dim_for_chart = st.selectbox(
            "Break down flagged lines by", [c for c in group_cols if c != "SKU_ID"] or group_cols, key="chart_dim"
        )
        chart_data = flagged_df.groupby(dim_for_chart).size().reset_index(name="Flagged_Count")
        fig = px.bar(
            chart_data.sort_values("Flagged_Count", ascending=True),
            x="Flagged_Count", y=dim_for_chart, orientation="h",
            title=None, color="Flagged_Count", color_continuous_scale="Reds",
        )
        fig.update_layout(showlegend=False, height=400, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("See all analyzed lines (flagged + not flagged)"):
        all_display_cols = group_cols + [
            "Max_Consecutive_Months", "Flagged", "Overall_Avg_Bias_Pct",
            "Total_Shipped_Units", "Total_Forecast_Units",
        ]
        st.dataframe(streaks[all_display_cols].sort_values("Max_Consecutive_Months", ascending=False),
                     use_container_width=True, height=300)

# --- Tab 2: Trend explorer ------------------------------------------------
with tab2:
    st.subheader("Forecast vs Shipped trend explorer")

    monthly["_label"] = monthly[group_cols].astype(str).agg(" | ".join, axis=1)
    options = sorted(monthly["_label"].unique())

    default_pick = []
    if not streaks[streaks["Flagged"]].empty:
        top = streaks[streaks["Flagged"]].sort_values("Max_Consecutive_Months", ascending=False).iloc[0]
        default_label = " | ".join(str(top[c]) for c in group_cols)
        if default_label in options:
            default_pick = [default_label]

    picks = st.multiselect("Select line(s) to plot", options, default=default_pick[:1] if default_pick else options[:1])

    if picks:
        plot_df = monthly[monthly["_label"].isin(picks)]

        for label in picks:
            sub = plot_df[plot_df["_label"] == label].sort_values("Date")
            fig = go.Figure()
            fig.add_trace(go.Bar(x=sub["Date"], y=sub["Shipped_Units"], name="Shipped", marker_color="#5b8c5a"))
            fig.add_trace(go.Scatter(x=sub["Date"], y=sub["Forecast_Lag0_Units"], name="Forecast (Lag-0)",
                                      mode="lines+markers", line=dict(color="#c96a4e", width=3)))
            fig.update_layout(title=label, height=350, legend=dict(orientation="h", y=1.15),
                               margin=dict(t=60, b=20))
            st.plotly_chart(fig, use_container_width=True)

            fig2 = go.Figure()
            fig2.add_trace(go.Bar(x=sub["Date"], y=sub["Bias_Pct"], name="Bias %",
                                   marker_color=np.where(sub["Bias_Pct"] > 0, "#b23b3b", "#2f6f4f")))
            fig2.add_hline(y=x_threshold, line_dash="dash", line_color="gray", annotation_text=f"+{x_threshold}%")
            fig2.add_hline(y=-x_threshold, line_dash="dash", line_color="gray", annotation_text=f"-{x_threshold}%")
            fig2.update_layout(title=f"Monthly Bias % — {label}", height=280, margin=dict(t=40, b=20))
            st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Select at least one line above to see its trend.")

# --- Tab 3: Raw / aggregated data -----------------------------------------
with tab3:
    st.subheader("Aggregated monthly data (at selected grain)")
    st.dataframe(
        monthly.drop(columns="_label", errors="ignore").sort_values(group_cols + ["Date"]),
        use_container_width=True, height=400,
    )
    csv_buf2 = io.StringIO()
    monthly.drop(columns="_label", errors="ignore").to_csv(csv_buf2, index=False)
    st.download_button("⬇ Download aggregated data (CSV)", data=csv_buf2.getvalue(),
                        file_name="aggregated_forecast_bias.csv", mime="text/csv")

    st.subheader("Raw uploaded / sample data (filtered)")
    st.dataframe(df, use_container_width=True, height=300)

st.divider()
st.caption(
    "Bias % = (Forecast Lag-0 − Shipped Units) / Shipped Units × 100. "
    "Positive = over-forecast, negative = under-forecast. "
    "Built for chocolate demand planning teams — adjust X (threshold) and Y (months) in the sidebar."
)
