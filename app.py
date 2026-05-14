# app.py
# ArchGrade Studio V2
# Faculty grading exploration app: Z-score, Quantile-Range, and K-means
# Run:
#   pip install streamlit pandas numpy scipy scikit-learn plotly openpyxl
#   streamlit run app.py

import io
import math
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

from scipy.stats import zscore
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Noto+Sans+Thai:wght@300;400;500;700&display=swap" rel="stylesheet">

<style>
html, body, [class*="css"] {
    font-family: 'Noto Sans Thai', sans-serif;
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="ArchGrade",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 ArchGrade — Professor Visual Review")
st.caption("ปรับเกณฑ์ → ดูกราฟทันที → ตรวจตารางด้านล่าง")

st.markdown(
    """
    <style>
    /* =====================================================
       ArchGrade Studio V3 — Minimal Faculty UI
       Text boxes use a soft dark card. Graphs use a separate
       dark Plotly theme, so there is no white-on-white problem.
       ===================================================== */

    :root {
        --ag-page: #0b1020;
        --ag-panel: #111827;
        --ag-panel-2: #162033;
        --ag-panel-3: #0f172a;
        --ag-border: rgba(148, 163, 184, 0.28);
        --ag-border-strong: rgba(148, 163, 184, 0.45);
        --ag-text: #f8fafc;
        --ag-muted: #cbd5e1;
        --ag-soft: #94a3b8;
        --ag-blue: #60a5fa;
        --ag-purple: #a78bfa;
        --ag-green: #4ade80;
        --ag-yellow: #facc15;
        --ag-red: #f87171;
    }

    .main .block-container {
        padding-top: 1.6rem;
        padding-bottom: 3rem;
        max-width: 1500px;
    }

    h1, h2, h3, h4, h5, h6, p, label, span, div {
        letter-spacing: 0.01em;
    }

    /* KPI cards: minimal, dark, readable */
    div[data-testid="stMetric"] {
        background: linear-gradient(180deg, rgba(17,24,39,0.98), rgba(15,23,42,0.98));
        border: 1px solid var(--ag-border);
        padding: 15px 17px;
        border-radius: 16px;
        box-shadow: 0 8px 22px rgba(0,0,0,0.22);
    }
    div[data-testid="stMetric"] label,
    div[data-testid="stMetric"] [data-testid="stMetricLabel"],
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: var(--ag-text) !important;
    }
    div[data-testid="stMetric"] [data-testid="stMetricLabel"] p {
        color: var(--ag-muted) !important;
        font-size: 0.92rem;
    }

    /* Thai explanation card: separate visual language from graph */
    .thai-help-box {
        background: linear-gradient(135deg, rgba(30,41,59,0.96), rgba(49,46,129,0.70));
        color: var(--ag-text) !important;
        border: 1px solid rgba(129,140,248,0.36);
        border-left: 6px solid var(--ag-purple);
        padding: 1.00rem 1.20rem;
        border-radius: 18px;
        margin-bottom: 1.15rem;
        line-height: 1.75;
        box-shadow: 0 10px 28px rgba(0,0,0,0.22);
        font-size: 1.00rem;
    }
    .thai-help-box b {
        color: #ddd6fe !important;
        font-size: 1.06rem;
    }

    /* Small note cards: readable, not white */
    .review-note {
        background: linear-gradient(135deg, rgba(15,23,42,0.98), rgba(30,41,59,0.90));
        color: var(--ag-text) !important;
        border: 1px solid var(--ag-border);
        border-left: 5px solid var(--ag-blue);
        padding: 0.90rem 1.05rem;
        border-radius: 15px;
        line-height: 1.7;
        margin: 0.65rem 0 1rem 0;
        box-shadow: 0 8px 22px rgba(0,0,0,0.16);
    }
    .review-note b {
        color: #bfdbfe !important;
    }
    .visual-priority-note {
        background: linear-gradient(135deg, rgba(88,28,135,0.34), rgba(15,23,42,0.88));
        color: var(--ag-text) !important;
        border: 1px solid rgba(167,139,250,0.30);
        border-left: 5px solid var(--ag-purple);
        padding: 0.70rem 0.95rem;
        border-radius: 14px;
        line-height: 1.60;
        margin: 0.35rem 0 0.85rem 0;
        font-size: 0.96rem;
    }
    .thai-help-box *, .review-note *, .visual-priority-note * {
        color: inherit !important;
    }

    /* Streamlit containers */
    div[data-testid="stExpander"] {
        border-radius: 14px;
        border-color: var(--ag-border) !important;
    }
    button[data-baseweb="tab"] p {
        font-size: 1.00rem;
        font-weight: 650;
    }

    /* Inputs: slightly softer */
    div[data-baseweb="select"] > div,
    div[data-testid="stNumberInput"] input,
    div[data-testid="stTextInput"] input {
        border-radius: 10px !important;
    }

    /* Avoid overly bright dataframes in dark mode */
    div[data-testid="stDataFrame"] {
        border-radius: 14px;
        overflow: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="thai-help-box">
    <b>แนวคิดของแอปนี้</b><br>
    แอปนี้ไม่ได้ตัดสินเกรดแทนอาจารย์ แต่ช่วยให้อาจารย์เห็นภาพรวมของคะแนน การกระจายของเกรด
    และผลของการปรับเกณฑ์แบบทันที เหมาะสำหรับใช้ในการประชุมพิจารณาเกรด ตรวจสอบความสมเหตุสมผล
    และอธิบายเหตุผลของการปรับเกณฑ์อย่างโปร่งใส<br>
    <b>ลำดับการใช้งานที่แนะนำ</b>: ปรับเกณฑ์ → ดูกราฟจำนวนเกรดและ Ranked Score → ตรวจตารางรายชื่อด้านล่าง
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# CONSTANTS
# =========================================================

DEFAULT_GRADES_8 = ["A", "B+", "B", "C+", "C", "D+", "D", "F"]
DEFAULT_GRADES_7 = ["A", "B+", "B", "C+", "C", "D+", "D"]
DEFAULT_GRADES_6 = ["A", "B+", "B", "C+", "C", "D+"]
DEFAULT_GRADES_5 = ["A", "B+", "B", "C+", "C"]
DEFAULT_GRADES_4 = ["A", "B+", "B", "C+"]

GRADE_POINTS = {
    "A": 4.0,
    "B+": 3.5,
    "B": 3.0,
    "C+": 2.5,
    "C": 2.0,
    "D+": 1.5,
    "D": 1.0,
    "F": 0.0,
    "-": np.nan,
}

GRADE_ORDER = {g: i for i, g in enumerate(DEFAULT_GRADES_8)}

GRADE_COLORS = {
    "A": "#ef4444",
    "B+": "#f97316",
    "B": "#f59e0b",
    "C+": "#eab308",
    "C": "#22c55e",
    "D+": "#06b6d4",
    "D": "#6366f1",
    "F": "#a855f7",
    "-": "#94a3b8",
}
OUTLIER_COLORS = {"Normal": "#60a5fa", "Outlier": "#f87171"}


def apply_chart_theme(fig, height=None):
    """Minimal dark chart theme, separated from text-box styling."""
    layout_kwargs = dict(
        template="plotly_dark",
        paper_bgcolor="rgba(15,23,42,0.00)",
        plot_bgcolor="rgba(15,23,42,0.35)",
        font=dict(color="#e5e7eb", size=13),
        title_font=dict(color="#f8fafc", size=18),
        legend=dict(
            font=dict(color="#e5e7eb"),
            bgcolor="rgba(15,23,42,0.35)",
            bordercolor="rgba(148,163,184,0.20)",
            borderwidth=1,
        ),
        margin=dict(t=68, b=46, l=54, r=28),
        xaxis=dict(
            title_font=dict(color="#e5e7eb"),
            tickfont=dict(color="#cbd5e1"),
            gridcolor="rgba(148,163,184,0.16)",
            zerolinecolor="rgba(148,163,184,0.30)",
            linecolor="rgba(148,163,184,0.30)",
        ),
        yaxis=dict(
            title_font=dict(color="#e5e7eb"),
            tickfont=dict(color="#cbd5e1"),
            gridcolor="rgba(148,163,184,0.16)",
            zerolinecolor="rgba(148,163,184,0.30)",
            linecolor="rgba(148,163,184,0.30)",
        ),
    )
    if height is not None:
        layout_kwargs["height"] = height
    fig.update_layout(**layout_kwargs)
    fig.update_annotations(font=dict(color="#e5e7eb", size=12), bgcolor="rgba(15,23,42,0.65)", bordercolor="rgba(148,163,184,0.25)")
    return fig


# =========================================================
# GENERAL HELPERS
# =========================================================

def read_uploaded_file(uploaded_file):
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file, encoding="utf-8-sig")
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(uploaded_file)
    raise ValueError("Please upload CSV or Excel file.")


def clean_dataframe(raw_df, id_col, score_col):
    df = raw_df[[id_col, score_col]].copy()
    df.columns = ["student_id", "raw_score"]
    df["student_id"] = df["student_id"].astype(str).str.strip()
    df["raw_score"] = pd.to_numeric(df["raw_score"], errors="coerce")
    df = df.dropna(subset=["student_id", "raw_score"]).copy()
    df = df.reset_index(drop=True)
    return df


def normalize_scores(df, mode, raw_min=None, raw_max=None):
    df = df.copy()

    if mode == "Score is already 0–4":
        df["score_4"] = df["raw_score"].clip(0, 4)
        return df

    if mode == "Normalize by observed min/max to 0–4":
        smin = df["raw_score"].min()
        smax = df["raw_score"].max()
    else:
        smin = raw_min
        smax = raw_max

    if smax is None or smin is None or smax == smin:
        df["score_4"] = 0.0
    else:
        df["score_4"] = (df["raw_score"] - smin) / (smax - smin) * 4
        df["score_4"] = df["score_4"].clip(0, 4)

    return df


def detect_outliers(df, score_col="score_4"):
    out = df.copy()
    scores = out[score_col]

    if len(out) < 2 or scores.std(ddof=0) == 0:
        out["z_for_outlier"] = 0.0
        out["z_outlier"] = False
        out["iqr_outlier"] = False
        out["outlier_any"] = False
        info = {
            "Q1": scores.quantile(0.25),
            "Median": scores.quantile(0.50),
            "Q3": scores.quantile(0.75),
            "IQR": 0,
            "IQR lower bound": np.nan,
            "IQR upper bound": np.nan,
            "Z threshold": "|z| > 3",
        }
        return out, info

    out["z_for_outlier"] = zscore(scores)
    out["z_outlier"] = out["z_for_outlier"].abs() > 3

    q1 = scores.quantile(0.25)
    q2 = scores.quantile(0.50)
    q3 = scores.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    out["iqr_outlier"] = (scores < lower) | (scores > upper)
    out["outlier_any"] = out["z_outlier"] | out["iqr_outlier"]

    info = {
        "Q1": q1,
        "Median": q2,
        "Q3": q3,
        "IQR": iqr,
        "IQR lower bound": lower,
        "IQR upper bound": upper,
        "Z threshold": "|z| > 3",
    }
    return out, info


def get_calc_df(df, outlier_policy, manual_exclude_ids):
    if outlier_policy == "Use all students":
        return df.copy()

    if outlier_policy == "Exclude detected outliers from calculation only":
        return df[~df["outlier_any"]].copy()

    if outlier_policy == "Manually exclude selected students from calculation only":
        return df[~df["student_id"].isin(manual_exclude_ids)].copy()

    if outlier_policy == "Exclude detected + manually selected from calculation":
        return df[(~df["outlier_any"]) & (~df["student_id"].isin(manual_exclude_ids))].copy()

    return df.copy()


def add_rank(df, score_col="score_4"):
    out = df.copy()
    out["rank"] = out[score_col].rank(method="min", ascending=False).astype(int)
    return out


def grade_distribution(df, grade_col):
    if grade_col not in df.columns:
        return pd.DataFrame(columns=["grade", "count", "percent"])
    dist = df[grade_col].value_counts(dropna=False).rename_axis("grade").reset_index(name="count")
    dist["percent"] = dist["count"] / dist["count"].sum() * 100
    return dist


def grade_distribution_ordered(df, grade_col):
    """Return grade distribution in official grade order for stable visual comparison."""
    dist = grade_distribution(df, grade_col)
    if dist.empty:
        return dist
    dist["_order"] = dist["grade"].map(GRADE_ORDER).fillna(99)
    dist = dist.sort_values("_order").drop(columns="_order").reset_index(drop=True)
    return dist


def grade_distribution_plot(df, grade_col, title="Grade distribution"):
    """Dynamic bar chart: number and percentage of students in each grade."""
    dist = grade_distribution_ordered(df, grade_col)
    if dist.empty:
        fig = go.Figure()
        fig.update_layout(title=title, height=360)
        return fig

    dist["label"] = dist.apply(lambda r: f"{int(r['count'])} คน<br>{r['percent']:.1f}%", axis=1)
    fig = px.bar(
        dist,
        x="grade",
        y="count",
        text="label",
        title=title,
        category_orders={"grade": DEFAULT_GRADES_8},
        color="grade",
        color_discrete_map=GRADE_COLORS,
        hover_data={"grade": True, "count": True, "percent": ":.1f", "label": False},
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        font=dict( family="Tahoma", size=18, color="white" ),
        height=390,
        yaxis_title="Number of students / จำนวนนิสิต",
        xaxis_title="Grade / เกรด",
        bargap=0.25,
        margin=dict(t=70, b=40, l=40, r=30),
    )
    return apply_chart_theme(fig)


def grade_distribution_donut(df, grade_col, title="Grade ratio"):
    """Compact ratio chart for review meetings."""
    dist = grade_distribution_ordered(df, grade_col)
    if dist.empty:
        fig = go.Figure()
        fig.update_layout(title=title, height=350)
        return fig
    fig = px.pie(
        dist,
        names="grade",
        values="count",
        hole=0.45,
        title=title,
        category_orders={"grade": DEFAULT_GRADES_8},
        color_discrete_map=GRADE_COLORS,
    )
    fig.update_traces(textinfo="label+percent")
    fig.update_layout(height=390, margin=dict(t=70, b=30, l=30, r=30))
    return apply_chart_theme(fig)


def grade_review_metrics(df, grade_col):
    """Small KPI table for faculty review."""
    if grade_col not in df.columns:
        return {"A": 0, "B+ or above": 0, "C or below": 0, "F": 0, "Average GP": np.nan}
    gp = df[grade_col].map(GRADE_POINTS)
    return {
        "A": int((df[grade_col] == "A").sum()),
        "B+ or above": int(df[grade_col].isin(["A", "B+"]).sum()),
        "C or below": int(df[grade_col].isin(["C", "D+", "D", "F"]).sum()),
        "F": int((df[grade_col] == "F").sum()),
        "Average GP": gp.dropna().mean(),
    }


def show_grade_review_panel(df, grade_col, title_prefix=""):
    """Reusable professor-facing visual panel shown after every adjustment.

    UX logic: graphs come first because professors decide visually;
    tables come later for checking specific students.
    """
    metrics = grade_review_metrics(df, grade_col)
    st.markdown(
        """
        <div class="visual-priority-note">
        <b>Visual feedback หลังปรับเกณฑ์</b>: ดูกราฟจำนวนเกรดและเส้นอันดับก่อน เพื่อประเมินว่า A มาก/น้อยเกินไปหรือไม่
        และมีเกรดต่ำกระจุกตัวผิดปกติหรือไม่ จากนั้นค่อยตรวจตารางรายชื่อด้านล่าง
        </div>
        """,
        unsafe_allow_html=True,
    )

    v1, v2 = st.columns([1.10, 1.15])
    with v1:
        st.plotly_chart(
            grade_distribution_plot(df, grade_col, title=f"{title_prefix}Grade distribution"),
            use_container_width=True,
        )
    with v2:
        st.plotly_chart(
            ranked_score_plot(df, grade_col=grade_col, title=f"{title_prefix}Ranked score"),
            use_container_width=True,
        )

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("A", metrics["A"])
    m2.metric("A/B+", metrics["B+ or above"])
    m3.metric("C or below", metrics["C or below"])
    m4.metric("F", metrics["F"])
    m5.metric("Avg GP", f"{metrics['Average GP']:.2f}" if not np.isnan(metrics["Average GP"]) else "-")


def show_supporting_visuals(df, grade_col, title_prefix="", main_fig=None):
    """Secondary visuals below the main feedback. Keeps tables lowest."""
    s1, s2 = st.columns([1.20, 0.95])
    with s1:
        if main_fig is not None:
            st.plotly_chart(main_fig, use_container_width=True)
        else:
            st.plotly_chart(
                distribution_plot(df, title=f"{title_prefix}Score histogram"),
                use_container_width=True,
            )
    with s2:
        st.plotly_chart(
            grade_distribution_donut(df, grade_col, title=f"{title_prefix}Grade ratio"),
            use_container_width=True,
        )


def show_bottom_check_tables(full_df, grade_col, summary_df=None, summary_title="Summary table", table_title="Student checking table"):
    """Tables are intentionally placed at the bottom for verification."""
    st.markdown("---")
    st.subheader("ตารางตรวจสอบด้านล่าง / Bottom checking tables")
    if summary_df is not None:
        with st.expander(summary_title, expanded=False):
            st.dataframe(summary_df, use_container_width=True, height=320)
    with st.expander(table_title, expanded=True):
        st.dataframe(add_rank(full_df).sort_values("score_4", ascending=False), use_container_width=True, height=560)



def to_csv_bytes(df):
    return df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")


def sort_dataframe(df, sort_mode, grade_col=None):
    out = df.copy()
    if sort_mode == "Student ID":
        return out.sort_values("student_id", ascending=True)
    if sort_mode == "Score descending":
        return out.sort_values("score_4", ascending=False)
    if sort_mode == "Score ascending":
        return out.sort_values("score_4", ascending=True)
    if sort_mode == "Grade" and grade_col is not None and grade_col in out.columns:
        out["_grade_order"] = out[grade_col].map(GRADE_ORDER).fillna(99)
        out = out.sort_values(["_grade_order", "score_4"], ascending=[True, False])
        return out.drop(columns=["_grade_order"])
    return out


def assign_by_threshold(value, thresholds):
    # thresholds: [(grade, cutoff), ...] sorted high-to-low
    for grade, cutoff in thresholds:
        if value >= cutoff:
            return grade
    return thresholds[-1][0]


def excel_geomean(series):
    """Excel GEOMEAN-style central score. GEOMEAN uses positive values only."""
    s = pd.to_numeric(series, errors="coerce").dropna()
    s = s[s > 0]
    if len(s) == 0:
        return np.nan
    return float(np.exp(np.mean(np.log(s))))


def excel_quartile_exc(series, quart):
    """Excel QUARTILE.EXC(array, quart) equivalent for quart = 1, 2, 3."""
    s = pd.to_numeric(series, errors="coerce").dropna().sort_values().to_numpy(dtype=float)
    n = len(s)
    if n == 0:
        return np.nan
    p = quart / 4.0
    pos = p * (n + 1)
    if pos < 1 or pos > n:
        return np.nan
    lo = int(np.floor(pos))
    hi = int(np.ceil(pos))
    if lo == hi:
        return float(s[lo - 1])
    lo_val = s[lo - 1]
    hi_val = s[hi - 1]
    frac = pos - lo
    return float(lo_val + frac * (hi_val - lo_val))


# =========================================================
# VISUALIZATION HELPERS
# =========================================================

def distribution_plot(df, score_col="score_4", title="Score distribution", boundaries=None, outlier_col="outlier_any"):
    plot_df = df.copy()
    plot_df["Outlier"] = np.where(plot_df[outlier_col], "Outlier", "Normal")

    fig = px.histogram(
        plot_df,
        x=score_col,
        color="Outlier",
        color_discrete_map=OUTLIER_COLORS,
        nbins=20,
        # marginal box removed to avoid crowded labels in dark UI
        hover_data=["student_id", "raw_score", "score_4"],
        title=title,
    )

    mean = round_for_cell_assignment(excel_geomean(plot_df[score_col]), 3)
    std = plot_df[score_col].std(ddof=0)
    q1 = excel_quartile_exc(plot_df[score_col], 1)
    q2 = excel_quartile_exc(plot_df[score_col], 2)
    q3 = excel_quartile_exc(plot_df[score_col], 3)

    fig.add_vline(x=mean, line_dash="solid", line_color="#facc15", annotation_text="Central (GEOMEAN)", annotation_position="top")
    fig.add_vline(x=q1, line_dash="dot", line_color="#94a3b8", annotation_text="Q1", annotation_position="top left")
    fig.add_vline(x=q2, line_dash="dot", line_color="#94a3b8", annotation_text="Q2", annotation_position="top")
    fig.add_vline(x=q3, line_dash="dot", line_color="#94a3b8", annotation_text="Q3", annotation_position="top right")

    if std > 0:
        fig.add_vline(x=mean + std, line_dash="dash", line_color="#38bdf8", annotation_text="+1SD", annotation_position="bottom right")
        fig.add_vline(x=mean - std, line_dash="dash", line_color="#38bdf8", annotation_text="-1SD", annotation_position="bottom left")

    if boundaries:
        for label, x in boundaries:
            if 0 <= x <= 4:
                fig.add_vline(x=x, line_dash="dash", line_color="#818cf8", annotation_text=label, annotation_position="top right")

    fig.update_layout(height=480, bargap=0.05)
    return apply_chart_theme(fig)


def ranked_score_plot(df, score_col="score_4", grade_col=None, title="Ranked scores"):
    plot_df = add_rank(df, score_col=score_col).sort_values(score_col, ascending=False)
    color = grade_col if grade_col and grade_col in plot_df.columns else None

    fig = px.scatter(
        plot_df,
        x="rank",
        y=score_col,
        color=color,
        hover_data=["student_id", "raw_score", "score_4"],
        title=title,
    )
    fig.update_traces(marker=dict(size=9))
    fig.update_layout(height=480)
    return apply_chart_theme(fig)


def quantile_box_plot(df):
    fig = px.box(
        df,
        y="score_4",
        points="all",
        hover_data=["student_id", "raw_score"],
        title="Boxplot with all students and outliers",
    )
    fig.update_layout(height=480)
    return apply_chart_theme(fig)


# =========================================================
# Z-SCORE MODE
# =========================================================

def zscore_grading(df_all, df_calc, scheme_name, z_thresholds):
    out = df_all.copy()
    mean = df_calc["score_4"].mean()
    std = df_calc["score_4"].std(ddof=0)

    if std == 0 or np.isnan(std):
        out[f"z_{scheme_name}"] = 0.0
    else:
        out[f"z_{scheme_name}"] = (out["score_4"] - mean) / std

    out[f"t_{scheme_name}"] = 50 + 10 * out[f"z_{scheme_name}"]
    out[f"grade_z_{scheme_name}"] = out[f"z_{scheme_name}"].apply(lambda x: assign_by_threshold(x, z_thresholds))

    return out[["student_id", f"z_{scheme_name}", f"t_{scheme_name}", f"grade_z_{scheme_name}"]]


def z_boundaries_to_score(df_calc, z_thresholds):
    mean = df_calc["score_4"].mean()
    std = df_calc["score_4"].std(ddof=0)
    if std == 0 or np.isnan(std):
        return []
    boundaries = []
    for grade, zcut in z_thresholds:
        if zcut > -100:
            boundaries.append((grade, mean + zcut * std))
    return boundaries


# =========================================================
# SIMPLE RANGE MODE
# =========================================================

def simple_range_grading(df_all, scheme_name, thresholds):
    out = df_all.copy()
    out[f"grade_range_{scheme_name}"] = out["score_4"].apply(lambda x: assign_by_threshold(x, thresholds))
    return out[["student_id", f"grade_range_{scheme_name}"]]


# =========================================================
# QUANTILE-CELL RANGE MODE
# =========================================================

def make_score_cells(cell_width=0.025, min_score=0.0, max_score=4.0):
    """
    Create descending score cells using full float64 precision.

    Important: do NOT round upper/lower values during calculation.
    Rounding is only used when displaying values in the UI. This prevents
    students exactly on a boundary from being moved to the wrong cell.
    """
    cell_width = float(cell_width)
    min_score = float(min_score)
    max_score = float(max_score)

    if cell_width <= 0 or max_score <= min_score:
        return pd.DataFrame(columns=["cell_id", "upper", "lower", "mid"])

    eps = 1e-12
    span = max_score - min_score
    n_steps = int(np.floor(span / cell_width + eps))

    edges = [max_score - i * cell_width for i in range(n_steps + 1)]

    # If the last full-width edge has not reached the minimum, append the exact minimum.
    # If it is only numerically close to the minimum, force it to the exact minimum.
    if edges[-1] > min_score + eps:
        edges.append(min_score)
    else:
        edges[-1] = min_score

    rows = []
    for i in range(len(edges) - 1):
        upper = float(edges[i])
        lower = float(edges[i + 1])
        rows.append({"cell_id": i + 1, "upper": upper, "lower": lower, "mid": (upper + lower) / 2.0})
    return pd.DataFrame(rows)


def round_for_cell_assignment(x, digits=3):
    """Round only for assigning scores to cells; do not use for GEOMEAN/quartiles."""
    return float(np.round(float(x), digits))


def locate_cell_index_for_assignment(cells, value, digits=3):
    """
    Locate the central-score cell.

    The central score itself may be rounded to 3 decimals to match the Excel
    anchor shown in the sheet. Cell boundaries remain the actual grid values.
    Boundary rule follows Excel COUNTIFS: value == boundary goes to the lower cell.
    """
    if len(cells) == 0:
        return None

    v = round_for_cell_assignment(value, digits)
    cells_reset = cells.reset_index(drop=True)

    for idx, row in cells_reset.iterrows():
        upper = float(row["upper"])
        lower = float(row["lower"])
        is_bottom = idx == len(cells_reset) - 1

        if is_bottom:
            ok = (v <= upper) and (v >= lower)
        else:
            ok = (v <= upper) and (v > lower)

        if ok:
            return int(idx)

    mids = cells_reset["mid"].astype(float)
    return int((mids - v).abs().idxmin())


def count_students_in_cells(df, cells, score_col="score_4"):
    """
    Count students in cells using Excel COUNTIFS-style logic.

    This matches the worksheet formula:
        COUNTIFS(score_range, "<=" & upper, score_range, ">" & lower)

    Important:
    - Raw scores are NOT rounded before counting.
    - Cell boundaries are generated by the 0.025 grid.
    - A score exactly equal to a boundary belongs to the LOWER cell,
      because the upper condition of the lower cell is <= boundary,
      while the upper cell uses > lower.
    """
    cells = cells.copy()
    scores = pd.to_numeric(df[score_col], errors="coerce")
    counts = []
    ids_in_cell = []

    cells_reset = cells.reset_index(drop=True)
    for idx, row in cells_reset.iterrows():
        upper = float(row["upper"])
        lower = float(row["lower"])
        is_bottom = idx == len(cells_reset) - 1

        if is_bottom:
            # Last visible cell includes the lower end of the selected grid.
            mask = (scores <= upper) & (scores >= lower)
        else:
            # Excel COUNTIFS behavior: <= upper and > lower.
            mask = (scores <= upper) & (scores > lower)

        sub = df[mask]
        counts.append(len(sub))
        ids_in_cell.append(", ".join(sub["student_id"].astype(str).tolist()))

    cells["count"] = counts
    cells["student_ids"] = ids_in_cell
    return cells


def build_mean_centered_cell_grades(cells, mean_score, grade_labels, cell_counts_by_grade):
    """
    Assign grade bands around/above/below mean by number of cells.

    The idea follows the Excel-style workflow:
    - Create many score cells.
    - Locate the cell containing mean.
    - Let professor assign how many cells each grade occupies.
    - By default, grades are assigned from top to bottom.

    cell_counts_by_grade is a dict, e.g.
    {"A": 11, "B+": 26, "B": 11, "C+": 10, ...}
    """
    out = cells.copy()
    out["assigned_grade"] = ""

    # Find mean cell.
    mean_idx_tmp = locate_cell_index_for_assignment(out, mean_score)
    mean_cell_id = int(mean_idx_tmp + 1) if mean_idx_tmp is not None else None

    # Simple and robust version: assign from high score down using cell counts.
    start_idx = 0
    for grade in grade_labels:
        n = int(cell_counts_by_grade.get(grade, 0))
        if n <= 0:
            continue
        end_idx = min(start_idx + n, len(out))
        out.loc[start_idx:end_idx - 1, "assigned_grade"] = grade
        start_idx = end_idx

    # Remaining lower cells get the last grade.
    if len(grade_labels) > 0 and start_idx < len(out):
        out.loc[start_idx:, "assigned_grade"] = grade_labels[-1]

    return out, mean_cell_id



def build_mean_anchored_cell_grades(cells, mean_score, grade_labels, cell_counts_by_grade):
    """
    Assign grade bands outward from the mean cell.

    Odd number of grades:
    - The middle grade is centered on the cell containing the mean.
    - Higher grades expand upward from that middle band.
    - Lower grades expand downward from that middle band.

    Even number of grades:
    - The boundary between the two middle grades is placed closest to the mean.
    - The upper-middle grade expands upward from the mean boundary.
    - The lower-middle grade expands downward from the mean boundary.
    """
    out = cells.copy().reset_index(drop=True)
    out["assigned_grade"] = ""
    n_cells = len(out)
    n_grades = len(grade_labels)

    if n_cells == 0 or n_grades == 0:
        return out, None

    mean_idx = locate_cell_index_for_assignment(out, mean_score)
    if mean_idx is None:
        mean_idx = 0

    def count_for(g):
        return max(1, int(cell_counts_by_grade.get(g, 1)))

    if n_grades % 2 == 1:
        mid_gi = n_grades // 2
        mid_grade = grade_labels[mid_gi]
        mid_count = count_for(mid_grade)

        cells_above = mid_count // 2
        cells_below = mid_count - cells_above - 1
        start = max(0, mean_idx - cells_above)
        end = min(n_cells - 1, mean_idx + cells_below)

        # If clipping makes the middle band too short, extend where space remains.
        while (end - start + 1) < mid_count and (start > 0 or end < n_cells - 1):
            if start > 0:
                start -= 1
            if (end - start + 1) >= mid_count:
                break
            if end < n_cells - 1:
                end += 1

        out.loc[start:end, "assigned_grade"] = mid_grade

        upper_cursor = start - 1
        for gi in range(mid_gi - 1, -1, -1):
            grade = grade_labels[gi]
            n = count_for(grade)
            s = max(0, upper_cursor - n + 1)
            if upper_cursor >= 0:
                out.loc[s:upper_cursor, "assigned_grade"] = grade
            upper_cursor = s - 1

        lower_cursor = end + 1
        for gi in range(mid_gi + 1, n_grades):
            grade = grade_labels[gi]
            n = count_for(grade)
            e = min(n_cells - 1, lower_cursor + n - 1)
            if lower_cursor < n_cells:
                out.loc[lower_cursor:e, "assigned_grade"] = grade
            lower_cursor = e + 1

    else:
        lower_mid_gi = n_grades // 2
        upper_mid_gi = lower_mid_gi - 1

        # Boundary closest to mean: cells with midpoint >= mean stay above, others below.
        below = out.index[out["mid"] < mean_score].tolist()
        boundary_idx = below[0] if below else n_cells

        upper_cursor = boundary_idx - 1
        for gi in range(upper_mid_gi, -1, -1):
            grade = grade_labels[gi]
            n = count_for(grade)
            s = max(0, upper_cursor - n + 1)
            if upper_cursor >= 0:
                out.loc[s:upper_cursor, "assigned_grade"] = grade
            upper_cursor = s - 1

        lower_cursor = boundary_idx
        for gi in range(lower_mid_gi, n_grades):
            grade = grade_labels[gi]
            n = count_for(grade)
            e = min(n_cells - 1, lower_cursor + n - 1)
            if lower_cursor < n_cells:
                out.loc[lower_cursor:e, "assigned_grade"] = grade
            lower_cursor = e + 1

    # Fill uncovered extremes with the nearest extreme grade.
    if (out["assigned_grade"] == "").any():
        first_assigned = out.index[out["assigned_grade"] != ""].min()
        last_assigned = out.index[out["assigned_grade"] != ""].max()
        if pd.isna(first_assigned):
            out["assigned_grade"] = grade_labels[-1]
        else:
            out.loc[:first_assigned - 1, "assigned_grade"] = grade_labels[0]
            out.loc[last_assigned + 1:, "assigned_grade"] = grade_labels[-1]

    return out, int(mean_idx + 1)


def build_mean_cell_width_grades(cells, mean_score, grade_labels, cells_per_grade=15, include_mean_cell_in_upper=True):
    """
    Excel-style central-score cell-width method with spreadsheet-like one-step rim extension.

    Core logic:
    1) Discretize the selected min-max range into score cells.
    2) Find the cell containing the central score, normally Excel GEOMEAN.
    3) Assign the user-selected core grades around that central cell.
    4) If cells remain beyond the selected lower/upper grade range, append only
       one next-lower grade at the lower rim, matching the Excel sheet.

    Example:
    - Selected core grades = A, B+, B, C+
    - If lower cells remain after C+, they become C only.
    """
    out = cells.copy().reset_index(drop=True)
    out["assigned_grade"] = ""
    n_cells = len(out)
    n_grades = len(grade_labels)

    if n_cells == 0 or n_grades == 0:
        return out, None

    mean_idx = locate_cell_index_for_assignment(out, mean_score)
    if mean_idx is None:
        mean_idx = 0

    step = max(1, int(cells_per_grade))

    if n_grades % 2 == 0:
        lower_mid_gi = n_grades // 2
        upper_mid_gi = lower_mid_gi - 1

        # include_mean_cell_in_upper=True means the central-score cell belongs
        # to the upper-middle grade. This usually matches the spreadsheet layout.
        boundary_idx = mean_idx + 1 if include_mean_cell_in_upper else mean_idx

        upper_cursor = boundary_idx - 1
        for gi in range(upper_mid_gi, -1, -1):
            grade = grade_labels[gi]
            s = max(0, upper_cursor - step + 1)
            if upper_cursor >= 0:
                out.loc[s:upper_cursor, "assigned_grade"] = grade
            upper_cursor = s - 1

        lower_cursor = boundary_idx
        for gi in range(lower_mid_gi, n_grades):
            grade = grade_labels[gi]
            e = min(n_cells - 1, lower_cursor + step - 1)
            if lower_cursor < n_cells:
                out.loc[lower_cursor:e, "assigned_grade"] = grade
            lower_cursor = e + 1

    else:
        mid_gi = n_grades // 2

        # Odd number of selected grades: Excel-style middle-band placement.
        #
        # Key detail checked from the Excel sheet:
        # Scheme 2 has 5 grades and 12 cells per grade. The central score
        # is on/near cell 27, and the middle grade B occupies cells 22–33.
        # That means for an EVEN step size, the central cell sits slightly
        # above the mathematical center of the middle band:
        #   cells above = step//2 - 1
        #   cells below = step//2
        #
        # This avoids a one-cell upward shift that makes A too small and B+
        # too large. We use mean_idx directly instead of continuous geometry
        # because Excel operates by cell index after the central score is
        # located in the score-cell table.
        if step % 2 == 0:
            cells_above = max(0, step // 2 - 1)
        else:
            cells_above = step // 2

        start = mean_idx - cells_above
        start = max(0, min(n_cells - 1, start))
        end = min(n_cells - 1, start + step - 1)

        # If clipping at the bottom shortens the middle band, shift upward.
        if (end - start + 1) < step and start > 0:
            start = max(0, end - step + 1)

        out.loc[start:end, "assigned_grade"] = grade_labels[mid_gi]

        upper_cursor = start - 1
        for gi in range(mid_gi - 1, -1, -1):
            grade = grade_labels[gi]
            s = max(0, upper_cursor - step + 1)
            if upper_cursor >= 0:
                out.loc[s:upper_cursor, "assigned_grade"] = grade
            upper_cursor = s - 1

        lower_cursor = end + 1
        for gi in range(mid_gi + 1, n_grades):
            grade = grade_labels[gi]
            e = min(n_cells - 1, lower_cursor + step - 1)
            if lower_cursor < n_cells:
                out.loc[lower_cursor:e, "assigned_grade"] = grade
            lower_cursor = e + 1

    # ---------------------------------------------------------
    # Excel-like rim extension.
    #
    # The spreadsheet does NOT recursively create multiple rim grades
    # by step-size blocks. It first preserves the fixed core bands,
    # then uses only one rim grade for all remaining cells.
    #
    # Examples:
    # - Core A/B+/B/C+  -> lower rim becomes C
    # - Core A/B+/B/C+/C -> lower rim becomes D+
    # - Core A/B+/B/C+/C/D+ -> lower rim becomes D
    #
    # Upper unused cells are kept as the highest selected grade.
    # This matches the ตัดGrade sheet summary behavior:
    # Scheme 1 = A 11, B+ 26, B 11, C+ 8, C 2
    # Scheme 2 = A 7, B+ 24, B 13, C+ 4, C 8, D+ 2
    # ---------------------------------------------------------
    official = DEFAULT_GRADES_8

    def next_lower_grade(g):
        if g not in official:
            return g
        i = official.index(g)
        return official[min(len(official) - 1, i + 1)]

    assigned_idx = out.index[out["assigned_grade"] != ""].tolist()
    if not assigned_idx:
        out["assigned_grade"] = grade_labels[-1]
        return out, int(mean_idx + 1)

    first_assigned = min(assigned_idx)
    last_assigned = max(assigned_idx)

    # Upper rim: preserve the highest selected/core grade, not recursive expansion.
    if first_assigned > 0:
        out.loc[:first_assigned - 1, "assigned_grade"] = out.loc[first_assigned, "assigned_grade"]

    # Lower rim: append only the next lower official grade for all remaining cells.
    # Do not cascade into D+/D/F unless those grades are already part of the core.
    if last_assigned < n_cells - 1:
        bottom_core = out.loc[last_assigned, "assigned_grade"]
        out.loc[last_assigned + 1:, "assigned_grade"] = next_lower_grade(bottom_core)

    return out, int(mean_idx + 1)

def build_manual_boundary_cell_grades(cells, boundary_table):
    """
    Boundary table:
    grade | cutoff
    cutoff means score >= cutoff receives grade.
    """
    out = cells.copy()
    thresholds = [(r["grade"], r["cutoff"]) for _, r in boundary_table.iterrows()]
    out["assigned_grade"] = out["mid"].apply(lambda x: assign_by_threshold(x, thresholds))
    return out


def grade_from_cells(df, cells_with_grades, scheme_name):
    """
    Assign grades using the same logic as Excel COUNTIFS:
        score <= upper and score > lower

    Raw student scores are NOT rounded. This is important because a score such
    as 3.725384 should remain above the 3.725 boundary and stay in the upper
    cell, exactly as in Excel.
    """
    out = df.copy()

    def assign_cell_grade(score):
        score = float(score)
        cells_reset = cells_with_grades.reset_index(drop=True)

        for idx, row in cells_reset.iterrows():
            upper = float(row["upper"])
            lower = float(row["lower"])
            is_bottom = idx == len(cells_reset) - 1

            if is_bottom:
                ok = (score <= upper) and (score >= lower)
            else:
                ok = (score <= upper) and (score > lower)

            if ok:
                return row["assigned_grade"]

        # Values outside the selected min/max receive nearest rim grade.
        if score > float(cells_reset.iloc[0]["upper"]):
            return cells_reset.iloc[0]["assigned_grade"]
        return cells_reset.iloc[-1]["assigned_grade"]

    out[f"grade_qrange_{scheme_name}"] = out["score_4"].apply(assign_cell_grade)
    return out[["student_id", f"grade_qrange_{scheme_name}"]]


def quantile_cell_plot(cells, mean_score=None, q1=None, q2=None, q3=None, title="Quantile-range cells"):
    plot_df = cells.copy()
    plot_df["label"] = plot_df.apply(
        lambda r: f"{r['upper']:.4f} – {r['lower']:.4f}<br>Count: {r['count']}<br>Grade: {r.get('assigned_grade', '')}",
        axis=1,
    )

    fig = px.bar(
        plot_df,
        x="count",
        y="mid",
        orientation="h",
        color="assigned_grade" if "assigned_grade" in plot_df.columns else None,
        hover_name="label",
        title=title,
        height=750,
    )
    fig.update_yaxes(title="Score cell midpoint", autorange="reversed")
    fig.update_xaxes(title="Number of students")

    if mean_score is not None:
        fig.add_hline(y=mean_score, line_dash="solid", annotation_text="Central (GEOMEAN)", annotation_position="right")
    if q1 is not None:
        fig.add_hline(y=q1, line_dash="dot", annotation_text="Q1", annotation_position="left")
    if q2 is not None:
        fig.add_hline(y=q2, line_dash="dot", annotation_text="Q2", annotation_position="left")
    if q3 is not None:
        fig.add_hline(y=q3, line_dash="dot", annotation_text="Q3", annotation_position="left")

    return fig


# =========================================================
# K-MEANS MODE
# =========================================================

def kmeans_grading(df_all, df_calc, scheme_name, k, grade_labels, random_state=42):
    """
    K-means grading with Excel-like excluded-outlier handling.

    K-means is fitted only on df_calc. Students excluded from df_calc but still
    shown in df_all are handled as follows:
    - upper-end excluded students, score > max(df_calc), are assigned to A / top grade
    - lower-end excluded students, score < min(df_calc), are assigned to one grade
      lower than the lowest selected K-means grade
      e.g. A/B+/B/C+ -> lower outliers become C
           A/B+/B/C+/C -> lower outliers become D+
    - excluded students inside the fitted range keep the predicted K-means grade
    """
    out = df_all.copy()

    cluster_col = f"cluster_kmeans_{scheme_name}"
    grade_col = f"grade_kmeans_{scheme_name}"
    note_col = f"kmeans_note_{scheme_name}"

    if len(df_calc) < k or k <= 0:
        out[cluster_col] = np.nan
        out[grade_col] = "-"
        out[note_col] = "not enough calculation data"
        return out[["student_id", cluster_col, grade_col, note_col]], None, np.nan

    model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    model.fit(df_calc[["score_4"]])

    out[cluster_col] = model.predict(out[["score_4"]])

    center_df = pd.DataFrame({
        "cluster": range(k),
        "center": model.cluster_centers_.flatten(),
    }).sort_values("center", ascending=False).reset_index(drop=True)

    cluster_to_grade = {int(row["cluster"]): grade_labels[i] for i, row in center_df.iterrows()}
    out[grade_col] = out[cluster_col].map(cluster_to_grade)
    out[note_col] = "k-means prediction"

    official = DEFAULT_GRADES_8

    def next_lower_grade(g):
        if g not in official:
            return g
        i = official.index(g)
        return official[min(len(official) - 1, i + 1)]

    calc_ids = set(df_calc["student_id"].astype(str))
    out["_excluded_from_kmeans_fit"] = ~out["student_id"].astype(str).isin(calc_ids)

    calc_min = float(df_calc["score_4"].min())
    calc_max = float(df_calc["score_4"].max())

    top_grade = grade_labels[0] if len(grade_labels) else "A"
    bottom_grade = grade_labels[-1] if len(grade_labels) else "-"
    lower_outlier_grade = next_lower_grade(bottom_grade)

    upper_mask = out["_excluded_from_kmeans_fit"] & (out["score_4"] > calc_max)
    lower_mask = out["_excluded_from_kmeans_fit"] & (out["score_4"] < calc_min)

    out.loc[upper_mask, grade_col] = top_grade
    out.loc[upper_mask, note_col] = "excluded upper-end outlier -> top grade"

    out.loc[lower_mask, grade_col] = lower_outlier_grade
    out.loc[lower_mask, note_col] = f"excluded lower-end outlier -> one lower than {bottom_grade}"

    inside_excluded = out["_excluded_from_kmeans_fit"] & ~(upper_mask | lower_mask)
    out.loc[inside_excluded, note_col] = "excluded from fit but inside fitted range -> predicted grade"

    summary = (
        out.groupby(grade_col)
        .agg(
            students=("student_id", "count"),
            min_score=("score_4", "min"),
            max_score=("score_4", "max"),
            mean_score=("score_4", "mean"),
        )
        .reset_index()
        .rename(columns={grade_col: "grade"})
    )

    # Keep summary in official grade order.
    summary["_order"] = summary["grade"].map(GRADE_ORDER).fillna(99)
    summary = summary.sort_values("_order").drop(columns="_order").reset_index(drop=True)

    sil = np.nan
    if k > 1 and len(df_calc) > k:
        labels = model.predict(df_calc[["score_4"]])
        try:
            sil = silhouette_score(df_calc[["score_4"]], labels)
        except Exception:
            sil = np.nan

    return out[["student_id", cluster_col, grade_col, note_col]], summary, sil


# =========================================================
# SESSION STATE
# =========================================================

if "result_tables" not in st.session_state:
    st.session_state.result_tables = {}

if "scheme_meta" not in st.session_state:
    st.session_state.scheme_meta = []

if "latest_grade_col" not in st.session_state:
    st.session_state.latest_grade_col = None


# =========================================================
# SIDEBAR: UPLOAD AND SETTINGS
# =========================================================

st.sidebar.header("1. Upload / อัปโหลดข้อมูล")
st.sidebar.caption("ไฟล์ควรมีอย่างน้อย 2 คอลัมน์: รหัสนิสิต และคะแนนรวม")
uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])

if uploaded_file is None:
    st.info("Upload a CSV or Excel file. The minimum required columns are student ID and final score.")
    st.stop()

raw_df = read_uploaded_file(uploaded_file)
cols = list(raw_df.columns)

st.sidebar.header("2. Columns / เลือกคอลัมน์")

nid_default = 0
score_default = min(1, len(cols) - 1)
id_col = st.sidebar.selectbox("Student ID column", cols, index=nid_default)
score_col = st.sidebar.selectbox("Final score column", cols, index=score_default)

base_df = clean_dataframe(raw_df, id_col, score_col)

st.sidebar.header("3. Score scale / สเกลคะแนน")
scale_mode = st.sidebar.radio(
    "Score handling",
    [
        "Score is already 0–4",
        "Normalize by observed min/max to 0–4",
        "Normalize by custom raw min/max to 0–4",
    ],
    index=0,
)

raw_min = None
raw_max = None
if scale_mode == "Normalize by custom raw min/max to 0–4":
    raw_min = st.sidebar.number_input("Raw minimum", value=0.0, step=1.0)
    raw_max = st.sidebar.number_input("Raw maximum", value=100.0, step=1.0)

base_df = normalize_scores(base_df, scale_mode, raw_min, raw_max)
base_df, outlier_info = detect_outliers(base_df, score_col="score_4")

st.sidebar.header("4. Outlier policy / การจัดการคะแนนผิดปกติ")
outlier_policy = st.sidebar.radio(
    "Use outliers in calculation?",
    [
        "Use all students",
        "Exclude detected outliers from calculation only",
        "Manually exclude selected students from calculation only",
        "Exclude detected + manually selected from calculation",
    ],
    index=0,
)
manual_exclude_ids = st.sidebar.multiselect(
    "Manual exclusion from calculation",
    options=base_df["student_id"].tolist(),
)

calc_df = get_calc_df(base_df, outlier_policy, manual_exclude_ids)


# =========================================================
# MAIN OVERVIEW
# =========================================================

with st.expander("Raw data preview", expanded=False):
    st.dataframe(raw_df.head(30), use_container_width=True)

st.subheader("Dataset overview")

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Students", len(base_df))
c2.metric("Calc. students", len(calc_df))
central_score_raw = excel_geomean(calc_df["score_4"])
central_score = round_for_cell_assignment(central_score_raw, 3)
c3.metric("Central (GEOMEAN)", f"{central_score:.3f}")
c4.metric("Median", f"{calc_df['score_4'].median():.4f}")
c5.metric("Std", f"{calc_df['score_4'].std(ddof=0):.4f}")
c6.metric("Outliers", int(base_df["outlier_any"].sum()))

st.plotly_chart(distribution_plot(base_df, title="0–4 score distribution with outliers and Excel quartiles"), use_container_width=True)

with st.expander("คำแนะนำสำหรับการใช้ในที่ประชุมอาจารย์ / Faculty review guide", expanded=True):
    st.markdown(
        """
        - เริ่มจากดูการกระจายคะแนนรวม ค่าเฉลี่ย มัธยฐาน และ outlier ก่อนตัดสินใจเลือกวิธีตัดเกรด
        - ใช้ **Z-score** เมื่ออยากให้เกรดสัมพันธ์กับค่าเฉลี่ยและส่วนเบี่ยงเบนมาตรฐานของห้อง
        - ใช้ **Simple range** เมื่อคณะหรือรายวิชามีเกณฑ์คะแนนตายตัว เช่น A ≥ 3.70
        - ใช้ **Quantile-range** เมื่ออยากปรับช่วงเกรดแบบละเอียดและเห็นจำนวนคนในแต่ละ cell
        - ใช้ **K-means** เพื่อดูการแบ่งกลุ่มคะแนนตามธรรมชาติของข้อมูล แต่ควรใช้เป็นข้อมูลประกอบ ไม่ใช่การตัดสินอัตโนมัติ
        - กราฟแท่งและ donut chart จะแสดงผลใหม่ทุกครั้งเมื่อปรับเกณฑ์ เพื่อช่วยดูผลกระทบของการปรับเกรดแบบทันที
        """
    )

with st.expander("Outlier and Excel quartile details", expanded=False):
    q_df = pd.DataFrame([outlier_info]).T.reset_index()
    q_df.columns = ["Metric", "Value"]
    st.dataframe(q_df, use_container_width=True)

    outlier_df = base_df[base_df["outlier_any"]].copy()
    if outlier_df.empty:
        st.success("No outliers detected by Z-score or IQR.")
    else:
        st.dataframe(
            outlier_df[["student_id", "raw_score", "score_4", "z_for_outlier", "z_outlier", "iqr_outlier"]],
            use_container_width=True,
        )


# =========================================================
# TABS
# =========================================================

tab_z, tab_range, tab_qrange, tab_kmeans, tab_compare = st.tabs(
    ["Z-score", "Simple range", "Quantile-range", "K-means", "Final comparison & export"]
)


# =========================================================
# TAB 1: Z-SCORE
# =========================================================

with tab_z:
    st.subheader("Z-score grading")
    st.write("Grades are assigned by z-score boundaries. Calculation can include or exclude outliers depending on the sidebar policy.")
    st.info("ภาษาไทย: วิธีนี้เหมาะเมื่ออาจารย์ต้องการดูว่านิสิตอยู่สูงหรือต่ำกว่าค่าเฉลี่ยของห้องมากเพียงใด เกณฑ์ z ที่ปรับจะทำให้จำนวนเกรดเปลี่ยนทันทีในกราฟด้านล่าง")

    z_scheme_name = st.text_input("Z-score scheme name", value="Z1", key="z_scheme_name")
    z_grades = st.multiselect("Z-score grades", DEFAULT_GRADES_8, default=DEFAULT_GRADES_8, key="z_grades")

    if len(z_grades) < 2:
        st.warning("Select at least two grades.")
    else:
        default_z = {
            "A": 1.50,
            "B+": 1.00,
            "B": 0.50,
            "C+": 0.00,
            "C": -0.50,
            "D+": -1.00,
            "D": -1.50,
            "F": -999.0,
        }

        z_thresholds = []
        z_cols = st.columns(4)
        for i, grade in enumerate(z_grades):
            if grade == z_grades[-1]:
                cutoff = -999.0
                st.caption(f"{grade}: below previous boundary")
            else:
                cutoff = z_cols[i % 4].number_input(
                    f"{grade} if z ≥",
                    value=float(default_z.get(grade, 0.0)),
                    step=0.05,
                    key=f"z_cut_{z_scheme_name}_{grade}",
                )
            z_thresholds.append((grade, cutoff))

        z_res = zscore_grading(base_df, calc_df, z_scheme_name, z_thresholds)
        z_full = base_df.merge(z_res, on="student_id", how="left")
        z_grade_col = f"grade_z_{z_scheme_name}"

        st.session_state.result_tables[z_grade_col] = z_res
        st.session_state.latest_grade_col = z_grade_col
        if not any(m["column"] == z_grade_col for m in st.session_state.scheme_meta):
            st.session_state.scheme_meta.append({"mode": "Z-score", "scheme": z_scheme_name, "column": z_grade_col})

        boundaries_score = z_boundaries_to_score(calc_df, z_thresholds)

        show_grade_review_panel(z_full, z_grade_col, title_prefix=f"Z-score {z_scheme_name}: ")
        show_supporting_visuals(
            z_full,
            z_grade_col,
            title_prefix=f"Z-score {z_scheme_name}: ",
            main_fig=distribution_plot(z_full, title=f"Score distribution with Z-score cutoffs: {z_scheme_name}", boundaries=boundaries_score),
        )
        show_bottom_check_tables(
            z_full,
            z_grade_col,
            summary_df=grade_distribution_ordered(z_full, z_grade_col),
            summary_title="Grade distribution table",
            table_title="Student checking table — Z-score result",
        )


# =========================================================
# TAB 2: SIMPLE RANGE
# =========================================================

with tab_range:
    st.subheader("Simple hand-adjusted range grading")
    st.write("This is the direct cutoff mode. It is useful for final manual adjustment or official fixed-boundary grading.")
    st.info("ภาษาไทย: วิธีนี้เหมาะสำหรับการปรับเกณฑ์ด้วยมือ เช่น ขยับเส้น A, B+, B แล้วดูทันทีว่าจำนวนนิสิตในแต่ละเกรดเปลี่ยนอย่างไร")

    r_scheme_name = st.text_input("Simple range scheme name", value="R1", key="r_scheme_name")
    r_grades = st.multiselect("Range grades", DEFAULT_GRADES_8, default=DEFAULT_GRADES_8, key="r_grades")

    if len(r_grades) < 2:
        st.warning("Select at least two grades.")
    else:
        default_r = {
            "A": 3.70,
            "B+": 3.30,
            "B": 3.00,
            "C+": 2.50,
            "C": 2.00,
            "D+": 1.50,
            "D": 1.00,
            "F": -999.0,
        }

        r_thresholds = []
        r_cols = st.columns(4)
        for i, grade in enumerate(r_grades):
            if grade == r_grades[-1]:
                cutoff = -999.0
                st.caption(f"{grade}: below previous boundary")
            else:
                cutoff = r_cols[i % 4].number_input(
                    f"{grade} if score ≥",
                    value=float(default_r.get(grade, 0.0)),
                    min_value=0.0,
                    max_value=4.0,
                    step=0.025,
                    format="%.4f",
                    key=f"r_cut_{r_scheme_name}_{grade}",
                )
            r_thresholds.append((grade, cutoff))

        r_res = simple_range_grading(base_df, r_scheme_name, r_thresholds)
        r_full = base_df.merge(r_res, on="student_id", how="left")
        r_grade_col = f"grade_range_{r_scheme_name}"

        st.session_state.result_tables[r_grade_col] = r_res
        st.session_state.latest_grade_col = r_grade_col
        if not any(m["column"] == r_grade_col for m in st.session_state.scheme_meta):
            st.session_state.scheme_meta.append({"mode": "Simple range", "scheme": r_scheme_name, "column": r_grade_col})

        boundaries = [(g, c) for g, c in r_thresholds if c > -100]

        show_grade_review_panel(r_full, r_grade_col, title_prefix=f"Simple range {r_scheme_name}: ")
        show_supporting_visuals(
            r_full,
            r_grade_col,
            title_prefix=f"Simple range {r_scheme_name}: ",
            main_fig=distribution_plot(r_full, title=f"Score distribution with manual cutoffs: {r_scheme_name}", boundaries=boundaries),
        )
        show_bottom_check_tables(
            r_full,
            r_grade_col,
            summary_df=grade_distribution_ordered(r_full, r_grade_col),
            summary_title="Grade distribution table",
            table_title="Student checking table — Simple range result",
        )


# =========================================================
# TAB 3: QUANTILE-RANGE
# =========================================================

with tab_qrange:
    st.subheader("Quantile-range grading")
    st.write(
        "This mode follows the Excel-style workflow: divide the 0–4 score range into small cells, count students in each cell, inspect quantiles/mean, then assign grades by cell groups."
    )
    st.info("ภาษาไทย: วิธีนี้เหมาะกับการประชุมที่ต้องการเห็นช่วงคะแนนละเอียดมากขึ้น คล้ายการแบ่ง cell ใน Excel แล้วค่อยพิจารณาว่าช่วงใดควรเป็น A, B+, B หรือ C+")

    q_scheme_name = st.text_input("Quantile-range scheme name", value="QR1", key="q_scheme_name")

    q_col1, q_col2, q_col3 = st.columns(3)
    cell_width = q_col1.number_input("Cell width", value=0.025, min_value=0.005, max_value=0.500, step=0.005, format="%.4f")
    assignment_mode = q_col2.radio(
        "Assignment mode",
        [
            "Cell counts from top",
            "Cell counts from mean",
            "Central-score cell width mode",
            "Manual cutoffs",
        ],
        index=0,
    )
    q_use_calc = q_col3.radio("Cell count source", ["All students", "Calculation set"], index=0)

    q_count_df = base_df if q_use_calc == "All students" else calc_df

    range_col1, range_col2, range_col3 = st.columns(3)
    q_range_mode = range_col1.radio(
        "Cell min/max source",
        ["0–4 scale", "Observed min/max", "Custom min/max"],
        index=0,
        horizontal=True,
    )

    observed_min = float(q_count_df["score_4"].min())
    observed_max = float(q_count_df["score_4"].max())

    if q_range_mode == "0–4 scale":
        cell_min_score, cell_max_score = 0.0, 4.0
    elif q_range_mode == "Observed min/max":
        cell_min_score, cell_max_score = observed_min, observed_max
    else:
        cell_min_score = range_col2.number_input(
            "Custom cell minimum",
            value=max(0.0, math.floor(observed_min / cell_width) * cell_width),
            min_value=0.0,
            max_value=4.0,
            step=cell_width,
            format="%.4f",
        )
        cell_max_score = range_col3.number_input(
            "Custom cell maximum",
            value=min(4.0, math.ceil(observed_max / cell_width) * cell_width),
            min_value=0.0,
            max_value=4.0,
            step=cell_width,
            format="%.4f",
        )

    if cell_max_score <= cell_min_score:
        st.error("Cell maximum must be greater than cell minimum.")
        st.stop()

    cells = make_score_cells(cell_width=cell_width, min_score=cell_min_score, max_score=cell_max_score)
    cells = count_students_in_cells(q_count_df, cells, score_col="score_4")

    mean_score_raw = excel_geomean(calc_df["score_4"])
    mean_score = round_for_cell_assignment(mean_score_raw, 3)
    q1 = excel_quartile_exc(calc_df["score_4"], 1)
    q2 = excel_quartile_exc(calc_df["score_4"], 2)
    q3 = excel_quartile_exc(calc_df["score_4"], 3)

    st.write(f"Central score (Excel GEOMEAN, rounded anchor) = **{mean_score:.3f}**, Q1/Q2/Q3 (Excel QUARTILE.EXC) = **{q1:.4f}**, **{q2:.4f}**, **{q3:.4f}**")

    if assignment_mode == "Cell counts from top":
        st.write("Define how many sub-range cells each grade occupies from top score downward.")
        q_grades = st.multiselect("Grades", DEFAULT_GRADES_8, default=DEFAULT_GRADES_5, key="q_grades_count")

        if len(q_grades) < 2:
            st.warning("Select at least two grades.")
        else:
            default_counts = {
                "A": 11,
                "B+": 17,
                "B": 18,
                "C+": 10,
                "C": 10,
                "D+": 10,
                "D": 10,
                "F": 10,
            }

            count_inputs = {}
            count_cols = st.columns(4)
            for i, grade in enumerate(q_grades):
                if grade == q_grades[-1]:
                    st.caption(f"{grade}: remaining lower cells")
                    count_inputs[grade] = 9999
                else:
                    count_inputs[grade] = count_cols[i % 4].number_input(
                        f"{grade}: number of cells",
                        value=int(default_counts.get(grade, 10)),
                        min_value=1,
                        max_value=len(cells),
                        step=1,
                        key=f"q_count_{q_scheme_name}_{grade}",
                    )

            cells_assigned, mean_cell_id = build_mean_centered_cell_grades(
                cells,
                mean_score=mean_score,
                grade_labels=q_grades,
                cell_counts_by_grade=count_inputs,
            )

            st.caption(f"Central score is located around cell ID: {mean_cell_id}")

            q_res = grade_from_cells(base_df, cells_assigned, q_scheme_name)
            q_full = base_df.merge(q_res, on="student_id", how="left")
            q_grade_col = f"grade_qrange_{q_scheme_name}"

            st.session_state.result_tables[q_grade_col] = q_res
            st.session_state.latest_grade_col = q_grade_col
            if not any(m["column"] == q_grade_col for m in st.session_state.scheme_meta):
                st.session_state.scheme_meta.append({"mode": "Quantile-range", "scheme": q_scheme_name, "column": q_grade_col})

            show_grade_review_panel(q_full, q_grade_col, title_prefix=f"Quantile-range {q_scheme_name}: ")
            show_supporting_visuals(
                q_full,
                q_grade_col,
                title_prefix=f"Quantile-range {q_scheme_name}: ",
                main_fig=quantile_cell_plot(cells_assigned, mean_score=mean_score, q1=q1, q2=q2, q3=q3, title=f"Quantile-range cells: {q_scheme_name}"),
            )
            show_bottom_check_tables(
                q_full,
                q_grade_col,
                summary_df=grade_distribution_ordered(q_full, q_grade_col),
                summary_title="Grade distribution table",
                table_title="Student checking table — Quantile-range result",
            )
            with st.expander("Cell table — detailed score cells", expanded=False):
                st.dataframe(cells_assigned, use_container_width=True, height=500)

    elif assignment_mode == "Cell counts from mean":
        st.write("Define how many cells each grade occupies, anchored at the mean and expanding upward/downward from the middle grade area.")
        q_grades = st.multiselect("Grades", DEFAULT_GRADES_8, default=DEFAULT_GRADES_5, key="q_grades_mean")

        if len(q_grades) < 2:
            st.warning("Select at least two grades.")
        else:
            default_counts = {
                "A": 11,
                "B+": 17,
                "B": 18,
                "C+": 10,
                "C": 10,
                "D+": 10,
                "D": 10,
                "F": 10,
            }

            count_inputs = {}
            count_cols = st.columns(4)
            for i, grade in enumerate(q_grades):
                count_inputs[grade] = count_cols[i % 4].number_input(
                    f"{grade}: number of cells",
                    value=int(default_counts.get(grade, 10)),
                    min_value=1,
                    max_value=len(cells),
                    step=1,
                    key=f"q_mean_count_{q_scheme_name}_{grade}",
                )

            if len(q_grades) % 2 == 1:
                middle_grade = q_grades[len(q_grades) // 2]
                st.caption(f"Mean anchor: middle grade = {middle_grade}. This grade is centered on the mean cell, then grades expand upward and downward.")
            else:
                upper_middle = q_grades[len(q_grades) // 2 - 1]
                lower_middle = q_grades[len(q_grades) // 2]
                st.caption(f"Mean anchor: boundary between {upper_middle} and {lower_middle} is placed closest to the mean, then grades expand upward and downward.")

            cells_assigned, mean_cell_id = build_mean_anchored_cell_grades(
                cells,
                mean_score=mean_score,
                grade_labels=q_grades,
                cell_counts_by_grade=count_inputs,
            )

            st.caption(f"Central score is located around cell ID: {mean_cell_id}")

            q_res = grade_from_cells(base_df, cells_assigned, q_scheme_name)
            q_full = base_df.merge(q_res, on="student_id", how="left")
            q_grade_col = f"grade_qrange_{q_scheme_name}"

            st.session_state.result_tables[q_grade_col] = q_res
            st.session_state.latest_grade_col = q_grade_col
            if not any(m["column"] == q_grade_col for m in st.session_state.scheme_meta):
                st.session_state.scheme_meta.append({"mode": "Quantile-range", "scheme": q_scheme_name, "column": q_grade_col})

            show_grade_review_panel(q_full, q_grade_col, title_prefix=f"Quantile-range {q_scheme_name}: ")
            show_supporting_visuals(
                q_full,
                q_grade_col,
                title_prefix=f"Quantile-range {q_scheme_name}: ",
                main_fig=quantile_cell_plot(cells_assigned, mean_score=mean_score, q1=q1, q2=q2, q3=q3, title=f"Quantile-range cells: {q_scheme_name}"),
            )
            show_bottom_check_tables(
                q_full,
                q_grade_col,
                summary_df=grade_distribution_ordered(q_full, q_grade_col),
                summary_title="Grade distribution table",
                table_title="Student checking table — Quantile-range result",
            )
            with st.expander("Cell table — detailed score cells", expanded=False):
                st.dataframe(cells_assigned, use_container_width=True, height=500)

    elif assignment_mode == "Central-score cell width mode":
        st.write("Discretize the selected min-max range, find the cell containing the central score, then move grade boundaries upward/downward by a fixed number of cells. If cells remain after the selected grades are used, the app automatically adds lower/upper rim grades from the official grade ladder.")
        q_grades = st.multiselect("Grades", DEFAULT_GRADES_8, default=DEFAULT_GRADES_4, key="q_grades_mean_width")

        if len(q_grades) < 2:
            st.warning("Select at least two grades.")
        else:
            mean_width_cols = st.columns(3)
            cells_per_grade = mean_width_cols[0].number_input(
                "Cells per grade step",
                value=15,
                min_value=1,
                max_value=len(cells),
                step=1,
                key=f"q_mean_width_cells_{q_scheme_name}",
            )
            include_mean_cell = mean_width_cols[1].checkbox(
                "Include mean cell in upper-middle grade",
                value=True,
                key=f"q_mean_width_include_{q_scheme_name}",
            )

            if len(q_grades) % 2 == 0:
                upper_middle = q_grades[len(q_grades) // 2 - 1]
                lower_middle = q_grades[len(q_grades) // 2]
                st.caption(f"Central-cell anchor: boundary between {upper_middle} and {lower_middle}; each grade step = {cells_per_grade} cells.")
            else:
                middle_grade = q_grades[len(q_grades) // 2]
                st.caption(f"Central-cell anchor: {middle_grade} is centered on the mean cell; each grade step = {cells_per_grade} cells.")

            cells_assigned, mean_cell_id = build_mean_cell_width_grades(
                cells,
                mean_score=mean_score,
                grade_labels=q_grades,
                cells_per_grade=cells_per_grade,
                include_mean_cell_in_upper=include_mean_cell,
            )

            st.caption(f"Central score is located around cell ID: {mean_cell_id}")

            q_res = grade_from_cells(base_df, cells_assigned, q_scheme_name)
            q_full = base_df.merge(q_res, on="student_id", how="left")
            q_grade_col = f"grade_qrange_{q_scheme_name}"

            st.session_state.result_tables[q_grade_col] = q_res
            st.session_state.latest_grade_col = q_grade_col
            if not any(m["column"] == q_grade_col for m in st.session_state.scheme_meta):
                st.session_state.scheme_meta.append({"mode": "Quantile-range", "scheme": q_scheme_name, "column": q_grade_col})

            show_grade_review_panel(q_full, q_grade_col, title_prefix=f"Quantile-range {q_scheme_name}: ")
            show_supporting_visuals(
                q_full,
                q_grade_col,
                title_prefix=f"Quantile-range {q_scheme_name}: ",
                main_fig=quantile_cell_plot(cells_assigned, mean_score=mean_score, q1=q1, q2=q2, q3=q3, title=f"Quantile-range cells: {q_scheme_name}"),
            )
            show_bottom_check_tables(
                q_full,
                q_grade_col,
                summary_df=grade_distribution_ordered(q_full, q_grade_col),
                summary_title="Grade distribution table",
                table_title="Student checking table — Quantile-range result",
            )
            with st.expander("Cell table — detailed score cells", expanded=False):
                st.dataframe(cells_assigned, use_container_width=True, height=500)

    else:
        st.write("Manual cutoffs still use the quantile-cell table for visual checking.")
        q_grades = st.multiselect("Grades", DEFAULT_GRADES_8, default=DEFAULT_GRADES_8, key="q_grades_manual")

        if len(q_grades) < 2:
            st.warning("Select at least two grades.")
        else:
            default_cutoffs = {
                "A": 3.725,
                "B+": 3.325,
                "B": 3.000,
                "C+": 2.500,
                "C": 2.000,
                "D+": 1.500,
                "D": 1.000,
                "F": -999.0,
            }

            rows = []
            m_cols = st.columns(4)
            for i, grade in enumerate(q_grades):
                if grade == q_grades[-1]:
                    cutoff = -999.0
                    st.caption(f"{grade}: below previous boundary")
                else:
                    cutoff = m_cols[i % 4].number_input(
                        f"{grade} if score ≥",
                        value=float(default_cutoffs.get(grade, 0.0)),
                        min_value=0.0,
                        max_value=4.0,
                        step=cell_width,
                        format="%.4f",
                        key=f"q_manual_{q_scheme_name}_{grade}",
                    )
                rows.append({"grade": grade, "cutoff": cutoff})

            boundary_table = pd.DataFrame(rows)
            cells_assigned = build_manual_boundary_cell_grades(cells, boundary_table)

            q_res = grade_from_cells(base_df, cells_assigned, q_scheme_name)
            q_full = base_df.merge(q_res, on="student_id", how="left")
            q_grade_col = f"grade_qrange_{q_scheme_name}"

            st.session_state.result_tables[q_grade_col] = q_res
            st.session_state.latest_grade_col = q_grade_col
            if not any(m["column"] == q_grade_col for m in st.session_state.scheme_meta):
                st.session_state.scheme_meta.append({"mode": "Quantile-range", "scheme": q_scheme_name, "column": q_grade_col})

            show_grade_review_panel(q_full, q_grade_col, title_prefix=f"Quantile-range {q_scheme_name}: ")
            show_supporting_visuals(
                q_full,
                q_grade_col,
                title_prefix=f"Quantile-range {q_scheme_name}: ",
                main_fig=quantile_cell_plot(cells_assigned, mean_score=mean_score, q1=q1, q2=q2, q3=q3, title=f"Quantile-range cells: {q_scheme_name}"),
            )
            show_bottom_check_tables(
                q_full,
                q_grade_col,
                summary_df=grade_distribution_ordered(q_full, q_grade_col),
                summary_title="Grade distribution table",
                table_title="Student checking table — Quantile-range result",
            )
            with st.expander("Cell table — detailed score cells", expanded=False):
                st.dataframe(cells_assigned, use_container_width=True, height=500)


# =========================================================
# TAB 4: K-MEANS
# =========================================================

with tab_kmeans:
    st.subheader("K-means grading")
    st.write("Scores are clustered first using the calculation set. If outliers are excluded from fitting, upper-end excluded students are assigned to the top grade, while lower-end excluded students are assigned to one grade lower than the lowest selected grade.")
    st.info("ภาษาไทย: K-means ใช้ดูว่าคะแนนแบ่งเป็นกลุ่มธรรมชาติอย่างไร เหมาะเป็นภาพประกอบการตัดสินใจ แต่ควรตรวจสอบด้วยสายตาและบริบทของรายวิชาเสมอ")

    k_scheme_name = st.text_input("K-means scheme name", value="K1", key="k_scheme_name")
    k = st.slider("Number of clusters / grades", min_value=1, max_value=8, value=5)

    label_options = {
        1: ["-"],
        2: ["A", "B"],
        3: ["A", "B", "C"],
        4: DEFAULT_GRADES_4,
        5: DEFAULT_GRADES_5,
        6: DEFAULT_GRADES_6,
        7: DEFAULT_GRADES_7,
        8: DEFAULT_GRADES_8,
    }

    grade_label_text = st.text_input(
        "Grade labels from highest to lowest",
        value=",".join(label_options[k]),
        key="k_labels",
    )
    grade_labels = [x.strip() for x in grade_label_text.split(",") if x.strip()]

    if len(grade_labels) != k:
        st.warning("Number of grade labels must match k.")
    else:
        k_res, k_summary, sil = kmeans_grading(base_df, calc_df, k_scheme_name, k, grade_labels)
        k_full = base_df.merge(k_res, on="student_id", how="left")
        k_grade_col = f"grade_kmeans_{k_scheme_name}"

        st.session_state.result_tables[k_grade_col] = k_res
        st.session_state.latest_grade_col = k_grade_col
        if not any(m["column"] == k_grade_col for m in st.session_state.scheme_meta):
            st.session_state.scheme_meta.append({"mode": "K-means", "scheme": k_scheme_name, "column": k_grade_col})

        if not np.isnan(sil):
            st.metric("Silhouette score", f"{sil:.3f}")

        show_grade_review_panel(k_full, k_grade_col, title_prefix=f"K-means {k_scheme_name}: ")
        show_supporting_visuals(
            k_full,
            k_grade_col,
            title_prefix=f"K-means {k_scheme_name}: ",
            main_fig=distribution_plot(k_full, title=f"Score distribution with K-means grades: {k_scheme_name}"),
        )
        show_bottom_check_tables(
            k_full,
            k_grade_col,
            summary_df=k_summary if k_summary is not None else grade_distribution_ordered(k_full, k_grade_col),
            summary_title="Cluster / grade summary",
            table_title="Student checking table — K-means result",
        )


# =========================================================
# TAB 5: COMPARISON AND EXPORT
# =========================================================

with tab_compare:
    st.subheader("Comparison & export")

    comparison = base_df[["student_id", "raw_score", "score_4", "outlier_any"]].copy()

    for grade_col, table in st.session_state.result_tables.items():
        comparison = comparison.merge(table, on="student_id", how="left")

    grade_cols = [c for c in comparison.columns if c.startswith("grade_")]

    if len(grade_cols) == 0:
        st.info("Run at least one grading scheme first.")
    else:
        st.write("Available schemes")
        st.dataframe(pd.DataFrame(st.session_state.scheme_meta), use_container_width=True)

        preferred_final = st.session_state.get("latest_grade_col")
        if preferred_final not in grade_cols:
            preferred_final = st.session_state.get("final_grade_col")
        default_index = grade_cols.index(preferred_final) if preferred_final in grade_cols else len(grade_cols) - 1

        final_grade_col = st.selectbox(
            "Select final grade column for official export",
            grade_cols,
            index=default_index,
            key="final_grade_col",
        )
        comparison["final_selected_grade"] = comparison[final_grade_col]
        comparison["grade_point"] = comparison["final_selected_grade"].map(GRADE_POINTS)
        comparison = add_rank(comparison, score_col="score_4")

        avg_gp = comparison["grade_point"].dropna().mean()
        f_count = int((comparison["final_selected_grade"] == "F").sum())

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Selected scheme", final_grade_col)
        c2.metric("Average grade point", f"{avg_gp:.2f}" if not np.isnan(avg_gp) else "-")
        c3.metric("F count", f_count)
        c4.metric("A count", int((comparison["final_selected_grade"] == "A").sum()))

        st.write("Selected grade distribution table")
        st.dataframe(grade_distribution_ordered(comparison, "final_selected_grade"), use_container_width=True)

        show_grade_review_panel(comparison, "final_selected_grade", title_prefix="Final selected: ")

        st.plotly_chart(
            ranked_score_plot(comparison, grade_col="final_selected_grade", title="Final selected grade by score rank"),
            use_container_width=True,
        )

        sort_mode = st.radio(
            "Display sorting",
            ["Student ID", "Score descending", "Score ascending", "Grade"],
            horizontal=True,
        )

        display_df = sort_dataframe(comparison, sort_mode, grade_col="final_selected_grade")
        st.dataframe(display_df, use_container_width=True, height=600)

        official_df = comparison.sort_values("student_id", ascending=True)
        ranked_df = comparison.sort_values("score_4", ascending=False)

        export_minimal = official_df[["student_id", "raw_score", "score_4", "final_selected_grade"]].copy()
        export_minimal = export_minimal.rename(columns={"final_selected_grade": "grade"})

        st.download_button(
            "Download official CSV — student ID order",
            data=to_csv_bytes(export_minimal),
            file_name="official_grades_by_student_id.csv",
            mime="text/csv",
        )

        st.download_button(
            "Download full comparison CSV — student ID order",
            data=to_csv_bytes(official_df),
            file_name="full_grade_comparison_by_student_id.csv",
            mime="text/csv",
        )

        st.download_button(
            "Download ranked review CSV — score order",
            data=to_csv_bytes(ranked_df),
            file_name="ranked_grade_review.csv",
            mime="text/csv",
        )
