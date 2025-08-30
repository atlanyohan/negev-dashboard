# -*- coding: utf-8 -*-
import json, re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ========================
# Page + RTL styling
# ========================
st.set_page_config(page_title="דשבורד נגב – סקירה · דירוג · מפה · חקירה", layout="wide")

st.markdown(
    """
    <style>
      .rtl { direction: rtl; text-align: right; }
      .stApp, .rtl .stMarkdown p { direction: rtl; text-align: right; }
      .stSelectbox > div > div, .stMultiSelect > div > div, .stTextInput > div > div input { direction: rtl; text-align: right; }
      .stDataFrame, .stTable { direction: rtl; text-align: right; }
      .kpi { border:1px solid #eee; padding:12px; border-radius:12px; margin-bottom:8px; }
      .kpi h3 { font-size:1rem; margin:0 0 6px 0; }
      .kpi .val { font-size:1.4rem; font-weight:700; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="rtl"><h1>📊 דשבורד נגב (2019) — סקירה · דירוג · מפה · חקירה</h1></div>', unsafe_allow_html=True)

# ========================
# File paths
# ========================
DATA_FILE = Path(__file__).parent / "negev_data.json"        # קובץ נתונים מעובד
AUTH_FILE = Path(__file__).parent / "negev_31_list.json"     # אופציונלי: רשימת 31 הרשויות
GEO_FILE_CANDIDATES = [
    Path(__file__).parent / "negev_geo.json",
    Path(__file__).parent / "israel_munis.geojson",
    Path(__file__).parent / "israel_munis.json",
]

# ========================
# Helpers
# ========================
POSSIBLE_NAME_COLS = ["שם רשות", "שם הרשות", "רשות", "ישוב", "רשות מקומית"]
DELIMS = [" – ", "–", " - ", "-", ":", "|", "/", "—"]
YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
HE_WORDS = re.compile(r"[\u0590-\u05FFA-Za-z0-9]+")

def make_unique_columns(cols):
    seen, out = {}, []
    for c in cols:
        c = "" if c is None else str(c)
        if c not in seen:
            seen[c] = 1
            out.append(c)
        else:
            seen[c] += 1
            out.append(f"{c}__{seen[c]}")
    return out

def smart_split(label: str):
    """Extract topic/subtopic even if there is no clear '–' delimiter."""
    s = str(label).strip()
    if not s:
        return []
    for d in DELIMS:
        if d in s:
            parts = [p.strip() for p in s.split(d) if p.strip()]
            if parts:
                return parts
    m = YEAR_RE.search(s)
    if m:
        s = s[:m.start()].strip()
    s = re.sub(r"\([^)]*\)", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    words = HE_WORDS.findall(s)
    if not words:
        return [label]
    topic = " ".join(words[:2])
    rest = " ".join(words[2:]) if len(words) > 2 else None
    return [topic] + ([rest] if rest else [])

def build_topic_index(columns):
    idx = {}
    for c in columns:
        if c == "שם רשות":
            continue
        parts = smart_split(c)
        if not parts:
            continue
        topic = parts[0]
        sub = parts[1] if len(parts) > 1 else None
        idx.setdefault(topic, {}).setdefault(sub, []).append(c)
    return idx

def columns_for_selection(topic_index, topic, subtopic):
    if not topic:
        return []
    if subtopic == "(הכול)":
        cols, seen, out = [], set(), []
        for _sub, clist in topic_index.get(topic, {}).items():
            cols.extend(clist)
        for c in cols:
            if c not in seen:
                seen.add(c)
                out.append(c)
        return out
    return topic_index.get(topic, {}).get(subtopic, [])

def format_number(x):
    if pd.isna(x):
        return "—"
    try:
        if abs(x) >= 1000:
            return f"{x:,.0f}"
        return f"{x:,.2f}"
    except Exception:
        return str(x)

# ========================
# Loaders
# ========================
@st.cache_data
def load_data(data_path: Path):
    if not data_path.exists():
        st.error("לא נמצא קובץ נתונים (negev_data.json).")
        st.stop()

    df = pd.read_json(data_path)
    df.columns = make_unique_columns(df.columns)

    name_col = None
    for c in df.columns:
        if c in POSSIBLE_NAME_COLS:
            name_col = c
            break
    if name_col is None:
        name_col = df.columns[0]

    df[name_col] = df[name_col].astype(str).str.strip()
    if name_col != "שם רשות":
        df = df.rename(columns={name_col: "שם רשות"})

    # numeric coercion (>=50% convertible)
    for c in df.columns:
        if c == "שם רשות":
            continue
        try:
            series = (
                df[c].astype(str)
                    .str.replace(",", "", regex=False)
                    .str.replace("%", "", regex=False)
                    .str.replace("\u200f", "", regex=False)
                    .str.replace("\xa0", "", regex=False)
                    .str.strip()
            )
            conv = pd.to_numeric(series, errors="coerce")
            if conv.notna().mean() >= 0.5:
                df[c] = conv
        except Exception:
            pass

    numeric_cols = [c for c in df.columns if c != "שם רשות" and pd.api.types.is_numeric_dtype(df[c])]
    return df, numeric_cols

@st.cache_data
def load_authorities(auth_path: Path):
    if not auth_path.exists():
        return []
    try:
        with open(auth_path, "r", encoding="utf-8") as f:
            lst = json.load(f)
            return [str(x).strip() for x in lst if str(x).strip()]
    except Exception:
        return []

@st.cache_data
def load_geojson(candidates):
    for p in candidates:
        if p.exists():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    return json.load(f), p.name
            except Exception:
                continue
    return None, None

@st.cache_data
def detect_geo_name_key(geojson, authority_names):
    """Try to detect which property key holds the municipal Hebrew name matching 'שם רשות'."""
    if geojson is None:
        return None
    candidates = ["שם רשות","שם הרשות","name","NAME","Muni_Heb","HEB_NAME","heb_name","muni_name","MUN_HEB"]
    for key in candidates:
        matched = 0
        for feat in geojson.get("features", []):
            val = feat.get("properties", {}).get(key)
            if val and str(val).strip() in authority_names:
                matched += 1
                if matched >= 3:
                    return key
    return None

# ========================
# Load data
# ========================
df, numeric_cols = load_data(DATA_FILE)
negev31 = load_authorities(AUTH_FILE)

df_negev = df[df["שם רשות"].isin(negev31)].copy() if negev31 else df.copy()
df_negev.columns = make_unique_columns(df_negev.columns)

topic_index = build_topic_index(df_negev.columns)
topics = sorted([t for t in topic_index.keys() if t])

# ========================
# Sidebar
# ========================
with st.sidebar:
    st.markdown('<div class="rtl"><h3>הגדרות</h3></div>', unsafe_allow_html=True)

    default_authorities = negev31[:2] if len(negev31) >= 2 else sorted(df_negev["שם רשות"].unique().tolist())[:2]
    authorities = st.multiselect(
        "בחר רשויות (עד 3):",
        options=negev31 if negev31 else sorted(df_negev["שם רשות"].unique().tolist()),
        default=default_authorities,
        max_selections=3,
    )

    if not topics:
        st.error("לא זוהו נושאים. ודא ש־negev_data.json מכיל שמות עמודות קריאים.")
        st.stop()

    topic = st.selectbox("נושא:", options=topics)
    sub_options = sorted([s for s in topic_index.get(topic, {}).keys() if s is not None])
    subtopic = st.selectbox("תת־נושא (אופציונלי):", options=["(הכול)"] + sub_options, index=0)

    topic_cols = columns_for_selection(topic_index, topic, subtopic)
    numeric_in_topic = [c for c in topic_cols if c in numeric_cols]

    selected_metric = st.selectbox(
        "בחר מדד מרכזי להשוואה:",
        options=(numeric_in_topic if numeric_in_topic else ["(אין מדדים מספריים בנושא)"]),
        index=0 if numeric_in_topic else None
    )

    # טבלת הנושא - בחירת עמודות (דיפולט עד 12)
    default_cols = topic_cols[: min(12, len(topic_cols))]
    secondary_metrics = st.multiselect(
        "עמודות להצגה בטבלת הנושא:",
        options=topic_cols,
        default=default_cols
    )

    st.markdown("---")
    st.download_button(
        "📥 הורדת כלל הנתונים (CSV)",
        data=df_negev.to_csv(index=False).encode("utf-8"),
        file_name="negev_all.csv",
        mime="text/csv",
    )

# ========================
# Tabs
# ========================
tab_overview, tab_ranking, tab_map, tab_explore = st.tabs(["סקירה", "דירוג", "מפה", "חקירה"])

# ---- סקירה ----
with tab_overview:
    st.markdown(f'<div class="rtl"><h2>סקירה — {topic}{" · "+subtopic if subtopic and subtopic!="(הכול)" else ""}</h2></div>', unsafe_allow_html=True)

    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown('<div class="kpi"><h3>מס׳ רשויות</h3><div class="val">{}</div></div>'.format(df_negev["שם רשות"].nunique()), unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="kpi"><h3>מס׳ מדדים בנושא</h3><div class="val">{}</div></div>'.format(len(topic_cols)), unsafe_allow_html=True)
    with c3:
        if isinstance(selected_metric, str) and selected_metric in df_negev.columns:
            mu = df_negev[selected_metric].dropna().mean()
            st.markdown(f'<div class="kpi"><h3>ממוצע המדד</h3><div class="val">{mu:,.2f}</div></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="kpi"><h3>ממוצע המדד</h3><div class="val">—</div></div>', unsafe_allow_html=True)
    with c4:
        import numpy as np
        try:
            total_cells = len(df_negev) * max(1, len([c for c in topic_cols if c != "שם רשות"]))
            non_missing = df_negev[topic_cols].drop(columns=["שם רשות"], errors="ignore").count().sum() if topic_cols else 0
            missing_pct = 100.0 * (1.0 - (non_missing / total_cells)) if total_cells else np.nan
        except Exception:
            missing_pct = np.nan
        if np.isfinite(missing_pct):
            st.markdown(f'<div class="kpi"><h3>חוסר נתונים~%</h3><div class="val">{missing_pct:,.1f}%</div></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="kpi"><h3>חוסר נתונים~%</h3><div class="val">—</div></div>', unsafe_allow_html=True)

    # התפלגות המדד המרכזי
    if isinstance(selected_metric, str) and selected_metric in df_negev.columns and pd.api.types.is_numeric_dtype(df_negev[selected_metric]):
        fig = px.histogram(df_negev, x=selected_metric, nbins=20, title="התפלגות המדד בכלל רשויות הנגב")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("בחר/י מדד מספרי להצגת התפלגות.")

# ---- דירוג ----
with tab_ranking:
    st.markdown(f'<div class="rtl"><h2>דירוג רשויות — {topic}</h2></div>', unsafe_allow_html=True)
    show_cols = ["שם רשות"] + list(dict.fromkeys(secondary_metrics))
    show_cols = [c for c in show_cols if c in df_negev.columns]
    if len(show_cols) <= 1:
        st.warning("בחר/י עמודות להצגה בטבלה מתוך הנושא/תת־הנושא.")
    else:
        table = df_negev[show_cols].copy()
        st.dataframe(table, use_container_width=True)
        st.download_button(
            "📥 הורדת טבלת הנושא (CSV)",
            data=table.to_csv(index=False).encode("utf-8"),
            file_name=f"negev_topic_{topic.replace(' ','_')}.csv",
            mime="text/csv"
        )

    # בר השוואה מהירה
    if authorities and isinstance(selected_metric, str) and selected_metric in df_negev.columns:
        cmp = df_negev[df_negev["שם רשות"].isin(authorities)][["שם רשות", selected_metric]].dropna()
        if not cmp.empty:
            fig2 = px.bar(cmp, x="שם רשות", y=selected_metric, text=selected_metric,
                          title=f"השוואת {selected_metric} לרשויות שנבחרו")
            fig2.update_traces(texttemplate="%{text:.2f}", textposition="outside")
            fig2.update_layout(xaxis_title="", yaxis_title="")
            st.plotly_chart(fig2, use_container_width=True)

# ---- מפה ----
with tab_map:
    st.markdown(f'<div class="rtl"><h2>מפה – {topic}{" · "+subtopic if subtopic and subtopic!="(הכול)" else ""}</h2></div>', unsafe_allow_html=True)
    geojson, geo_name = load_geojson(GEO_FILE_CANDIDATES)
    if geojson is None:
        st.info("לא נמצא קובץ GeoJSON (לדוגמה: negev_geo.json). העלה/י קובץ מפה כדי לאפשר כלורופלת.")
    elif not (isinstance(selected_metric, str) and selected_metric in df_negev.columns and pd.api.types.is_numeric_dtype(df_negev[selected_metric])):
        st.info("בחר/י מדד מספרי לניגון על המפה.")
    else:
        key = detect_geo_name_key(geojson, set(df_negev['שם רשות'].unique()))
        if key is None:
            st.warning("לא הצלחתי להתאים בין שמות הרשויות בקובץ הנתונים לתכונות ה-GeoJSON. ודא/י שקיים שדה תכונה תואם (למשל 'שם רשות').")
        else:
            figm = px.choropleth(
                df_negev,
                geojson=geojson,
                featureidkey=f"properties.{key}",
                locations="שם רשות",
                color=selected_metric,
                color_continuous_scale="Viridis",
                title=f"מפה – {selected_metric}"
            )
            figm.update_geos(fitbounds="locations", visible=False)
            st.plotly_chart(figm, use_container_width=True)
            st.caption(f"עיבוד מפה לפי {geo_name}, התאמה לשדה תכונה: {key}")

# ---- חקירה ----
with tab_explore:
    st.markdown(f'<div class="rtl"><h2>חקירת קשרים — {topic}</h2></div>', unsafe_allow_html=True)
    numeric_in_topic = [c for c in topic_cols if c in numeric_cols]
    if len(numeric_in_topic) >= 2:
        c1, c2 = st.columns(2)
        x_col = c1.selectbox("ציר X (מספרי):", options=numeric_in_topic, index=0)
        y_col = c2.selectbox("ציר Y (מספרי):", options=[c for c in numeric_in_topic if c != x_col], index=0)
        scat = df_negev[["שם רשות", x_col, y_col]].dropna()
        if not scat.empty:
            fig = px.scatter(scat, x=x_col, y=y_col, text="שם רשות", trendline=None,
                             title=f"פיזור: {y_col} ~ {x_col}")
            fig.update_traces(textposition="top center")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("לבחינת קשרים יש צורך בלפחות שני מדדים מספריים באותו נושא.")
