# -*- coding: utf-8 -*-
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ========================
# הגדרות כלליות + RTL
# ========================
st.set_page_config(page_title="דשבורד נגב – נושאים", layout="wide")

st.markdown(
    """
    <style>
      .rtl { direction: rtl; text-align: right; }
      .stApp, .rtl .stMarkdown p { direction: rtl; text-align: right; }
      .stSelectbox > div > div, .stMultiSelect > div > div, .stTextInput > div > div input { direction: rtl; text-align: right; }
      .stDataFrame, .stTable { direction: rtl; text-align: right; }
      /* KPI cards look */
      .metric-row { margin-bottom: 6px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="rtl"><h1>📊 דשבורד נגב (2019) — סקירה · דירוג · חקירה</h1></div>', unsafe_allow_html=True)

# ========================
# קובצי נתונים
# ========================
DATA_FILE = Path(__file__).parent / "negev_data.json"       # הקובץ המעובד
AUTH_FILE = Path(__file__).parent / "negev_31_list.json"    # רשימת 31 הרשויות (אופציונלי)

# ========================
# פונקציות עזר
# ========================
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

# זיהוי שם רשות בעמודת זהות
POSSIBLE_NAME_COLS = ["שם רשות", "שם הרשות", "רשות", "ישוב", "רשות מקומית"]

# פיצול “נושא/תת־נושא” עמיד – גם בלי מפריד “–”
DELIMS = [" – ", "–", " - ", "-", ":", "|", "/", "—"]
YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
HE_WORDS = re.compile(r"[\u0590-\u05FFA-Za-z0-9]+")

def smart_split(label: str):
    s = str(label).strip()
    if not s:
        return []
    # 1) ננסה מפרידים נפוצים
    for d in DELIMS:
        if d in s:
            parts = [p.strip() for p in s.split(d) if p.strip()]
            if parts:
                return parts
    # 2) אין מפריד ברור – נחתוך לפני שנה אם קיימת
    m = YEAR_RE.search(s)
    if m:
        s = s[:m.start()].strip()
    # 3) ניקוי סוגריים/רווחים
    s = re.sub(r"\([^)]*\)", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    # 4) שתי מילים ראשונות כנושא, היתר תת־נושא
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

# ========================
# טעינת נתונים
# ========================
@st.cache_data
def load_data(data_path: Path):
    if not data_path.exists():
        st.error("לא נמצא negev_data.json בתיקיית האפליקציה.")
        st.stop()
    df = pd.read_json(data_path)
    df.columns = make_unique_columns(df.columns)

    # עמודת שם רשות
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

    # המרה נומרית “אגרסיבית מספיק”
    for c in df.columns:
        if c == "שם רשות":
            continue
        ser = (
            df[c].astype(str)
                 .str.replace(",", "", regex=False)
                 .str.replace("%", "", regex=False)
                 .str.replace("\u200f", "", regex=False)
                 .str.replace("\xa0", "", regex=False)
                 .str.strip()
        )
        conv = pd.to_numeric(ser, errors="coerce")
        if conv.notna().mean() >= 0.5:
            df[c] = conv  # נשמר כערכי float
    numeric_cols = [c for c in df.columns if c != "שם רשות" and pd.api.types.is_numeric_dtype(df[c])]
    return df, numeric_cols

@st.cache_data
def load_authorities(path: Path):
    if not path.exists():
        return []  # נשתמש בכל הרשויות שבקובץ
    try:
        import json
        with open(path, "r", encoding="utf-8") as f:
            return [str(x).strip() for x in pd.json_loads(f.read()) if str(x).strip()]
    except Exception:
        try:
            import json
            with open(path, "r", encoding="utf-8") as f:
                lst = json.load(f)
            return [str(x).strip() for x in lst if str(x).strip()]
        except Exception:
            return []

df, numeric_cols = load_data(DATA_FILE)
negev31 = load_authorities(AUTH_FILE)

# מסנן ל־31 רשויות אם יש קובץ רשימה
df_negev = df[df["שם רשות"].isin(negev31)].copy() if negev31 else df.copy()
df_negev.columns = make_unique_columns(df_negev.columns)

# בניית נושאים
topic_index = build_topic_index(df_negev.columns)
topics = sorted([t for t in topic_index.keys() if t])

# ========================
# Sidebar
# ========================
with st.sidebar:
    st.markdown('<div class="rtl"><h3>הגדרות</h3></div>', unsafe_allow_html=True)

    # בחירת רשויות להשוואה מהירה (עד 3)
    default_authorities = negev31[:2] if len(negev31) >= 2 else sorted(df_negev["שם רשות"].unique().tolist())[:2]
    authorities = st.multiselect(
        "בחר רשויות (עד 3):",
        options=negev31 if negev31 else sorted(df_negev["שם רשות"].unique().tolist()),
        default=default_authorities,
        max_selections=3,
    )

    # בחירת נושא/תת־נושא
    if not topics:
        st.error("לא זוהו נושאים. ודא ששמות העמודות קריאים; אפשר גם בלי מפריד, ההיוריסטיקה תזהה.")
        st.stop()

    topic = st.selectbox("נושא:", options=topics)
    sub_options = sorted([s for s in topic_index.get(topic, {}).keys() if s is not None])
    subtopic = st.selectbox("תת־נושא (אופציונלי):", options=["(הכול)"] + sub_options, index=0)

    # עמודות שייכות לנושא
    topic_cols = columns_for_selection(topic_index, topic, subtopic)
    numeric_in_topic = [c for c in topic_cols if c in numeric_cols]

    # מדד ייעודי לגרף
    selected_metric = st.selectbox(
        "בחר מדד מרכזי להשוואה:",
        options=(numeric_in_topic if numeric_in_topic else ["(אין מדדים מספריים בנושא)"]),
        index=0 if numeric_in_topic else None
    )

    # אילו עמודות להציג בטבלה
    default_cols = topic_cols[: min(10, len(topic_cols))]
    secondary_metrics = st.multiselect(
        "עמודות להצגה בטבלה:",
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
# Tabs: סקירה | דירוג | חקירה
# ========================
tab_overview, tab_ranking, tab_explore = st.tabs(["סקירה", "דירוג", "חקירה"])

# ---- סקירה ----
with tab_overview:
    st.markdown(f'<div class="rtl"><h2>סקירה — {topic}{" · "+subtopic if subtopic and subtopic!="(הכול)" else ""}</h2></div>', unsafe_allow_html=True)

    # KPI ממוצע + מקס/מינ לרמיזה
    if isinstance(selected_metric, str) and selected_metric in df_negev.columns:
        series = df_negev[selected_metric].dropna()
        avg = series.mean() if len(series) else np.nan
        mx = series.max() if len(series) else np.nan
        mn = series.min() if len(series) else np.nan

        c1, c2, c3 = st.columns(3)
        c1.metric(f"ממוצע נגב – {selected_metric}", f"{avg:,.2f}" if pd.notna(avg) else "—")
        c2.metric("גבוהה ביותר", f"{mx:,.2f}" if pd.notna(mx) else "—")
        c3.metric("נמוכה ביותר", f"{mn:,.2f}" if pd.notna(mn) else "—")

        # התפלגות
        if pd.api.types.is_numeric_dtype(df_negev[selected_metric]):
            fig = px.histogram(df_negev, x=selected_metric, nbins=20, title="התפלגות המדד בכלל רשויות הנגב")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("בחר/י מדד מספרי להצגת סקירה.")

# ---- דירוג ----
with tab_ranking:
    st.markdown(f'<div class="rtl"><h2>דירוג רשויות — {topic}</h2></div>', unsafe_allow_html=True)
    show_cols = ["שם רשות"] + list(dict.fromkeys(secondary_metrics))
    show_cols = [c for c in show_cols if c in df_negev.columns]
    if len(show_cols) <= 1:
        st.warning("בחר/י עמודות להצגה בטבלה מתוך הנושא/תת־נושא.")
    else:
        table = df_negev[show_cols].copy()
        st.dataframe(table, use_container_width=True)
        st.download_button(
            "📥 הורדת טבלת הנושא (CSV)",
            data=table.to_csv(index=False).encode("utf-8"),
            file_name=f"negev_topic_{topic.replace(' ','_')}.csv",
            mime="text/csv"
        )

    # גרף השוואה מהירה למדד המרכזי
    if authorities and isinstance(selected_metric, str) and selected_metric in df_negev.columns:
        cmp = df_negev[df_negev["שם רשות"].isin(authorities)][["שם רשות", selected_metric]].dropna()
        if not cmp.empty:
            fig = px.bar(cmp, x="שם רשות", y=selected_metric, text=selected_metric,
                         title=f"השוואת {selected_metric} לרשויות שנבחרו")
            fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
            fig.update_layout(xaxis_title="", yaxis_title="")
            st.plotly_chart(fig, use_container_width=True)

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
