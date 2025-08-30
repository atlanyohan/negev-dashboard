# -*- coding: utf-8 -*-
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

# ---------- הגדרות דף ----------
st.set_page_config(page_title="דשבורד נתוני נגב (נושאים)", layout="wide")

# ---------- עיצוב RTL ----------
st.markdown(
    """
    <style>
      .rtl { direction: rtl; text-align: right; }
      .rtl .stMarkdown p { direction: rtl; text-align: right; }
      .stSelectbox > div > div { direction: rtl; text-align: right; }
      .stMultiSelect > div > div { direction: rtl; text-align: right; }
      .stTextInput > div > div input { direction: rtl; text-align: right; }
      .stDataFrame, .stTable { direction: rtl; text-align: right; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="rtl"><h1>📊 דשבורד נתוני נגב (2019) – לפי נושאים</h1></div>', unsafe_allow_html=True)
st.markdown('<div class="rtl">בחר/י נושא (ואופציונלית תת־נושא) כדי להציג את כל המדדים הרלוונטיים לכל 31 הרשויות.</div>', unsafe_allow_html=True)

# ---------- קבצים ----------
DATA_FILE = Path(__file__).parent / "negev_data.json"
AUTH_FILE = Path(__file__).parent / "negev_31_list.json"

# ---------- עזר: ייחוד שמות עמודות ----------
def make_unique_columns(cols):
    seen = {}
    out = []
    for c in cols:
        c = "" if c is None else str(c)
        if c not in seen:
            seen[c] = 1
            out.append(c)
        else:
            seen[c] += 1
            out.append(f"{c}__{seen[c]}")
    return out

# --- נושאים/תתי־נושאים לפי שם העמודה ("תחום – תת־תחום – שנה") ---
DELIM = "–"

def split_parts(col):
    return [p.strip() for p in str(col).split(DELIM) if str(col) and p.strip()]

def build_topic_index(columns):
    idx = {}
    for c in columns:
        if c == "שם רשות":
            continue
        parts = split_parts(c)
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
        cols = []
        for _sub, clist in topic_index.get(topic, {}).items():
            cols.extend(clist)
        seen = set()
        out = []
        for c in cols:
            if c not in seen:
                seen.add(c)
                out.append(c)
        return out
    return topic_index.get(topic, {}).get(subtopic, [])

# ---------- טעינת נתונים ----------
@st.cache_data
def load_data(data_path: Path):
    if not data_path.exists():
        st.error(f"לא נמצא קובץ נתונים: {data_path.name}. העלה/י את negev_data.json לתיקיית האפליקציה.")
        st.stop()

    df = pd.read_json(data_path)
    df.columns = make_unique_columns(df.columns)

    # זיהוי עמודת שם רשות באופן גמיש
    possible_names = ["שם רשות", "שם הרשות", "רשות", "ישוב", "רשות מקומית"]
    name_col = None
    for c in df.columns:
        if c in possible_names:
            name_col = c
            break
    if name_col is None:
        name_col = df.columns[0]

    df[name_col] = df[name_col].astype(str).str.strip()
    if name_col != "שם רשות":
        df = df.rename(columns={name_col: "שם רשות"})
        name_col = "שם רשות"

    # המרה נומרית
    for c in df.columns:
        if c == "שם רשות":
            continue
        try:
            series = (
                df[c]
                .astype(str)
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
        import json
        with open(auth_path, "r", encoding="utf-8") as f:
            lst = json.load(f)
            return [str(x).strip() for x in lst if str(x).strip()]
    except Exception:
        return []

df, numeric_cols = load_data(DATA_FILE)
negev31 = load_authorities(AUTH_FILE)

df_negev = df[df["שם רשות"].isin(negev31)].copy() if negev31 else df.copy()
df_negev.columns = make_unique_columns(df_negev.columns)

topic_index = build_topic_index(df_negev.columns)
topics = sorted([t for t in topic_index.keys() if t])

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown('<div class="rtl"><h3>הגדרות</h3></div>', unsafe_allow_html=True)

    default_authorities = negev31[:2] if len(negev31) >= 2 else sorted(df_negev["שם רשות"].unique().tolist())[:2]
    authorities = st.multiselect(
        "בחר רשויות (עד 3):",
        options=negev31 if negev31 else sorted(df_negev["שם רשות"].unique().tolist())[:31],
        default=default_authorities,
        max_selections=3,
    )

    st.markdown("**בחר נושא להצגה**")
    if not topics:
        st.error("לא נמצאו נושאים בטבלת הנתונים (ודא ששמות העמודות בפורמט 'נושא – ...').")
        st.stop()

    topic = st.selectbox("נושא:", options=topics)

    sub_options = sorted([s for s in topic_index.get(topic, {}).keys() if s is not None])
    subtopic = st.selectbox("תת־נושא (אופציונלי):", options=["(הכול)"] + sub_options, index=0)

    topic_cols = columns_for_selection(topic_index, topic, subtopic)
    numeric_in_topic = [c for c in topic_cols if c in numeric_cols]

    selected_metric = st.selectbox(
        "בחר מדד להשוואה (מתוך הנושא):",
        options=(numeric_in_topic if numeric_in_topic else ["(אין מדדים מספריים בנושא)"])
    )
    secondary_metrics = st.multiselect(
        "עמודות לטבלה (מתוך הנושא):",
        options=topic_cols,
        default=topic_cols[: min(8, len(topic_cols))]
    )

    st.markdown("---")
    st.download_button(
        "📥 הורדת כל הנתונים (CSV)",
        data=df_negev.to_csv(index=False).encode("utf-8"),
        file_name="negev_all.csv",
        mime="text/csv",
    )

# ---------- KPI ----------
st.markdown('<div class="rtl"><h2>מדדים מרכזיים</h2></div>', unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)

def kpi(col, title, series):
    with col:
        try:
            val = float(series.dropna().mean())
            st.metric(label=title, value=f"{val:,.2f}")
        except Exception:
            st.metric(label=title, value="—")

if isinstance(selected_metric, str) and selected_metric in df_negev.columns:
    kpi(col1, f"ממוצע נגב – {selected_metric}", df_negev[selected_metric])
else:
    st.info("בחר/י מדד מספרי מתוך הנושא להצגת KPI וגרף.")

# ---------- גרף השוואה ----------
st.markdown('<div class="rtl"><h2>השוואת רשויות</h2></div>', unsafe_allow_html=True)
if authorities and isinstance(selected_metric, str) and selected_metric in df_negev.columns:
    cmp = df_negev[df_negev["שם רשות"].isin(authorities)][["שם רשות", selected_metric]].dropna()
    cmp.columns = make_unique_columns(cmp.columns)
    if not cmp.empty:
        fig = px.bar(cmp, x="שם רשות", y=selected_metric, text=selected_metric, title=f"השוואת {selected_metric}")
        fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
        fig.update_layout(xaxis_title="", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("אין נתונים להשוואה עבור המדד שנבחר.")
else:
    st.info("בחר/י עד 3 רשויות להשוואה ובחר/י מדד מספרי.")

# ---------- טבלת הנושא ----------
st.markdown('<div class="rtl"><h2>טבלת הנושא לכל הרשויות</h2></div>', unsafe_allow_html=True)

requested_cols = ["שם רשות"] + list(dict.fromkeys(secondary_metrics))
available_cols = [c for c in requested_cols if c in df_negev.columns]

if not available_cols or len(available_cols) == 1:
    st.warning("לא נמצאו עמודות להצגה עבור הנושא/תת־הנושא שנבחרו.")
else:
    table = df_negev[available_cols].copy()
    table.columns = make_unique_columns(table.columns)
    st.dataframe(table, use_container_width=True)
    st.download_button(
        "📥 הורדת הטבלה (CSV)",
        data=table.to_csv(index=False).encode("utf-8"),
        file_name=f"negev_{topic.replace(' ','_')}.csv",
        mime="text/csv",
    )

st.markdown('<div class="rtl"><small>עדכון נתונים: 2019 (אלא אם צוין אחרת), מבוסס על קובץ המקור שסיפקת.</small></div>', unsafe_allow_html=True)
