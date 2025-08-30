# -*- coding: utf-8 -*-
import json
import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

# ---------- הגדרות דף ----------
st.set_page_config(page_title="דשבורד נגב", layout="wide")

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

st.markdown('<div class="rtl"><h1>📊 דשבורד נתוני נגב (2019)</h1></div>', unsafe_allow_html=True)
st.markdown('<div class="rtl">ממשק אינפוגרפי ל־31 רשויות הנגב מתוך קובץ המקור. ניתן להשוות בין רשויות, לבחור מדדים ולייצא נתונים.</div>', unsafe_allow_html=True)

DATA_FILE = Path(__file__).parent / "negev_data.json"
AUTH_FILE = Path(__file__).parent / "negev_31_list.json"


# ---------- טעינת נתונים ----------
@st.cache_data
def load_data(data_path: Path):
    if not data_path.exists():
        st.error(f"לא נמצא קובץ נתונים: {data_path.name}. העלה את negev_data.json לתיקייה של האפליקציה.")
        st.stop()

    df = pd.read_json(data_path)
    if "שם רשות" not in df.columns:
        st.error("קובץ הנתונים חייב לכלול עמודה בשם 'שם רשות'.")
        st.stop()

    # ניקוי בסיסי
    df["שם רשות"] = df["שם רשות"].astype(str).str.strip()
    df = df[df["שם רשות"].notna() & (df["שם רשות"] != "")]

    # המרה נומרית חכמה
    def to_num(x):
        if pd.isna(x):
            return None
        if isinstance(x, (int, float)):
            return x
        t = str(x).strip().replace(",", "").replace("\u200f", "").replace("\xa0", "")
        if t.endswith("%"):
            try:
                return float(t[:-1])
            except Exception:
                return None
        try:
            return float(t)
        except Exception:
            return None

    numeric_cols = []
    for c in df.columns:
        if c == "שם רשות":
            continue
        s = df[c].dropna().head(30)
        if len(s) == 0:
            continue
        ok = 0
        for v in s:
            try:
                float(str(v).replace(",", ""))
                ok += 1
            except Exception:
                pass
        if ok >= max(1, int(min(30, len(s)) * 0.5)):
            numeric_cols.append(c)
            df[c] = df[c].map(to_num)

    return df, numeric_cols


@st.cache_data
def load_authorities(auth_path: Path):
    if not auth_path.exists():
        st.warning("לא נמצא negev_31_list.json — מוצגת רשימת רשויות ריקה. מצא/י את הקובץ והעלה/י אותו לתיקיית האפליקציה.")
        return []
    with open(auth_path, "r", encoding="utf-8") as f:
        try:
            lst = json.load(f)
            if not isinstance(lst, list):
                return []
            # ניקוי כפילויות וריקים
            cleaned = []
            for x in lst:
                x = str(x).strip()
                if x and x not in cleaned:
                    cleaned.append(x)
            return cleaned
        except Exception:
            return []


df, numeric_cols = load_data(DATA_FILE)
negev31 = load_authorities(AUTH_FILE)

# סינון ל-31 רשויות (אם קיימת הרשימה)
df_negev = df[df["שם רשות"].isin(negev31)].copy() if negev31 else df.copy()


# ---------- Sidebar ----------
with st.sidebar:
    st.markdown('<div class="rtl"><h3>הגדרות</h3></div>', unsafe_allow_html=True)

    # בחירת רשויות להשוואה
    authorities = st.multiselect(
        "בחר רשויות (עד 3):",
        options=negev31 if negev31 else sorted(df["שם רשות"].unique().tolist())[:31],
        default=(negev31[:2] if len(negev31) >= 2 else sorted(df["שם רשות"].unique().tolist())[:2]),
        max_selections=3,
    )

    # סינון עמודות לפי מילת מפתח
    keyword = st.text_input("סינון עמודות:", value="").strip()

    # קביעת אופציות למדדים
    if keyword:
        metric_options = [c for c in numeric_cols if keyword in c]
    else:
        metric_options = list(numeric_cols)

    # אם אין עמודות מספריות בכלל – נופלים חזרה לכל העמודות (מלבד שם רשות)
    if not metric_options:
        metric_options = [c for c in df.columns if c != "שם רשות"]

    if not metric_options:
        st.error("לא נמצאו עמודות מתאימות. בדוק/י את קובץ הנתונים (negev_data.json).")
        st.stop()

    selected_metric = st.selectbox("בחר מדד:", options=metric_options)
    default_secondary = metric_options[:5]
    secondary_metrics = st.multiselect("מדדים לטבלה:", options=metric_options, default=default_secondary)

    st.markdown("---")
    st.download_button(
        "📥 הורדת כל הנתונים (CSV)",
        data=df.to_csv(index=False).encode("utf-8"),
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


if selected_metric in df_negev.columns:
    kpi(col1, f"ממוצע נגב – {selected_metric}", df_negev[selected_metric])
    # הצגת השיאנית במדד
    try:
        top_row = (
            df_negev[["שם רשות", selected_metric]]
            .dropna()
            .sort_values(selected_metric, ascending=False)
            .head(1)
        )
        if len(top_row):
            st.markdown('<div class="rtl"><b>שיאנית במדד:</b></div>', unsafe_allow_html=True)
            st.table(top_row)
    except Exception:
        pass
else:
    st.info("המדד שנבחר לא קיים בנתונים. בחר/י מדד אחר או עדכן/ני את קובץ הנתונים.")


# ---------- גרף השוואה ----------
st.markdown('<div class="rtl"><h2>השוואת רשויות</h2></div>', unsafe_allow_html=True)
if authorities:
    if selected_metric in df_negev.columns:
        cmp = df_negev[df_negev["שם רשות"].isin(authorities)][["שם רשות", selected_metric]].dropna()
        if not cmp.empty:
            fig = px.bar(cmp, x="שם רשות", y=selected_metric, text=selected_metric, title=f"השוואת {selected_metric}")
            fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
            fig.update_layout(xaxis_title="", yaxis_title="")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("אין נתונים להשוואה עבור המדד שנבחר.")
    else:
        st.warning("המדד שנבחר לא קיים בנתונים.")
else:
    st.info("בחר/י עד 3 רשויות להשוואה בצד ימין.")


# ---------- טבלת השוואה ----------
st.markdown('<div class="rtl"><h2>טבלת השוואה</h2></div>', unsafe_allow_html=True)

requested_cols = ["שם רשות"] + [selected_metric] + list(secondary_metrics)
available_cols = [c for c in requested_cols if c in df_negev.columns]

if selected_metric not in df_negev.columns:
    st.error("המדד שנבחר לא קיים בעמודות הנתונים. נסה/י לבחור מדד אחר או לעדכן את קובץ הנתונים.")
else:
    if not available_cols:
        st.warning("לא נמצאו עמודות זמינות לטבלה. שנה/י את בחירת המדדים.")
    else:
        table = df_negev[available_cols].dropna(subset=[selected_metric], how="all")
        st.dataframe(table, use_container_width=True)
        st.download_button(
            "📥 הורדת טבלת השוואה (CSV)",
            data=table.to_csv(index=False).encode("utf-8"),
            file_name="negev_compare.csv",
            mime="text/csv",
        )

st.markdown('<div class="rtl"><small>עדכון נתונים: 2019 (אלא אם צוין אחרת), מבוסס על קובץ המקור שסיפקת.</small></div>', unsafe_allow_html=True)
