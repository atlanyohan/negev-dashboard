# -*- coding: utf-8 -*-
import json, re, io
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
ROOT = Path(__file__).parent
DATA_FILE = ROOT / "negev_data.json"        # ברירת מחדל (נטען גם CSV/Excel)
AUTH_FILE = ROOT / "negev_31_list.json"     # אופציונלי: 31 רשויות
GEO_FILE_CANDIDATES = [ROOT / "negev_geo.json", ROOT / "israel_munis.geojson", ROOT / "israel_munis.json"]

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

def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
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
    return df

def _normalize_name_col(df: pd.DataFrame) -> pd.DataFrame:
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
    return df

# ========================
# Robust loaders (file path or uploaded file)
# ========================
@st.cache_data
def load_from_bytes(data: bytes, filename: str, sheet: str|int|None):
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    df = None

    # CSV
    if ext in ("csv",):
        for enc in ("utf-8-sig", "utf-8", "cp1255"):
            try:
                df = pd.read_csv(io.BytesIO(data), encoding=enc)
                break
            except Exception:
                continue

    # Excel
    if df is None and ext in ("xlsx", "xls"):
        try:
            xl = pd.ExcelFile(io.BytesIO(data))
            use_sheet = sheet if sheet is not None else xl.sheet_names[0]
            df = xl.parse(use_sheet)
        except Exception as e:
            raise RuntimeError(f"שגיאה בקריאת Excel: {e}")

    # JSON / NDJSON
    if df is None and ext in ("json", "ndjson"):
        txt = None
        for enc in ("utf-8", "utf-8-sig"):
            try:
                txt = io.BytesIO(data).read().decode(enc)
                break
            except Exception:
                continue
        if txt is None:
            raise RuntimeError("לא הצלחתי לפענח את קובץ ה-JSON (קידוד תווים).")

        # JSON רגיל
        try:
            df = pd.read_json(io.StringIO(txt))
        except Exception:
            pass

        # NDJSON
        if df is None:
            try:
                df = pd.read_json(io.StringIO(txt), orient="records", lines=True)
            except Exception:
                pass

        # עטוף במפתח
        if df is None:
            try:
                j = json.loads(txt)
                if isinstance(j, dict):
                    for key in ("data", "records", "items", "rows"):
                        if key in j and isinstance(j[key], list):
                            df = pd.DataFrame(j[key]); break
                elif isinstance(j, list):
                    df = pd.DataFrame(j)
            except Exception:
                pass

    if df is None:
        head = (txt[:400] if isinstance(txt, str) else str(data[:200])) if "txt" in locals() else str(data[:200])
        raise RuntimeError("פורמט הנתונים לא זוהה כ-CSV/JSON/NDJSON/Excel.\nתצוגה מקדימה:\n" + head)

    df.columns = make_unique_columns(df.columns)
    df = _normalize_name_col(df)
    df = _coerce_numeric(df)
    numeric_cols = [c for c in df.columns if c != "שם רשות" and pd.api.types.is_numeric_dtype(df[c])]
    return df, numeric_cols

@st.cache_data
def load_from_repo(data_path: Path, sheet: str|int|None):
    if not data_path.exists():
        raise FileNotFoundError("לא נמצא קובץ נתונים בריפו.")
    suffix = data_path.suffix.lower()
    if suffix in (".csv",):
        for enc in ("utf-8-sig","utf-8","cp1255"):
            try:
                df = pd.read_csv(data_path, encoding=enc); break
            except Exception: continue
    elif suffix in (".xlsx",".xls"):
        try:
            df = pd.read_excel(data_path, sheet_name=sheet or 0)
        except Exception as e:
            raise RuntimeError(f"שגיאה בקריאת Excel: {e}")
    else:  # JSON/NDJSON or unknown
        df = None
        try:
            df = pd.read_json(data_path)
        except Exception:
            pass
        if df is None:
            try:
                df = pd.read_json(data_path, orient="records", lines=True)
            except Exception:
                pass
        if df is None:
            try:
                raw = data_path.read_text(encoding="utf-8")
            except Exception:
                raw = data_path.read_text(encoding="utf-8-sig")
            try:
                j = json.loads(raw)
                if isinstance(j, dict):
                    for key in ("data","records","items","rows"):
                        if key in j and isinstance(j[key], list):
                            df = pd.DataFrame(j[key]); break
                elif isinstance(j, list):
                    df = pd.DataFrame(j)
            except Exception:
                pass
        if df is None:
            head = ""
            try: head = data_path.read_text(encoding="utf-8")[:400]
            except Exception: head = "(לא ניתן לקרוא כ-UTF-8)"
            raise RuntimeError("פורמט הנתונים לא זוהה כ-CSV/JSON/NDJSON/Excel.\nתצוגה מקדימה:\n"+head)

    df.columns = make_unique_columns(df.columns)
    df = _normalize_name_col(df)
    df = _coerce_numeric(df)
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
# Sidebar: upload/choose data
# ========================
with st.sidebar:
    st.markdown('<div class="rtl"><h3>טעינת נתונים</h3></div>', unsafe_allow_html=True)
    up = st.file_uploader("העלה קובץ נתונים (CSV / JSON / NDJSON / Excel)", type=["csv","json","ndjson","xlsx","xls"])
    excel_sheet = None
    if up is not None and up.name.lower().endswith((".xlsx",".xls")):
        try:
            xl = pd.ExcelFile(up)
            excel_sheet = st.selectbox("בחר/י גיליון (Excel):", xl.sheet_names, index=0)
        except Exception:
            st.info("לא הצלחתי לקרוא את שמות הגיליונות; אנסה את הגיליון הראשון.")

# ========================
# Load data
# ========================
try:
    if up is not None:
        df, numeric_cols = load_from_bytes(up.getvalue(), up.name, excel_sheet)
    else:
        # אם אין העלאה – נטען מהקובץ שבריפו (תומך גם Excel/CSV/JSON)
        df, numeric_cols = load_from_repo(DATA_FILE, sheet=None)
except Exception as e:
    st.error(str(e))
    st.stop()

negev31 = load_authorities(AUTH_FILE)
df_negev = df[df["שם רשות"].isin(negev31)].copy() if negev31 else df.copy()
df_negev.columns = make_unique_columns(df_negev.columns)

topic_index = build_topic_index(df_negev.columns)
topics = sorted([t for t in topic_index.keys() if t])

# ========================
# Controls
# ========================
with st.sidebar:
    st.markdown('<div class="rtl"><h3>הגדרות תצוגה</h3></div>', unsafe_allow_html=True)

    default_authorities = negev31[:2] if len(negev31) >= 2 else sorted(df_negev["שם רשות"].unique().tolist())[:2]
    authorities = st.multiselect(
        "בחר רשויות (עד 3):",
        options=negev31 if negev31 else sorted(df_negev["שם רשות"].unique().tolist()),
        default=default_authorities,
        max_selections=3,
    )

    if not topics:
        st.error("לא זוהו נושאים. ודא ששמות העמודות ברורות (גם בלי מפריד – ההיוריסטיקה תזהה).")
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
    st.markdown(
        f'<div class="rtl"><h2>סקירה — {topic}{" · "+subtopic if subtopic and subtopic!="(הכול)" else ""}</h2></div>',
        unsafe_allow_html=True
    )
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="kpi"><h3>מס׳ רשויות</h3><div class="val">{df_negev["שם רשות"].nunique()}</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="kpi"><h3>מס׳ מדדים בנושא</h3><div class="val">{len(topic_cols)}</div></div>', unsafe_allow_html=True)
    with c3:
        if isinstance(selected_metric, str) and selected_metric in df_negev.columns:
            mu = df_negev[selected_metric].dropna().mean()
            st.markdown(f'<div class="kpi"><h3>ממוצע המדד</h3><div class="val">{mu:,.2f}</div></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="kpi"><h3>ממוצע המדד</h3><div class="val">—</div></div>', unsafe_allow_html=True)
    with c4:
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
    if authorities and isinstance(selected_metric, str) and selected_metric in df_negev.columns:
        cmp = df_negev[df_negev["שם רשות"].isin(authorities)][["שם רשות", selected_metric]].dropna()
        if not cmp.empty:
            fig2 = px.bar(cmp, x="שם רשות", y=selected_metric, text=selected_metric, title=f"השוואת {selected_metric} לרשויות שנבחרו")
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
                df_negev, geojson=geojson, featureidkey=f"properties.{key}",
                locations="שם רשות", color=selected_metric, color_continuous_scale="Viridis",
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
            fig = px.scatter(scat, x=x_col, y=y_col, text="שם רשות", trendline=None, title=f"פיזור: {y_col} ~ {x_col}")
            fig.update_traces(textposition="top center")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("לבחינת קשרים יש צורך בלפחות שני מדדים מספריים באותו נושא.")
