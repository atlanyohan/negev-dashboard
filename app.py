
# -*- coding: utf-8 -*-
import json
import pandas as pd
import streamlit as st
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title="דשבורד נגב", layout="wide")

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
    """, unsafe_allow_html=True)

st.markdown('<div class="rtl"><h1>📊 דשבורד נתוני נגב (2019)</h1></div>', unsafe_allow_html=True)

@st.cache_data
def load_data():
    data_path = Path(__file__).parent / "negev_data.json"
    df = pd.read_json(data_path)
    df["שם רשות"] = df["שם רשות"].astype(str).str.strip()
    df = df[df["שם רשות"].notna() & (df["שם רשות"] != "")]
    def to_num(x):
        if pd.isna(x): return None
        if isinstance(x,(int,float)): return x
        t = str(x).strip().replace(",","").replace("\u200f","").replace("\xa0","")
        if t.endswith("%"):
            try: return float(t[:-1])
            except: return None
        try: return float(t)
        except: return None
    numeric_cols = []
    for c in df.columns:
        if c=="שם רשות": continue
        s = df[c].dropna().head(30)
        if len(s)==0: continue
        ok = 0
        for v in s:
            try: float(str(v).replace(",","")); ok+=1
            except: pass
        if ok>=max(1,int(min(30,len(s))*0.5)):
            numeric_cols.append(c)
            df[c] = df[c].map(to_num)
    return df, numeric_cols

@st.cache_data
def load_authorities():
    path = Path(__file__).parent / "negev_31_list.json"
    with open(path,"r",encoding="utf-8") as f: return json.load(f)

df, numeric_cols = load_data()
negev31 = load_authorities()

with st.sidebar:
    st.markdown('<div class="rtl"><h3>הגדרות</h3></div>', unsafe_allow_html=True)
    authorities = st.multiselect("בחר רשויות (עד 3):", options=negev31, default=negev31[:2], max_selections=3)
    keyword = st.text_input("סינון עמודות:", value="")
    metric_options = [c for c in numeric_cols if keyword in c] if keyword.strip() else numeric_cols
    selected_metric = st.selectbox("בחר מדד:", options=metric_options)
    default_secondary = metric_options[:5] if len(metric_options)>=5 else metric_options
    secondary_metrics = st.multiselect("מדדים לטבלה:", options=metric_options, default=default_secondary)
    st.download_button("📥 הורדת כל הנתונים", data=df.to_csv(index=False).encode("utf-8"),
                       file_name="negev_all.csv", mime="text/csv")

df_negev = df[df["שם רשות"].isin(negev31)].copy()

st.markdown('<div class="rtl"><h2>מדדים מרכזיים</h2></div>', unsafe_allow_html=True)
col1,col2,col3 = st.columns(3)
def kpi(col,title,series):
    with col:
        try: val=float(series.dropna().mean())
        except: val=None
        st.metric(label=title,value=("—" if val is None else f"{val:,.2f}"))
if selected_metric in df_negev.columns:
    kpi(col1,f"ממוצע נגב – {selected_metric}",df_negev[selected_metric])
    top_row=df_negev[["שם רשות",selected_metric]].dropna().sort_values(selected_metric,ascending=False).head(1)
    if len(top_row): st.table(top_row)

if authorities and selected_metric in df_negev.columns:
    cmp=df_negev[df_negev["שם רשות"].isin(authorities)][["שם רשות",selected_metric]].dropna()
    if not cmp.empty:
        fig=px.bar(cmp,x="שם רשות",y=selected_metric,text=selected_metric,title=f"השוואת {selected_metric}")
        fig.update_traces(texttemplate="%{text:.2f}",textposition="outside")
        fig.update_layout(xaxis_title="",yaxis_title="")
        st.plotly_chart(fig,use_container_width=True)

table_cols=["שם רשות"]+list(dict.fromkeys([selected_metric]+list(secondary_metrics)))
table=df_negev[table_cols].dropna(subset=[selected_metric],how='all')
st.dataframe(table,use_container_width=True)
st.download_button("📥 הורדת טבלת השוואה", data=table.to_csv(index=False).encode("utf-8"),
                   file_name="negev_compare.csv", mime="text/csv")

st.markdown('<div class="rtl"><small>עדכון נתונים: 2019 (אלא אם צוין אחרת)</small></div>', unsafe_allow_html=True)
