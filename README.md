
# Negev Dashboard – Topic UI (Streamlit)

דשבורד נתוני נגב עם בחירת **נושא/תת־נושא**. מצפה לקובץ `negev_data.json` באותה תיקייה.

## הרצה
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Cloud
- New app → repo/branch → `app.py`
- אחרי עדכון נתונים: Manage app → Clear cache → Reboot

## מבנה נתונים
- עמודת מזהה: תזוהה אוטומטית ותיקרא "שם רשות".
- שמות עמודות בפורמט: `נושא – תת־נושא – ...` לצורך קיבוץ לנושאים.
