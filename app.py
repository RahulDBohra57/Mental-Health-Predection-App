# ============================================================
# Mental Health Cluster Insight Tool (Clinical-Grade)
# ============================================================

import streamlit as st
import pandas as pd
import joblib
from prince import MCA
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from io import BytesIO
from datetime import datetime

# ============================================================
# Load Artifacts & Dataset
# ============================================================

@st.cache_resource
def load_artifacts():
    mca = joblib.load("mca_transformer.joblib")
    cluster_models = joblib.load("risk_band_cluster_models.joblib")
    severity_map = joblib.load("severity_map.joblib")
    data = pd.read_csv("Mental Health Dataset.csv")
    return mca, cluster_models, severity_map, data

mca, cluster_models, severity_map, data = load_artifacts()

# ============================================================
# Feature Definitions
# ============================================================

features = [
    "family_history",
    "treatment",
    "Growing_Stress",
    "Changes_Habits",
    "Mood_Swings",
    "Coping_Struggles",
    "Work_Interest",
    "Social_Weakness"
]

core_symptoms = [
    "Growing_Stress",
    "Changes_Habits",
    "Mood_Swings",
    "Coping_Struggles"
]

functional_impact = [
    "Work_Interest",
    "Social_Weakness"
]

# ============================================================
# UI Labels
# ============================================================

friendly_labels = {
    "family_history": "Does your family have a history of mental health concerns?",
    "treatment": "Are you currently undergoing any mental health treatment?",
    "Growing_Stress": "Do you feel your stress levels have been increasing recently?",
    "Changes_Habits": "Have you noticed recent changes in your habits (sleep, diet, routines)?",
    "Mood_Swings": "Do you experience noticeable mood swings?",
    "Coping_Struggles": "Do you struggle to cope with everyday challenges?",
    "Work_Interest": "How interested are you in your work lately?",
    "Social_Weakness": "Do you feel socially withdrawn or less engaged than usual?"
}

# ============================================================
# Severity & Risk Logic
# ============================================================

def calculate_mdi(user_input):
    score = 0

    for col in core_symptoms + functional_impact:
        score += severity_map.get(user_input.get(col), 0)

    # Reverse work interest (low interest = higher distress)
    score += (3 - severity_map.get(user_input.get("Work_Interest"), 0))

    return score

def assign_risk_band(mdi):
    if mdi >= 10:
        return "High"
    elif mdi >= 6:
        return "Moderate"
    else:
        return "Low"

# ============================================================
# Diagnosis & Suggestions
# ============================================================

diagnosis_map = {
    "Low": "Your responses suggest stable emotional well-being with healthy coping patterns.",
    "Moderate": "Your responses indicate ongoing stress that may be affecting daily balance and energy.",
    "High": "Your responses reflect significant emotional strain that may be overwhelming your current coping capacity."
}

suggestions_map = {
    "Low": [
        "Maintain consistent sleep and daily routines.",
        "Continue activities that help you relax or feel fulfilled.",
        "Stay socially connected with people you trust.",
        "Practice occasional self-reflection or journaling.",
        "Keep clear boundaries between work and personal time.",
        "Monitor stress levels and respond early when they rise."
    ],
    "Moderate": [
        "Break daily tasks into smaller, manageable steps.",
        "Schedule at least one restorative break each day.",
        "Reduce non-essential commitments temporarily.",
        "Engage in light physical activity like walking or stretching.",
        "Practice slow breathing or grounding exercises.",
        "Talk openly with a trusted friend or family member.",
        "Re-establish consistent sleep and meal routines."
    ],
    "High": [
        "Prioritize rest and reduce mental overload wherever possible.",
        "Seek support from a trusted person instead of coping alone.",
        "Use grounding techniques such as slow breathing or sensory focus.",
        "Avoid making major decisions while feeling emotionally overwhelmed.",
        "Create predictable daily structure using small routines.",
        "Limit exposure to unnecessary stressors such as excessive news or social media.",
        "Consider reaching out to a mental health professional for support.",
        "Spend time in calming environments like nature or quiet spaces."
    ]
}

# ============================================================
# Cluster Prediction
# ============================================================

def predict_cluster(user_input):
    mdi = calculate_mdi(user_input)
    risk_band = assign_risk_band(mdi)

    df = pd.DataFrame([user_input]).astype(str)
    X_mca = mca.transform(df[core_symptoms + functional_impact])

    X_weighted = X_mca.copy()
    X_weighted.iloc[:, :3] = X_weighted.iloc[:, :3] * 2

    model = cluster_models[risk_band]
    cluster_id = int(model.predict(X_weighted)[0])

    return mdi, risk_band, cluster_id

# ============================================================
# PDF Generator
# ============================================================

def generate_pdf(user_name, risk_band, diagnosis, suggestions):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    y = 750

    c.setFont("Helvetica-Bold", 20)
    c.drawString(50, y, "Mental Well-Being Report")
    y -= 40

    c.setFont("Helvetica", 12)
    if user_name:
        c.drawString(50, y, f"Prepared For: {user_name}")
        y -= 20

    now = datetime.now().strftime("%d/%m/%Y, %I:%M %p")
    c.drawString(50, y, f"Date Generated: {now}")
    y -= 30

    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y, f"Risk Level: {risk_band}")
    y -= 20

    c.setFont("Helvetica", 12)
    c.drawString(50, y, diagnosis)
    y -= 30

    c.setFont("Helvetica-Bold", 13)
    c.drawString(50, y, "Recommended Activities")
    y -= 20

    c.setFont("Helvetica", 11)
    for s in suggestions:
        c.drawString(60, y, f"- {s}")
        y -= 15

    c.setFont("Helvetica", 9)
    c.drawString(
        50, 40,
        "Disclaimer: This report provides general well-being insights and is not a medical diagnosis."
    )

    c.save()
    buffer.seek(0)
    return buffer

# ============================================================
# STREAMLIT UI
# ============================================================

st.markdown("<h1 style='text-align:center;'>Mental Health Cluster Insight Tool</h1>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

user_name = st.text_input("Enter your name (optional)")

user_input = {}
for feature in features:
    options = sorted(data[feature].dropna().unique().tolist())
    user_input[feature] = st.selectbox(friendly_labels[feature], options)

if st.button("Generate My Well-Being Insights"):
    mdi, risk_band, cluster_id = predict_cluster(user_input)

    color_map = {
        "Low": "#2ecc71",
        "Moderate": "#f1c40f",
        "High": "#e74c3c"
    }

    st.markdown(
        f"""
        <div style="padding:12px;border-radius:6px;
        background-color:{color_map[risk_band]};
        color:black;font-weight:bold;">
        Risk Level: {risk_band}
        </div>
        """,
        unsafe_allow_html=True
    )

    st.subheader("Assessment Summary")
    st.write(diagnosis_map[risk_band])

    st.subheader("Suggested Activities")
    for s in suggestions_map[risk_band]:
        st.write(f"• {s}")

    pdf = generate_pdf(
        user_name,
        risk_band,
        diagnosis_map[risk_band],
        suggestions_map[risk_band]
    )

    st.download_button(
        "Download My Wellness Report (PDF)",
        data=pdf,
        file_name="Wellness_Report.pdf",
        mime="application/pdf"
    )

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    "<small><b>Disclaimer:</b> This tool does not provide medical advice or diagnosis. "
    "If you are experiencing emotional distress, please consult a qualified mental health professional.</small>",
    unsafe_allow_html=True
)
