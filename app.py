# ============================================================
# PsycheLens-AI-Driven-Mental-Well-Being-Analytics-Tool
# ============================================================

import streamlit as st
import pandas as pd
import joblib
from io import BytesIO
from datetime import datetime

# ReportLab
from reportlab.lib.colors import HexColor
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer,
    ListFlowable, ListItem, Table, TableStyle
)
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors

# Load Artifacts

@st.cache_resource
def load_artifacts():
    mca = joblib.load("mca_transformer.joblib")
    cluster_models = joblib.load("risk_band_cluster_models.joblib")
    severity_map = joblib.load("severity_map.joblib")
    return mca, cluster_models, severity_map

mca, cluster_models, severity_map = load_artifacts()

# Feature Definitions

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
    "Coping_Struggles",
    "Work_Interest",
    "Social_Weakness"
]

# UI Questions

question_config = {
    "family_history": {
        "question": "Is there a history of mental health concerns in your family?",
        "options": ["Yes", "No"]
    },
    "treatment": {
        "question": "Are you currently receiving mental health support or treatment?",
        "options": ["Yes", "No"]
    },
    "Growing_Stress": {
        "question": "How would you describe your recent stress levels?",
        "options": ["Manageable", "Elevated", "Overwhelming"]
    },
    "Changes_Habits": {
        "question": "Have you noticed changes in sleep, appetite, or daily routines?",
        "options": ["No noticeable changes", "Some changes", "Significant changes"]
    },
    "Mood_Swings": {
        "question": "How frequently do you experience mood fluctuations?",
        "options": ["Rarely", "Sometimes", "Often"]
    },
    "Coping_Struggles": {
        "question": "How well are you coping with everyday challenges?",
        "options": ["Coping well", "Struggling at times", "Struggling most of the time"]
    },
    "Work_Interest": {
        "question": "How engaged do you feel with work or daily responsibilities?",
        "options": ["Highly engaged", "Somewhat engaged", "Not engaged"]
    },
    "Social_Weakness": {
        "question": "How socially connected do you feel compared to usual?",
        "options": ["As connected as usual", "Slightly less connected", "Much less connected"]
    }
}

# MDI & Risk Band

def calculate_mdi(user_input):
    return sum(severity_map.get(user_input[col], 1) for col in core_symptoms)

def assign_risk_band(mdi):
    if mdi >= 8:
        return "High"
    elif mdi >= 4:
        return "Moderate"
    else:
        return "Low"

# Text Content

diagnosis_map = {
    "Low": "Your responses suggest stable emotional well-being with healthy coping patterns.",
    "Moderate": "Your responses indicate ongoing stress that may be affecting balance and daily functioning.",
    "High": "Your responses reflect significant emotional strain that may be overwhelming your current coping capacity."
}

meaning_map = {
    "Low": "This suggests that current stressors are being managed effectively.",
    "Moderate": "This suggests rising emotional strain that may begin to interfere with daily functioning.",
    "High": "This suggests significant emotional distress where additional support may be beneficial."
}

suggestions_map = {
    "Low": [
        "Maintain consistent daily routines.",
        "Stay socially connected.",
        "Respond early to rising stress."
    ],
    "Moderate": [
        "Break tasks into manageable steps.",
        "Schedule intentional rest.",
        "Talk openly with someone you trust."
    ],
    "High": [
        "Prioritize rest and reduce overload.",
        "Avoid major decisions while overwhelmed.",
        "Consider professional mental health support."
    ]
}

# Prediction

def predict_cluster(user_input):
    mdi = calculate_mdi(user_input)
    risk_band = assign_risk_band(mdi)

    df = pd.DataFrame([user_input]).astype(str)
    X_mca = pd.DataFrame(mca.transform(df[core_symptoms])).fillna(0)
    X_mca.iloc[:, :3] *= 2

    model = cluster_models[risk_band]
    cluster_id = int(model.predict(X_mca)[0])

    return mdi, risk_band, cluster_id

# PDF Generator

def generate_pdf(user_name, mdi, risk_band, diagnosis, meaning, suggestions):
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        leftMargin=48,
        rightMargin=48
    )

    styles = getSampleStyleSheet()
    story = []

    risk_colors = {
        "Low": HexColor("#27ae60"),
        "Moderate": HexColor("#f39c12"),
        "High": HexColor("#c0392b")
    }

    # Title
    story.append(Paragraph("<b>Mental Health Report</b>", styles["Title"]))
    story.append(Spacer(1, 12))

    now = datetime.now().strftime("%d/%m/%Y, %I:%M %p")
    meta = f"<b>Date Generated:</b> {now}"
    if user_name:
        meta += f"<br/><b>Prepared For:</b> {user_name}"

    story.append(Paragraph(meta, styles["Normal"]))
    story.append(Spacer(1, 16))

    # Risk badge
    risk_table = Table(
        [[f"Risk Level: {risk_band}"]],
        colWidths=[400],
        rowHeights=[34]
    )

    risk_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), risk_colors[risk_band]),
        ("TEXTCOLOR", (0, 0), (-1, -1), colors.white),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 14),
    ]))

    story.append(risk_table)
    story.append(Spacer(1, 16))

    # Summary
    story.append(Paragraph("<b>Clinical Summary</b>", styles["Heading2"]))
    story.append(Spacer(1, 6))
    story.append(Paragraph(diagnosis, styles["BodyText"]))
    story.append(Spacer(1, 14))

    story.append(Paragraph("<b>What This Means</b>", styles["Heading2"]))
    story.append(Spacer(1, 6))
    story.append(Paragraph(meaning, styles["BodyText"]))
    story.append(Spacer(1, 16))

    # Recommendations
    story.append(Paragraph("<b>Recommended Activities</b>", styles["Heading2"]))
    story.append(Spacer(1, 8))

    bullets = [
        ListItem(Paragraph(item, styles["BodyText"]))
        for item in suggestions
    ]

    story.append(ListFlowable(bullets, bulletType="bullet"))
    story.append(Spacer(1, 20))

    # Disclaimer
    story.append(Spacer(1, 20))
    story.append(Paragraph(
        "<font size='8' color='#7f8c8d'><b>Disclaimer:</b> "
        "This report is for informational purposes only and "
        "is not a medical diagnosis. Please consult a qualified "
        "mental health professional if you need support.</font>",
        styles["Normal"]
    ))

    doc.build(story)
    buffer.seek(0)
    return buffer


# STREAMLIT UI

st.title("PsycheLens - AI Driven Mental Well Being Analytics Tool")

user_name = st.text_input("Enter your name (optional)")

user_input = {
    f: st.selectbox(
        question_config[f]["question"],
        question_config[f]["options"]
    )
    for f in features
}

if st.button("Generate My Well-Being Insights"):
    mdi, risk_band, _ = predict_cluster(user_input)

    # Risk color mapping
    risk_ui = {
        "Low": ("#27ae60", "Risk Level: Low"),
        "Moderate": ("#f39c12", "Risk Level: Moderate"),
        "High": ("#c0392b", "Risk Level: High")
    }

    color, label = risk_ui[risk_band]

    st.markdown(
        f"""
        <div style="
            background-color:{color};
            padding:12px;
            border-radius:8px;
            color:white;
            font-weight:bold;
            text-align:left;
            font-size:16px;">
            {label}
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")
    st.write(diagnosis_map[risk_band])

    pdf = generate_pdf(
        user_name,
        mdi,
        risk_band,
        diagnosis_map[risk_band],
        meaning_map[risk_band],
        suggestions_map[risk_band]
    )

    st.download_button(
        "Download Wellness Report (PDF)",
        pdf,
        "Wellness_Report.pdf"
    )

# ============================================================
# DISCLAIMER
# ============================================================

st.markdown("---")
st.markdown(
    """
    <p style="font-size:14px;color:#7f8c8d;text-align:left;">
    <b>Disclaimer:</b> This app is just for project use and should not be considered as an
    alternative for professional help. If you are facing any kind of mental health symptoms,
    consult a professional.
    </p>
    """,
    unsafe_allow_html=True
)
