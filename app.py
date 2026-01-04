# ============================================================
# Mental Health Cluster Insight Tool
# FINAL PRODUCTION VERSION
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

# ============================================================
# Load Artifacts
# ============================================================

@st.cache_resource
def load_artifacts():
    mca = joblib.load("mca_transformer.joblib")
    cluster_models = joblib.load("risk_band_cluster_models.joblib")
    severity_map = joblib.load("severity_map.joblib")
    return mca, cluster_models, severity_map

mca, cluster_models, severity_map = load_artifacts()

# ============================================================
# Feature Definitions (MUST MATCH TRAINING)
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
    "Coping_Struggles",
    "Work_Interest",
    "Social_Weakness"
]

# ============================================================
# UI Questions
# ============================================================

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

# ============================================================
# MDI & Risk Band
# ============================================================

def calculate_mdi(user_input):
    return sum(severity_map.get(user_input[col], 1) for col in core_symptoms)

def assign_risk_band(mdi):
    if mdi >= 8:
        return "High"
    elif mdi >= 4:
        return "Moderate"
    else:
        return "Low"

# ============================================================
# Text Content
# ============================================================

diagnosis_map = {
    "Low": "Your responses suggest stable emotional well-being with healthy coping patterns.",
    "Moderate": "Your responses indicate ongoing stress that may be affecting balance and daily functioning.",
    "High": "Your responses reflect significant emotional strain that may be overwhelming your current coping capacity."
}

meaning_map = {
    "Low": (
        "This suggests that current stressors are being managed effectively and "
        "no immediate intervention is indicated."
    ),
    "Moderate": (
        "This suggests rising emotional strain that may begin to interfere with daily "
        "functioning if left unaddressed."
    ),
    "High": (
        "This suggests significant emotional distress where additional support or "
        "professional guidance may be beneficial."
    )
}

suggestions_map = {
    "Low": [
        "Maintain consistent sleep and daily routines.",
        "Continue activities that help you relax or feel fulfilled.",
        "Stay socially connected with trusted people.",
        "Practice occasional self-reflection or journaling.",
        "Maintain healthy work–life boundaries.",
        "Respond early when stress levels increase."
    ],
    "Moderate": [
        "Break daily tasks into smaller, manageable steps.",
        "Schedule intentional rest or recovery time.",
        "Reduce non-essential commitments temporarily.",
        "Engage in light physical activity such as walking.",
        "Practice breathing or grounding exercises.",
        "Talk openly with a trusted person.",
        "Rebuild consistent sleep and meal routines."
    ],
    "High": [
        "Prioritize rest and reduce mental overload.",
        "Seek support instead of coping alone.",
        "Use grounding techniques like slow breathing.",
        "Avoid major decisions while overwhelmed.",
        "Create predictable daily routines.",
        "Limit unnecessary stress exposure.",
        "Consider professional mental health support.",
        "Spend time in calming environments."
    ]
}

# ============================================================
# Prediction
# ============================================================

def predict_cluster(user_input):
    mdi = calculate_mdi(user_input)
    risk_band = assign_risk_band(mdi)

    df = pd.DataFrame([user_input]).astype(str)
    X_mca = pd.DataFrame(mca.transform(df[core_symptoms])).fillna(0)
    X_mca.iloc[:, :3] *= 2

    model = cluster_models[risk_band]
    cluster_id = int(model.predict(X_mca)[0])

    return mdi, risk_band, cluster_id

# ============================================================
# PDF GENERATOR
# ============================================================

def generate_pdf(user_name, mdi, risk_band, diagnosis, meaning, suggestions):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, leftMargin=48, rightMargin=48)

    styles = getSampleStyleSheet()
    story = []

    risk_colors = {
        "Low": HexColor("#27ae60"),
        "Moderate": HexColor("#f39c12"),
        "High": HexColor("#c0392b")
    }

    # Title
    story.append(Paragraph("<b>Mental Well-Being Report</b>", styles["Title"]))
    story.append(Spacer(1, 12))

    now = datetime.now().strftime("%d/%m/%Y, %I:%M %p")
    meta = f"<b>Date Generated:</b> {now}"
    if user_name:
        meta += f"<br/><b>Prepared For:</b> {user_name}"
    story.append(Paragraph(meta, styles["Normal"]))
    story.append(Spacer(1, 16))

    # Risk Badge (TABLE-BASED)
    risk_table = Table([[f"Risk Level: {risk_band}"]], colWidths=[400], rowHeights=[34])
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

    # MDI Scale
    marker = "■□□□□" if mdi <= 3 else "□□■□□" if mdi <= 7 else "□□□□■"
    story.append(Paragraph(
        f"<para align='center'><font size='10'>Low&nbsp;&nbsp;Moderate&nbsp;&nbsp;High</font><br/>"
        f"<font size='12'>{marker}</font><br/>"
        f"<font size='9'>MDI Score: {mdi}</font></para>",
        styles["Normal"]
    ))
    story.append(Spacer(1, 20))

    # Clinical Summary
    story.append(Paragraph("<b>Clinical Summary</b>", styles["Heading2"]))
    story.append(Spacer(1, 6))
    story.append(Paragraph(diagnosis, styles["BodyText"]))
    story.append(Spacer(1, 14))

    # What This Means
    story.append(Paragraph("<b>What This Means</b>", styles["Heading2"]))
    story.append(Spacer(1, 6))
    story.append(Paragraph(meaning, styles["BodyText"]))
    story.append(Spacer(1, 20))

    # Recommendations
    story.append(Paragraph("<b>Recommended Activities</b>", styles["Heading2"]))
    story.append(Spacer(1, 8))
    bullets = [ListItem(Paragraph(s, styles["BodyText"])) for s in suggestions]
    story.append(ListFlowable(bullets, bulletType="bullet"))
    story.append(Spacer(1, 24))

    # Footer
    story.append(Paragraph(
        "<para align='center'><font size='8' color='#7f8c8d'>"
        "Generated by <b>Mental Health Cluster Insight Tool</b><br/>"
        "Data-informed mental well-being insights</font></para>",
        styles["Normal"]
    ))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "<font size='8' color='#7f8c8d'>"
        "<b>Disclaimer:</b> This report is not a medical diagnosis.</font>",
        styles["Normal"]
    ))

    doc.build(story)
    buffer.seek(0)
    return buffer

# ============================================================
# STREAMLIT UI
# ============================================================

st.title("Mental Health Cluster Insight Tool")
user_name = st.text_input("Enter your name (optional)")

user_input = {f: st.selectbox(question_config[f]["question"], question_config[f]["options"]) for f in features}

if st.button("Generate My Well-Being Insights"):
    mdi, risk_band, _ = predict_cluster(user_input)

    st.success(f"Risk Level: {risk_band}")
    st.write(diagnosis_map[risk_band])

    pdf = generate_pdf(
        user_name,
        mdi,
        risk_band,
        diagnosis_map[risk_band],
        meaning_map[risk_band],
        suggestions_map[risk_band]
    )

    st.download_button("Download Wellness Report (PDF)", pdf, "Wellness_Report.pdf")
