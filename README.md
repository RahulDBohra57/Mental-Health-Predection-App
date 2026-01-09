App Link: https://your-mental-health-predection-app.streamlit.app/

# 🧠 PsycheLens: AI Driven Mental Well-Being Analytics Tool

A **clinical-grade, ethical, and explainable mental well-being assessment tool** built using unsupervised machine learning and rule-based severity logic. The application helps users understand **patterns** in their emotional well-being — **not** diagnose medical conditions.

---

## 📌 Project Objective

Traditional clustering models often fail in mental-health contexts because they **average emotional distress**, leading to false reassurance (e.g., severe cases classified as “mild stress”).

This project solves that problem by using a **hybrid approach**:

1. **Symptom severity scoring** (Mental Distress Index – MDI)
2. **Risk band assignment** (Low / Moderate / High)
3. **Clustering within risk bands only**

This ensures that **high-distress users are never mixed with low-distress profiles**, making the system safer, more interpretable, and clinically defensible.

---

## 🧩 Key Features

- 🧠 Mental Distress Index (MDI) for severity awareness
- 🛡 Risk-band first modeling (clinical safety layer)
- 📊 MCA-based clustering for categorical data
- 📄 Downloadable PDF well-being report
- 🧑‍⚕️ Interview & academic-ready architecture

---

## 📊 Dataset Overview

The dataset consists of **self-reported mental health indicators**, all categorical in nature.

### Feature Groups

#### 1. Contextual Variables

- `family_history`
- `treatment`

#### 2. Core Symptom Indicators

- `Growing_Stress`
- `Changes_Habits`
- `Mood_Swings`
- `Coping_Struggles`

#### 3. Functional Impact Indicators

- `Work_Interest`
- `Social_Weakness`

---

## 🛡 Risk Banding (Clinical Safety Layer)

Before clustering, each user is assigned a **risk band** based on MDI:

| Risk Band | Interpretation                         |
| --------- | -------------------------------------- |
| Low       | Emotionally stable / mild stress       |
| Moderate  | Emerging or persistent stress patterns |
| High      | Significant emotional distress         |

This prevents **severe cases from being averaged into mild clusters**.

---

## 🧪 Algorithms Used

- **Multiple Correspondence Analysis (MCA)**

  - For dimensionality reduction on categorical data

- **KMeans Clustering (per risk band)**

  - Applied only within Low / Moderate / High bands
  - Prevents centroid dilution

- **Rule-based Severity Logic**
  - Ensures ethical and safe outputs

---

## 🖥 Application (Streamlit)

The Streamlit app allows users to:

- Answer 8 well-being questions
- Receive:
  - Risk level (Low / Moderate / High)
  - Cluster-based emotional pattern
  - Supportive interpretation
- Download a **PDF wellness report**

---

## 👤 Author Notes

This project demonstrates how **responsible AI design** is essential when applying machine learning to sensitive domains like mental health.

> Accuracy is important — **safety is non-negotiable**.
