import streamlit as st
import joblib
import pandas as pd
from pathlib import Path

# =========================
# CONFIG UI
# =========================
st.set_page_config(
    page_title="Placement Dashboard",
    layout="wide"
)

# =========================
# LOAD MODEL
# =========================
BASE_DIR = Path(__file__).resolve().parent

cls_model = joblib.load(BASE_DIR / "models" / "classification_model.pkl")
reg_model = joblib.load(BASE_DIR / "models" / "regression_model.pkl")


# =========================
# FEATURE ENGINEERING
# =========================
def add_features(df):
    df["academic_score"] = (
        df["ssc_percentage"] +
        df["hsc_percentage"] +
        df["degree_percentage"]
    ) / 3

    df["skill_score"] = (
        df["technical_skill_score"] +
        df["soft_skill_score"]
    ) / 2

    df["engagement_score"] = (
        df["internship_count"] +
        df["live_projects"] +
        df["certifications"]
    )
    return df


# =========================
# SIDEBAR
# =========================
st.sidebar.title("Mid Project")
st.sidebar.markdown("Fill student data below:")

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
ssc = st.sidebar.slider("SSC Percentage", 0, 100, 75)
hsc = st.sidebar.slider("HSC Percentage", 0, 100, 75)
degree = st.sidebar.slider("Degree Percentage", 0, 100, 75)
cgpa = st.sidebar.slider("CGPA", 0.0, 10.0, 7.0)

entrance = st.sidebar.slider("Entrance Score", 0, 100, 70)
tech = st.sidebar.slider("Technical Skill", 0, 100, 70)
soft = st.sidebar.slider("Soft Skill", 0, 100, 70)

internship = st.sidebar.slider("Internship", 0, 10, 1)
projects = st.sidebar.slider("Projects", 0, 10, 2)
exp = st.sidebar.slider("Experience (months)", 0, 60, 6)

cert = st.sidebar.slider("Certifications", 0, 20, 1)
attendance = st.sidebar.slider("Attendance", 0, 100, 85)
backlogs = st.sidebar.slider("Backlogs", 0, 10, 0)

extra = st.sidebar.selectbox("Extracurricular", ["Yes", "No"])

predict_btn = st.sidebar.button("🚀 Run Prediction")


# =========================
# HEADER
# =========================
st.title("Student Placement Prediction Dashboard by Chelsy W.")
st.markdown("Predict placement status and expected salary using ML models.")


# =========================
# MAIN DASHBOARD
# =========================
if predict_btn:

    with st.spinner("Running model prediction..."):

        input_df = pd.DataFrame([{
            "gender": gender,
            "ssc_percentage": ssc,
            "hsc_percentage": hsc,
            "degree_percentage": degree,
            "cgpa": cgpa,
            "entrance_exam_score": entrance,
            "technical_skill_score": tech,
            "soft_skill_score": soft,
            "internship_count": internship,
            "live_projects": projects,
            "work_experience_months": exp,
            "certifications": cert,
            "attendance_percentage": attendance,
            "backlogs": backlogs,
            "extracurricular_activities": extra
        }])

        input_df = add_features(input_df)

        placement = int(cls_model.predict(input_df)[0])
        salary = float(reg_model.predict(input_df)[0])


    # =========================
    # KPI CARDS
    # =========================
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Placement Status", "Placed" if placement == 1 else "Not Placed")

    with col2:
        st.metric("Predicted Salary($)", f"{salary:.2f} LPA")

    with col3:
        st.metric("Academic Score", f"{input_df['academic_score'].values[0]:.2f}")


    st.divider()


    # =========================
    # INSIGHT SECTION
    # =========================
    st.subheader(">> Feature Insights <<")

    chart_df = pd.DataFrame({
        "Score Type": ["Academic", "Skill", "Engagement"],
        "Value": [
            input_df["academic_score"].values[0],
            input_df["skill_score"].values[0],
            input_df["engagement_score"].values[0]
        ]
    })

    st.bar_chart(chart_df.set_index("Score Type"))


    # =========================
    # RAW DATA (DEBUG STYLE)
    # =========================
    with st.expander("View Input Data"):
        st.dataframe(input_df)


# =========================
# EMPTY STATE
# =========================
else:
    st.info("<- Use the sidebar to input student data and run prediction")