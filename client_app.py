import pandas as pd
import streamlit as st
import requests

# =========================
# CONFIG
# =========================
PLACEMENT_API = "http://127.0.0.1:8000/predict/placement"
SALARY_API = "http://127.0.0.1:8000/predict/salary"

st.set_page_config(
    page_title="Client Prediction App",
    layout="centered"
)
st.title("Student Placement Prediction (Decoupled Client)")

# =========================
# INPUT FORM
# =========================
with st.form("prediction_form"):
    gender = st.selectbox("Gender", ["Male", "Female"])
    ssc = st.number_input("SSC Percentage", 0.0, 100.0, 75.0)
    hsc = st.number_input("HSC Percentage", 0.0, 100.0, 75.0)
    degree = st.number_input("Degree Percentage", 0.0, 100.0, 75.0)
    cgpa = st.number_input("CGPA", 0.0, 10.0, 7.0)
    entrance = st.number_input("Entrance Exam Score", 0.0, 100.0, 70.0)
    tech = st.number_input("Technical Skill Score", 0.0, 100.0, 70.0)
    soft = st.number_input("Soft Skill Score", 0.0, 100.0, 70.0)
    internship = st.number_input("Internship Count", 0, 10, 1)
    projects = st.number_input("Live Projects", 0, 10, 2)
    experience = st.number_input("Work Experience (months)", 0, 60, 6)
    certifications = st.number_input("Certifications", 0, 20, 1)
    attendance = st.number_input("Attendance Percentage", 0.0, 100.0, 85.0)
    backlogs = st.number_input("Backlogs", 0, 10, 0)
    extracurricular = st.selectbox(
        "Extracurricular Activities",
        ["Yes", "No"]
    )
    submit = st.form_submit_button("Run Prediction")

# =========================
# PREDICTION LOGIC
# =========================
if submit:
    data = {
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
        "work_experience_months": experience,
        "certifications": certifications,
        "attendance_percentage": attendance,
        "backlogs": backlogs,
        "extracurricular_activities": extracurricular
    }
    try:
        placement_response = requests.post(
            PLACEMENT_API,
            json=data,
            timeout=10
        )
        salary_response = requests.post(
            SALARY_API,
            json=data,
            timeout=10
        )
        if placement_response.status_code == 200 and salary_response.status_code == 200:
            placement_result = placement_response.json()
            salary_result = salary_response.json()
            if "error" in placement_result:
                st.error(placement_result["error"])
            elif "error" in salary_result:
                st.error(salary_result["error"])
            else:
                placement_text = (
                    "Placed"
                    if placement_result["placement"] == 1
                    else "Not Placed"
                )
                st.success("Prediction Successful")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        "Placement Status",
                        placement_text
                    )
                with col2:
                    st.metric(
                        "Predicted Salary (LPA)",
                        salary_result["salary"]
                    )
        else:
            st.error("API Error — check FastAPI server")
    except requests.exceptions.ConnectionError:
        st.error(
            "Cannot connect to FastAPI. Make sure FastAPI is running."
        )
    except requests.exceptions.Timeout:
        st.error(
            "Request timeout. Server too slow."
        )