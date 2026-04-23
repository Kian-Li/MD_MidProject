import streamlit as st
import requests
import pandas as pd

API_URL = "http://127.0.0.1:8000/predict"

def add_features(df):
    df['academic_score'] = (
        df['ssc_percentage'] +
        df['hsc_percentage'] +
        df['degree_percentage']
    ) / 3

    df['skill_score'] = (
        df['technical_skill_score'] +
        df['soft_skill_score']
    ) / 2

    df['engagement_score'] = (
        df['internship_count'] +
        df['live_projects'] +
        df['certifications']
    )
    return df

st.title("Client App (FastAPI)")

with st.form("form"):

    gender = st.selectbox("Gender", ["Male", "Female"])
    ssc = st.number_input("SSC", 0, 100)
    hsc = st.number_input("HSC", 0, 100)
    degree = st.number_input("Degree", 0, 100)
    cgpa = st.number_input("CGPA", 0.0, 10.0)

    entrance = st.number_input("Entrance", 0, 100)
    tech = st.number_input("Tech", 0, 100)
    soft = st.number_input("Soft", 0, 100)

    internship = st.number_input("Internship", 0, 10)
    projects = st.number_input("Projects", 0, 10)
    exp = st.number_input("Experience", 0, 60)

    cert = st.number_input("Certifications", 0, 20)
    attendance = st.number_input("Attendance", 0, 100)
    backlogs = st.number_input("Backlogs", 0, 10)

    extra = st.selectbox("Extracurricular", ["Yes", "No"])

    submit = st.form_submit_button("Predict")

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
        "work_experience_months": exp,
        "certifications": cert,
        "attendance_percentage": attendance,
        "backlogs": backlogs,
        "extracurricular_activities": extra
    }

    response = requests.post(API_URL, json=data)

    result = response.json()

    st.write("Placement:", "Placed" if result["placement"] == 1 else "Not Placed")
    st.write("Salary:", result["salary"])