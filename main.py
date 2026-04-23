from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# =========================
# INIT APP
# =========================
app = FastAPI(title="Student Placement Prediction API (Decoupled)")

# =========================
# LOAD MODELS
# =========================
cls_model = joblib.load("models/classification_model.pkl")
reg_model = joblib.load("models/regression_model.pkl")

# =========================
# REQUEST SCHEMA
# =========================
class StudentData(BaseModel):
    gender: str
    ssc_percentage: float
    hsc_percentage: float
    degree_percentage: float
    cgpa: float
    entrance_exam_score: float
    technical_skill_score: float
    soft_skill_score: float
    internship_count: int
    live_projects: int
    work_experience_months: int
    certifications: int
    attendance_percentage: float
    backlogs: int
    extracurricular_activities: str

# =========================
# FEATURE ENGINEERING
# =========================
def add_features(df: pd.DataFrame):
    df = df.copy()
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

# =========================
# HEALTH CHECK
# =========================
@app.get("/")
def home():
    return {
        "message": "API is running",
        "endpoints": [
            "/predict/placement",
            "/predict/salary"
        ]
    }

# =========================
# ENDPOINT 1 — PREDICT PLACEMENT
# =========================
@app.post("/predict/placement")
def predict_placement(data: StudentData):
    try:
        df = pd.DataFrame([data.dict()])
        df = add_features(df)
        placement = int(cls_model.predict(df)[0])
        return {
            "placement": placement
        }
    except Exception as e:
        return {
            "error": str(e)
        }

# =========================
# ENDPOINT 2 — PREDICT SALARY
# =========================
@app.post("/predict/salary")
def predict_salary(data: StudentData):
    try:
        df = pd.DataFrame([data.dict()])
        df = add_features(df)
        salary = float(reg_model.predict(df)[0])
        return {
            "salary": round(salary, 2)
        }
    except Exception as e:
        return {
            "error": str(e)
        }
