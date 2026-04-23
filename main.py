from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

# LOAD MODEL
cls_model = joblib.load("models/classification_model.pkl")
reg_model = joblib.load("models/regression_model.pkl")

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


@app.get("/")
def home():
    return {"message": "API is running"}


@app.post("/predict")
def predict(data: dict):

    df = pd.DataFrame([data])
    df = add_features(df)

    placement = int(cls_model.predict(df)[0])
    salary = float(reg_model.predict(df)[0])

    return {
        "placement": placement,
        "salary": salary
    }