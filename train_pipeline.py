import pandas as pd
import joblib
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, r2_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


# =========================
# 1. LOAD DATA
# =========================
def load_data(path="B.csv"):
    df = pd.read_csv(path)
    return df

df = load_data()

# =========================
# 2. FEATURE ENGINEERING (SAFE)
# =========================
def add_features(df):
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


df = add_features(df)


# =========================
# 3. DROP ID COLUMN (FIX UTAMA)
# =========================
if "student_id" in df.columns:
    df = df.drop(columns=["student_id"])


# =========================
# 4. SPLIT DATA
# =========================
X = df.drop(['placement_status', 'salary_package_lpa'], axis=1)
y_cls = df['placement_status']
y_reg = df['salary_package_lpa']


X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(
    X, y_cls,
    test_size=0.2,
    random_state=42,
    stratify=y_cls
)

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X, y_reg,
    test_size=0.2,
    random_state=42
)


# =========================
# 5. PREPROCESSING
# =========================
categorical_cols = ['gender', 'extracurricular_activities']

numerical_cols = [
    col for col in X.columns
    if col not in categorical_cols
]

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
])


# =========================
# 6. MLFLOW SETUP
# =========================
mlflow.set_experiment("student-placement-all-in-one")


# =========================
# 7. CLASSIFICATION MODEL
# =========================
with mlflow.start_run(run_name="RF_Classification"):

    cls_model = Pipeline([
        ('preprocess', preprocessor),
        ('model', RandomForestClassifier(
            random_state=42,
            class_weight='balanced'
        ))
    ])

    cls_model.fit(X_train_cls, y_train_cls)
    pred_cls = cls_model.predict(X_test_cls)

    acc = accuracy_score(y_test_cls, pred_cls)

    mlflow.log_param("model", "RandomForestClassifier")
    mlflow.log_metric("accuracy", acc)

    mlflow.sklearn.log_model(cls_model, "classification_model")
    joblib.dump(cls_model, "models/classification_model.pkl")

    print("Classification Accuracy:", acc)


# =========================
# 8. REGRESSION MODEL
# =========================
with mlflow.start_run(run_name="RF_Regression"):

    reg_model = Pipeline([
        ('preprocess', preprocessor),
        ('model', RandomForestRegressor(random_state=42))
    ])

    reg_model.fit(X_train_reg, y_train_reg)
    pred_reg = reg_model.predict(X_test_reg)

    r2 = r2_score(y_test_reg, pred_reg)

    mlflow.log_param("model", "RandomForestRegressor")
    mlflow.log_metric("r2_score", r2)

    mlflow.sklearn.log_model(reg_model, "regression_model")
    joblib.dump(reg_model, "models/regression_model.pkl")

    print("Regression R2:", r2)


print("Training selesai")
print("Model aman untuk deployment 🚀")