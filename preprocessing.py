from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer

def create_preprocessor(X):

    categorical_cols = ['gender', 'extracurricular_activities']
    numeric_cols = [col for col in X.columns if col not in categorical_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(drop='first'), categorical_cols),
            ('num', StandardScaler(), numeric_cols)
        ]
    )

    return preprocessor

def feature_engineering(df):
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
        df['attendance_percentage'] -
        df['backlogs']
    )

    return df