import pandas as pd
import numpy as np

def load_data(path):
    df = pd.read_csv(path)

    df = df.drop(columns=['student_id'])
    df = df.replace(r'^\s*$', np.nan, regex=True)
    df = df.dropna()

    return df