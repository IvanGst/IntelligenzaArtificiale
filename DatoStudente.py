import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


class DatoStudente:
    def __init__(self, filepath):
        self.filepath = filepath
        self.X_train = self.X_test = self.y_train = self.y_test = None

    def prepare(self):
        df = pd.read_csv(self.filepath)
        df["Extracurricular Activities"] = df["Extracurricular Activities"].map({"Yes": 1, "No": 0})

        X = df[["Hours Studied", "Previous Scores", "Extracurricular Activities",
                 "Sleep Hours", "Sample Question Papers Practiced"]]
        y = df["Performance Index"]

        # Random Forest non richiede StandardScaler
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    def transform_input(self, hours_studied, prev_scores, extracurricular, sleep_hours, papers):
        extracurricular = 1 if str(extracurricular).lower() == "yes" else 0
        return pd.DataFrame([[hours_studied, prev_scores, extracurricular, sleep_hours, papers]],
                            columns=["Hours Studied", "Previous Scores", "Extracurricular Activities",
                                     "Sleep Hours", "Sample Question Papers Practiced"])