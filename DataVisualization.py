import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class DataVisualization:

    def __init__(self, filepath, model, dataset):
        self.df = pd.read_csv(filepath)
        self.df["Extracurricular Activities"] = self.df["Extracurricular Activities"].map({"Yes": 1, "No": 0})
        self.model = model
        self.dataset = dataset

    # Grafico 1: distribuzione del Performance Index
    def plot_distribution(self):
        plt.figure(figsize=(8, 5))
        sns.histplot(self.df["Performance Index"], bins=20, color="steelblue")
        plt.title("Distribuzione del Performance Index")
        plt.xlabel("Performance Index")
        plt.ylabel("Numero studenti")
        plt.show()

    # Grafico 2: correlazione tra ogni feature e il voto
    def plot_correlations(self):
        features = ["Hours Studied", "Previous Scores", "Extracurricular Activities",
                    "Sleep Hours", "Sample Question Papers Practiced"]

        for feature in features:
            plt.figure(figsize=(6, 4))
            sns.scatterplot(x=self.df[feature], y=self.df["Performance Index"], color="steelblue")
            plt.title("Correlazione: " + feature + " vs Performance Index")
            plt.xlabel(feature)
            plt.ylabel("Performance Index")
            plt.show()

    # Grafico 3: heatmap della correlazione tra tutte le colonne
    def plot_heatmap(self):
        plt.figure(figsize=(8, 6))
        sns.heatmap(self.df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Heatmap correlazioni")
        plt.show()

    # Grafico 4: importanza delle features
    def plot_feature_importance(self):
        features = ["Hours Studied", "Previous Scores", "Extracurricular Activities",
                    "Sleep Hours", "Sample Question Papers Practiced"]
        importances = self.model.feature_importances_

        plt.figure(figsize=(8, 5))
        sns.barplot(x=importances, y=features, color="steelblue")
        plt.title("Importanza delle Features")
        plt.xlabel("Importanza")
        plt.ylabel("Feature")
        plt.show()

    # Grafico 5: valori reali vs valori predetti
    def plot_real_vs_predicted(self):
        y_pred = self.model.predict(self.dataset.X_test)
        y_real = self.dataset.y_test

        plt.figure(figsize=(7, 5))
        sns.scatterplot(x=y_real, y=y_pred, color="steelblue")
        plt.plot([0, 100], [0, 100], color="red", linestyle="--")  # linea perfetta
        plt.title("Valori Reali vs Valori Predetti")
        plt.xlabel("Reale")
        plt.ylabel("Predetto")
        plt.show()