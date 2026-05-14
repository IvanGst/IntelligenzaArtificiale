from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from DatoStudente import DatoStudente
from StudentPredictor import StudentPredictor

# 1. Dati
dataset = DatoStudente("Student_Performance.csv")
dataset.prepare()

# 2. Modello Random Forest
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(dataset.X_train, dataset.y_train)

# 3. Valutazione
y_pred = model.predict(dataset.X_test)
print(f"MAE: {mean_absolute_error(dataset.y_test, y_pred):.2f}")
print(f"R²:  {r2_score(dataset.y_test, y_pred):.4f}")

# 4. Feature importance (bonus esclusivo della Random Forest)
features = ["Hours Studied", "Previous Scores", "Extracurricular Activities", "Sleep Hours", "Sample Question Papers Practiced"]
importances = model.feature_importances_
for i in range(len(features)):
    print(features[i] + ": " + str(round(importances[i], 4)))

# 5. Predizione
predictor = StudentPredictor(model, dataset)
predictor.predict(
    hours_studied=6,
    prev_scores=75,
    extracurricular="Yes",
    sleep_hours=7,
    papers=3
)