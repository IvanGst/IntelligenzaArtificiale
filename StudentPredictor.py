class StudentPredictor:
    def __init__(self, model, dataset):
        self.model = model
        self.dataset = dataset

    def predict(self, hours_studied, prev_scores, extracurricular, sleep_hours, papers):
        input_data = self.dataset.transform_input(
            hours_studied, prev_scores, extracurricular, sleep_hours, papers
        )
        risultato = self.model.predict(input_data)[0]

        # Clamp tra 0 e 100
        risultato = max(0.0, min(100.0, risultato))

        if risultato >= 80:
            categoria = "Ottimo"
        elif risultato >= 60:
            categoria = "Sufficiente"
        else:
            categoria = "Insufficiente"

        print(f"Performance Index previsto: {risultato:.1f} → {categoria}")
        return risultato