import argparse
import joblib

class MentalHealthPredictor:
    def __init__(self, model_path):
        """Load the saved mental health prediction model."""
        self.model = joblib.load(model_path)

    def predict(self, gad_score, phq_score, epworth_score):
        """
        Predicts the mental health condition based on input scores.

        Args:
            gad_score (int): GAD-7 score.
            phq_score (int): PHQ-9 score.
            epworth_score (int): Epworth Sleepiness Scale score.

        Returns:
            int: Predicted class.
        """
        input_data = [[gad_score, phq_score, epworth_score]]
        return self.model.predict(input_data)[0]

def main():
    parser = argparse.ArgumentParser(description="Predict mental health condition based on GAD, PHQ, and Epworth scores.")
    parser.add_argument("--gad", type=int, required=True, help="GAD-7 Score (0-21)")
    parser.add_argument("--phq", type=int, required=True, help="PHQ-9 Score (0-27)")
    parser.add_argument("--epworth", type=int, required=True, help="Epworth Sleepiness Scale Score (0-24)")
    args = parser.parse_args()
    
    # Initialize predictor
    model_path = "mental_health_model.pkl"  # Adjust the path if needed
    predictor = MentalHealthPredictor(model_path)
    
    # Run prediction
    predicted_class = predictor.predict(args.gad, args.phq, args.epworth)
    
    # Class labels
    class_labels = {
        0: "Minimal Depression",
        1: "Mild Depression",
        2: "Moderate Depression",
        3: "No Depression",
        4: "Severe Depression"
    }
    
    print(f"Predicted Class: {predicted_class} - {class_labels.get(predicted_class, 'Unknown')}")

if __name__ == "__main__":
    main()
