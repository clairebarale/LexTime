import os
import pandas as pd
import argparse


class Evaluator:
    def __init__(self, base_folder: str):
        self.base_folder = base_folder

    def evaluate_model(self, model_name: str) -> float:
        model_folder = os.path.join(self.base_folder, model_name)
        print(f"\nEvaluating model: {model_name}")

        if not os.path.exists(model_folder):
            print(f"Folder does not exist: {model_folder}")
            return None

        accuracies = []
        for file_name in os.listdir(model_folder):
            file_path = os.path.join(model_folder, file_name)
            try:
                df = pd.read_csv(file_path, usecols=["label", "response"]).dropna()
            except Exception as e:
                print(f"  Skipping {file_name} (error: {e})")
                continue

            # in the col label: change "entailment" to "yes" and "contradiction" to "no"
            df["label"] = df["label"].str.lower()
            df["label"] = df["label"].replace({"entailment": "yes", "contradiction": "no"})
            df["response"] = df["response"].str.lower()
            df.loc[df["response"].str.contains("yes"), "response"] = "yes"
            df.loc[df["response"].str.contains("no"), "response"] = "no"
            df["correct"] = df["label"] == df["response"]
            #df["correct"] = df["label"] == df["response"]

            accuracy = df["correct"].mean()
            print(f"  {file_name}: {accuracy:.2%}")
            accuracies.append(accuracy)

        if accuracies:
            avg_accuracy = sum(accuracies) / len(accuracies)
            print(f"Average accuracy for {model_name}: {avg_accuracy:.2%}")
            return avg_accuracy
        else:
            print(f"No valid files found for {model_name}")
            return None
        
        
        
    def evaluate_all(self, models: list[str]) -> dict:
        results = {}
        for model in models:
            accuracy = self.evaluate_model(model)
            results[model] = accuracy
        return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Temporal Reasoning Models")
    parser.add_argument("--base_folder", type=str, required=True, help="Base folder containing model output directories.")
    parser.add_argument("--models", type=str, nargs="+", required=True, help="List of model names to evaluate.")
    args = parser.parse_args()
    BASE_FOLDER = args.base_folder
    MODELS = args.models
    evaluator = Evaluator(BASE_FOLDER)
    results = evaluator.evaluate_all(MODELS)
    
    print("\nFinal Summary:")
    for model, acc in results.items():
        acc_str = f"{acc:.2%}" if acc is not None else "N/A"
        print(f"{model}: {acc_str}")
