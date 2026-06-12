import numpy as np

class Evaluator:
    @staticmethod
    def evaluate_predictions(y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return {
            "RMSE": np.sqrt(np.mean((y_true - y_pred) ** 2)),
            "MAE": np.mean(np.abs(y_true - y_pred))
        }

    @staticmethod
    def calculate_map_at_k(actual_liked: set, predicted_rankings: list, k: int = 10) -> float:
        if not actual_liked: return 0.0
        score, hits = 0.0, 0.0
        for i, item in enumerate(predicted_rankings[:k]):
            if item in actual_liked:
                hits += 1.0
                score += hits / (i + 1.0)
        return score / min(len(actual_liked), k)

    @staticmethod
    def calculate_ndcg_at_k(actual_liked: set, predicted_rankings: list, k: int = 10) -> float:
        if not actual_liked: return 0.0
        dcg = sum([1.0 / np.log2(i + 2) for i, item in enumerate(predicted_rankings[:k]) if item in actual_liked])
        idcg = sum([1.0 / np.log2(j + 2) for j in range(min(k, len(actual_liked)))])
        return dcg / idcg if idcg > 0 else 0.0