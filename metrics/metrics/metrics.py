# metrics.py
import numpy as np

class EvaluationMetrics:
    @staticmethod
    def accuracy_score(y_true, y_pred):
        correct_predictions = np.sum(y_true == y_pred)
        total_predictions = len(y_true)
        return correct_predictions / total_predictions

    @staticmethod
    def precision_score(y_true, y_pred):
        true_positives = np.sum((y_true == 1) & (y_pred == 1))
        predicted_positives = np.sum(y_pred == 1)
        return true_positives / predicted_positives if predicted_positives > 0 else 0

    @staticmethod
    def recall_score(y_true, y_pred):
        true_positives = np.sum((y_true == 1) & (y_pred == 1))
        actual_positives = np.sum(y_true == 1)
        return true_positives / actual_positives if actual_positives > 0 else 0

    @staticmethod
    def f1_score(y_true, y_pred):
        precision = EvaluationMetrics.precision_score(y_true, y_pred)
        recall = EvaluationMetrics.recall_score(y_true, y_pred)
        return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
