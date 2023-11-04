import numpy as np
import pickle
from logistic_regression import LogisticRegression
from metrics import EvaluationMetrics

# Function to generate random data
def generate_data(num_samples, num_features):
    X = np.random.randint(0, 2, size=(num_samples, num_features))  # Binary features (0 or 1)
    Y = np.random.randint(0, 2, size=(num_samples, 1))  # Binary labels (0 or 1)
    return X.T, Y.T  # Transpose to match your LogisticRegression model's expectation

# Generate a larger training dataset
num_samples = 1000  # Number of samples
num_features = 100   # Number of features
X_train, Y_train = generate_data(num_samples, num_features)

# Initialize the LogisticRegression model
model = LogisticRegression()

# Initialize parameters
dim = X_train.shape[0]  # Number of features
model.initialize_parameters(dim)

# Train the model
num_iterations = 10000
learning_rate = 0.1
model.optimize(X_train, Y_train, num_iterations, learning_rate)

# Generate a larger test dataset
num_test_samples = 200  # Number of test samples
X_test, Y_test = generate_data(num_test_samples, num_features)

# Make predictions on new data
predictions = model.predict(X_test)

# Output predictions
print("Predictions:", predictions.flatten())

# Calculate our metrics
accuracy = EvaluationMetrics.accuracy_score(Y_test.flatten(), predictions.flatten())
precision = EvaluationMetrics.precision_score(Y_test.flatten(), predictions.flatten())
recall = EvaluationMetrics.recall_score(Y_test.flatten(), predictions.flatten())
f1 = EvaluationMetrics.f1_score(Y_test.flatten(), predictions.flatten())

# Output the metrics
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

# Save the model to disk
filename = 'logistic_regression_model.sav'
with open(filename, 'wb') as file:
    pickle.dump(model, file)  # Saving the entire model
