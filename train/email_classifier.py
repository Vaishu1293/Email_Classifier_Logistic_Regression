import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from vectorizer import TextVectorizer
from logistic_regression import LogisticRegression
from metrics import EvaluationMetrics

# Load your dataset
df_sms = pd.read_csv('../data/spam.csv', encoding='ISO-8859-1')  # Replace with your actual path

# Preprocess your data
df_sms.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
df_sms.rename(columns={'v1': 'label', 'v2': 'text'}, inplace=True)
df_sms['label'] = df_sms['label'].map({'spam': 1, 'ham': 0})

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df_sms['text'], df_sms['label'], test_size=0.2, random_state=42)

# Initialize the TextVectorizer with the training texts
vectorizer = TextVectorizer(X_train.tolist())

# Vectorize the training texts
X_train_vectors = vectorizer.vectorize()

# To vectorize the test texts, we need to modify the vectorize method to accept new texts.
# If such functionality doesn't exist, we need to add it to the TextVectorizer class.
# For the purpose of this example, we will assume vectorize can now accept new texts:
# X_test_vectors = vectorizer.vectorize(X_test.tolist())

# Alternatively, if your vectorize method cannot accept new texts, you would simply do:
# This is the correct way to vectorize the test set texts:
X_test_vectors = vectorizer.vectorize(X_test.tolist())
  # This will only work if the texts to vectorize are stored in the class instance

# Ensure the vectors are in the correct shape for training and testing
X_train_vectors = np.array(X_train_vectors).T
X_test_vectors = np.array(X_test_vectors).T  # This should also be transposed similarly to the training vectors

if X_train_vectors.ndim == 1:
    X_train_vectors = X_train_vectors.reshape(-1, 1)

if X_test_vectors.ndim == 1:
    X_test_vectors = X_test_vectors.reshape(-1, 1)

# Initialize the LogisticRegression model
model = LogisticRegression()

# Train the model
dim = X_train_vectors.shape[0]
model.initialize_parameters(dim)
model.optimize(X_train_vectors, np.array([y_train]).T, num_iterations=1, learning_rate=0.01)

# Make predictions
predictions = model.predict(X_test_vectors)

print(np.array(y_test).T)
print(predictions.flatten())

# Evaluate the model using the custom EvaluationMetrics
evaluation_metrics = EvaluationMetrics()  # If you've defined the metrics as class methods, you need to create an instance.
accuracy = evaluation_metrics.accuracy_score(np.array(y_test).T, predictions.flatten())
precision = evaluation_metrics.precision_score(np.array(y_test).T, predictions.flatten())
recall = evaluation_metrics.recall_score(np.array(y_test).T, predictions.flatten())
f1 = evaluation_metrics.f1_score(np.array(y_test).T, predictions.flatten())

# Output the metrics
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

# Save the model and vectorizer to disk
filename = 'email_classifier_model.sav'
with open(filename, 'wb') as file:
    pickle.dump(model, file)

vectorizer_filename = 'email_vectorizer.pkl'
with open(vectorizer_filename, 'wb') as file:
    pickle.dump(vectorizer, file)
