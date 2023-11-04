from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle

app = Flask(__name__)

# Load serialized model and vectorizer
with open('pickle/email_classifier_model.sav', 'rb') as file:
    model = pickle.load(file)

with open('pickle/email_vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the email text from the request.
    email_text = request.form['email_text']

    # Transform the email text using the loaded vectorizer.
    email_vect = np.array(vectorizer.vectorize([email_text])).T

    # Make a prediction on the transformed text.
    prediction = model.predict(email_vect)

    # Convert predictions to a list for JSON response, if necessary
    prediction = [int(pred) for pred in prediction.flatten()]

    # Return the result as JSON
    if(prediction[0] == 0):
        return jsonify({'prediction': prediction, 'result': 'ham'})
    else:
        return jsonify({'prediction': prediction, 'result': 'spam'})

if __name__ == '__main__':
    app.run(debug=True)
