from vectorizer import TextVectorizer
import pickle

texts = [
    'I love programming',
    'programming is fun',
    'love coding and programming'
]

vectorizer = TextVectorizer(texts)
vectors = vectorizer.vectorize()
print(vectors)

filename = 'vectorizer.sav'
with open(filename, 'wb') as file:
    pickle.dump(vectorizer, file)  # Saving the entire model