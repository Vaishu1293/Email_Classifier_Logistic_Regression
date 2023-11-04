class TextVectorizer:
    def __init__(self, texts):
        self.texts = texts
        self.vocabulary = self._build_vocabulary(texts)
        self.word_to_index = self._build_word_to_index()

    def _tokenize(self, texts):
        # Tokenize the provided texts
        return [text.split() for text in texts]

    def _build_vocabulary(self, texts):
        tokens = self._tokenize(texts)
        # Return sorted set of all unique words in the texts
        return sorted(set(sum(tokens, [])))

    def _build_word_to_index(self):
        # Build a mapping from words to their indices in the vocabulary
        return {word: index for index, word in enumerate(self.vocabulary)}

    def vectorize(self, new_texts=None):
        # If no new texts are provided, use the texts stored in the instance.
        texts_to_tokenize = self.texts if new_texts is None else new_texts

        # Tokenize the texts that need to be vectorized
        tokens = self._tokenize(texts_to_tokenize)

        vectors = []
        for text_tokens in tokens:
            vector = [0] * len(self.vocabulary)
            for token in text_tokens:
                index = self.word_to_index.get(token)
                if index is not None:
                    vector[index] += 1
            vectors.append(vector)
        return vectors
