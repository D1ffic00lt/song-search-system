from gensim.models import Word2Vec

class Word2VecModel(object):
    def __init__(self, model_path):
        self.model = Word2Vec.load(model_path)

    @staticmethod
    def process_text(text: str):
        return "".join([i if i.isalpha() else " " for i in text]).lower().split()

    def _sentence_to_vector(self, words):
        word_vectors = [self.model.wv[word] for word in words if word in self.model.wv]
        if not word_vectors:
            return None
        return sum(word_vectors) / len(word_vectors)

    def __call__(self, text):
        return self._sentence_to_vector(self.process_text(text))

if __name__ == "__main__":
    model = Word2VecModel(
        "../recommendation_system/word2vec_lyrics.model",
    )