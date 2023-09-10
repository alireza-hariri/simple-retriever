from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from ..utils import top_k_argsort


class TFIDF_Retriever:
    """
    basic warper aroud sklearn TfidfVectorizer usage
    DIP (Dependency Inversion Principle) is not followed here
    because I don't want make things complicated
    """

    def __init__(
        self,
        analyzer="char_wb",
        ngram_range=(3, 4),
        max_df=0.9,
        verbose=False,
        id_only=False,  # only return ids of docs
    ):
        self.vectorizer = TfidfVectorizer(
            analyzer=analyzer,
            ngram_range=ngram_range,
            max_df=max_df,
        )
        self.verbose = verbose
        self.is_fit = False
        self.docs = []
        self.X = None  # tf-idf term relevance matrix
        self.analyzer = None
        self.id_only = id_only

    def add_doc(self, doc):
        self.is_fit = False
        self.docs.append(doc)

    def add_doc_batch(self, docs):
        self.is_fit = False
        self.docs.extend(docs)

    def find_similars(self, query: str, top_k=20):
        if not self.is_fit:
            self.fit()
        query_vec = self.vectorizer.transform([query])
        # cosine similarity
        # https://stackoverflow.com/questions/12118720/python-tf-idf-cosine-to-find-document-similarity
        scores = np.dot(query_vec, self.X.T).toarray()[0]
        top_idx_sorted = top_k_argsort(scores, top_k)

        if self.id_only:
            return [(i, scores[i]) for i in top_idx_sorted]
        else:
            return [(self.docs[i], scores[i]) for i in top_idx_sorted]

    def fit(self):
        verbose = self.verbose or len(self.docs) > 50_000
        if verbose:
            print("\n calculating tf-idf term relevance matrix...", end="")
        self.X = self.vectorizer.fit_transform(self.docs)
        self.analyzer = self.vectorizer.build_analyzer()
        self.is_fit = True
        if verbose:
            print(" done")

    def get_vocababulary(self):
        return list(self.vectorizer.get_feature_names_out())


def test_TFIDF_Retriever():
    docs = [
        "hello world",
        "how are you woooorld",
        "i am fine ",
        "this is a junk sentence!",
        "this is a simlar word with a typo",
        "it's time to find most similar documents",
    ]
    retriever = TFIDF_Retriever(
        # uncomment this if order of words is also important
        # analyzer='char',
        # ngram_range=(4,4)
    )
    retriever.add_doc_batch(docs)
    # print(retriever.get_vocababulary())
    resutls = retriever.find_similars("similar term!", top_k=5)
    for item in resutls:
        print(item)
