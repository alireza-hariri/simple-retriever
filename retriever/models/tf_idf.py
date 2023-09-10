from sklearn.feature_extraction.text import TfidfVectorizer
from ..utils import top_k_argsort
import numpy as np
import json
import os


# TODO: define a class for TfidfVectorizer params


class TFIDF_Retriever:
    """
    basic warper aroud sklearn TfidfVectorizer usage
    DIP (Dependency Inversion Principle) is not followed here
    because I don't want to make things complicated
    """

    def __init__(
        self,
        analyzer="char_wb",
        ngram_range=(3, 4),
        max_df=0.9,
        id_only=False,  # only return ids of docs
        load_path=None,
    ):
        if load_path:
            # read meta file
            with open(load_path + "meta.json", "r") as f:
                self.meta = json.load(f)
            # read docs
            with open(load_path + "docs.json", "r") as f:
                self.docs = json.load(f)

            self.vectorizer = TfidfVectorizer(
                analyzer=self.meta["vectorizer_params"]["analyzer"],
                ngram_range=tuple(self.meta["vectorizer_params"]["ngram_range"]),
                max_df=self.meta["vectorizer_params"]["max_df"],
            )
            self.id_only = self.meta["id_only"]

        else:
            self.meta = {
                "method": "tf_idf",
                "vectorizer_params": {
                    "analyzer": analyzer,
                    "ngram_range": ngram_range,
                    "max_df": max_df,
                },
                "id_only": id_only,
            }

            self.vectorizer = TfidfVectorizer(
                analyzer=analyzer,
                ngram_range=ngram_range,
                max_df=max_df,
            )
            self.docs = []
            self.id_only = id_only

        self.is_fit = False
        self.X = None
        self.analyzer = None

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
        verbose = len(self.docs) > 50_000
        if verbose:
            print("\n calculating tf-idf term relevance matrix...", end="")
        self.X = self.vectorizer.fit_transform(self.docs)
        self.analyzer = self.vectorizer.build_analyzer()
        self.is_fit = True
        if verbose:
            print(" done")

    def get_vocababulary(self):
        return list(self.vectorizer.get_feature_names_out())

    def save(self, path, overwite_ok=False):
        if path[-1] != "/":
            path += "/"
        if (not overwite_ok) and os.path.exists(path):
            raise Exception("path already exists! please use overwite_ok=True")
        os.makedirs(path, exist_ok=True)
        with open(path + "docs.json", "w") as f:
            json.dump(self.docs, f)
        with open(path + "meta.json", "w") as f:
            json.dump(self.meta, f)
