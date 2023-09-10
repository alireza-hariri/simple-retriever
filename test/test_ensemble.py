# import retriever as rtv
from retriever import retriever_factory
# from easytimer import tick, tock

test_docs = [
    "hello world",
    "how are you woooorld",
    "i am fine ",
    "in this term i will go to school",
    "this is a junk sample",
    "this is a siiiiimilar term with typo",
    "these terms are alike",
]


def test_ensemble():
    retriever = retriever_factory("ensemble_cfg_1")
    retriever.add_doc_batch(test_docs)
    results = retriever.find_similars("no similar word!", top_k=2)
    for item in results:
        print(item)
