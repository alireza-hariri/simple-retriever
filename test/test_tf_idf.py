from easytimer import tick, tock



test_docs = [
        'hello world',
        'how are you woooorld',
        'i am fine ',
        'this is a junk sentence!',
        *(['unrelated docs!'] * 100_000),
        'this is a siiiiimilar word with a typo',
        "it's time to find most similar documents",
    ]

def method_tester(method):
    tick('importing')
    from retriever import Retriever
    tick('fitting')
    retriever = Retriever(method=method)
    retriever.add_doc(test_docs)
    retriever.retriever.fit()
    tick('finding result')
    results = retriever.find_similars('similar term!', top_k=4)
    tock()
    assert len(results) == 4
    assert results[0][0] == test_docs[-1]
    assert results[1][0] == test_docs[-2]
    assert results[0][1] > results[1][1] # bigger score

def test_config_1():
    method_tester('tf_idf_cfg_1')

def test_config_2():
    method_tester('tf_idf_cfg_2')
        

def test_TFIDF_Retriever():
    from retriever import TFIDF_Retriever
    
    tick('fitting')
    retriever = TFIDF_Retriever(
        analyzer='char', 
        ngram_range=(3,5)
    )
    retriever.add_doc_batch(test_docs)
    retriever.fit()
    tick('finding result')
    results = retriever.find_similars('similar term!', top_k=4)
    tock()
    assert len(results) == 4
    assert results[0][0] == test_docs[-1]
    assert results[1][0] == test_docs[-2]
    assert results[0][1] > results[1][1] # bigger score


