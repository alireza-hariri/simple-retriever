
from . import TFIDF_Retriever


class Retriever:
    """
    wrapper class for different retrieval methods
    read more in the README.md
    """
    def __init__(self, method, **params):

        if method == 'tf_idf_cfg_1': # config-1
            self.retriever = TFIDF_Retriever(
                # all parameters are default
            )

        elif method == 'tf_idf_cfg_2': # config-2
            self.retriever = TFIDF_Retriever(
                analyzer='char', 
                ngram_range=(4,4)  
            )
    
        elif method == 'tf_idf_custom':
            self.retriever = TFIDF_Retriever(
                **params
            )
        
        else:
            raise ValueError(f'unknown method: {method}')

    def add_doc(self,doc):
        if type(doc) != list:
            self.retriever.add_doc(doc)
        else:
            self.retriever.add_doc_batch(doc)
    
    def find_similars(self,query:str,top_k=20):
        return self.retriever.find_similars(query,top_k)

    



