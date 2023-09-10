
from . import TFIDF_Retriever
from . import DenseRetriever


def retriever_factory(method='tf_idf_cfg_1', **params):

    ''' factory function for creating proper retriever model '''

    match method:

        case 'tf_idf_cfg_1': # config-1
            return TFIDF_Retriever() # default params
            # analyzer='char_wb',
            # ngram_range=(3,4),

        case 'tf_idf_cfg_2': # config-2
            return TFIDF_Retriever(
                analyzer='char', 
                ngram_range=(4,5)  
            )

        case 'tf_idf_custom':
            return TFIDF_Retriever(**params)
        
        case 'dense_MiniLM':
            return DenseRetriever(
                model='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
                size=384,
            )

        case 'dense_LaBSE':
            return DenseRetriever(
                model='sentence-transformers/LaBSE',
                size=768,
            )

        case 'dense_multilingual_e5':
            return DenseRetriever(
                model='intfloat/multilingual-e5-base',
                size=768,
                sentence_transformer=False,
            )   

        case 'dense_custom':
            return DenseRetriever(**params)
        
        case _:
            raise ValueError(f'unknown method: {method}')
