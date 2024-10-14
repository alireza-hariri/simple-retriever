

# <b>simple-retriever</b> <small>(basic retrieval methods)</small>
### A simple wrapper around common retrieval tools
This repository is a simple wrapper around common retrieval tools especially
the sklearn `tf-idf` and some huggingface models.  
The examples supports fast and typo-tolerant tf-idf, multilingual sentence embedding models, and hybrid methods for retrieval.  
this repo is mainly for educational purposes so the code is super readable and there is no abstraction (just duck-typing)


## editable install
1. clone repository from github 
2. cd to the repository folder
3. install with pip in editable mode  
`pip install -e .`

## usage:
you can find the most relevant document in just 3 lines of code!
```python
from retriever import retriever_factory
docs = [
    'hello world',
    'How are you woooorld',
    'I am fine ',
    'This is a junk sentence!',
    'This is a siiiiimilar word with a typo',
    "it's time to find most similar documents",
]
retriever = retriever_factory(method='tf_idf_cfg_1')
retriever.add_doc_batch(docs)
results = retriever.find_similars('similar term!', top_k=4)
```

dense models also have same interface
```python
samples = [
    "میوه تازه",
    "شیر",
    "ماست",
    "شیرینی",
    "دوغ و نوشابه",
    "دروغ گفتن",
    "بادام",
    "کره",
    "هلو های تازه",
    "نارنج",
    "شکلات صبحانه",
    "فندق",
    "آزادی بیان",
    "مبارزه با تروریسم",
    "مبارزه با فساد",
    "مبارزه ی مدنی",
    "یادگیری ماشین",
    "الگوریتم های دسته بندی",
    "سربار مالیاتی",
    "نت ضعیفه",
    "ورزش صبحگاه",
]
retriever = retriever_factory(method='dense_LaBSE')
retriever.add_doc_batch(samples)
results1 = retriever.find_similars("فعالیت بدنی") # ->  "ورزش صبحگاه"
results2 = retriever.find_similars("نوشیدنی") # -> "دوغ و نوشابه"
results3 = retriever.find_similars("هزینه های پنهان") # -> "سربار مالیاتی"
results4 = retriever.find_similars("کلاه برداری کردن") # -> "دروغ گفتن"

```

# Supported retrieval methods

### tf_idf_cfg_1
tf-idf config-1 -> fast and typo tolerant tf-idf (insensitive to word orders)

### tf_idf_cfg_2
tf-idf config-2 ->  less typo telorant tf-idf + little bit of order sensitiveness

### dense_LaBSE
good and big model.  
you can try this model [here](https://huggingface.co/sentence-transformers/LaBSE)
on hugging-face 


### dense_multilingual_e5
It's a little smaller than LaBSE, but a good one.  
you can try this model [here](https://huggingface.co/intfloat/multilingual-e5-base)
on hugging-face 

### dense_MiniLM
small model.  
you can try this model [here](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)
on hugging-face 

### ensemble_cfg_1
an ensemble of 4 different models (labse,minilm,e5,tf-ifd) (it may need 4GB of free RAM for initialization)


## custom configs 
for understanding custom configs please refer to  [factory.py](./retriever/factory.py) file

