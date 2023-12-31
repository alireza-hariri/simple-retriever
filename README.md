

# <b>simple-retriever</b> <small>(basic retrival methods)</small>
### A simple warpper around common retrival tools
this repository is a simple warpper around common retrival tools especially
sklearn tf-idf and some of huggingface models.  
We support fast and typo telorant tf-idf and multilingual sentence embedding models for retrieval.  
this repo is mainly for educational porposes so the code is very basic and super readable (no fancy abstraction)


## editable install
1. clone repository from github 
2. cd to repository folder
3. install with pip in editable mode  
`pip install -e .`

## usage:
you can find most relevent document in just 3 line of code !
```python
from retriever import retriever_factory
docs = [
    'hello world',
    'how are you woooorld',
    'i am fine ',
    'this is a junk sentence!',
    'this is a siiiiimilar word with a typo',
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

# supported retrival methods

### tf_idf_cfg_1
tf-idf config-1 -> fast and typo telorant tf-idf (insensitive to word orders)

### tf_idf_cfg_2
tf-idf config-2 ->  less typo telorant tf-idf + little bit of order sensitiveness

### dense_LaBSE
good and big model.  
you can try this model [here](https://huggingface.co/sentence-transformers/LaBSE)
on hugging-face 


### dense_multilingual_e5
a little smaller than LaBSE but still good model.  
you can try this model [here](https://huggingface.co/intfloat/multilingual-e5-base)
on hugging-face 

### dense_MiniLM
small model.  
you can try this model [here](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)
on hugging-face 

### ensemble_cfg_1
an ensemble of 4 different models (labse,minilm,e5,tf-ifd) (it may need 4GB of free RAM for initialization)


## custom configs 
for undestanding custom configs please refere to  [factory.py](./retriever/factory.py) file

