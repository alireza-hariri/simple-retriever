

# retrival and text similarity tools
### A simple warpper around common retrival tools


this repo is for educational porposes so the code is very basic and super readable (no fancy abstraction)


## dependencies:
 1- sklearn: (for tf-idf)
 ``` pip install scikit-learn`` 
 2- sentence transformer (for dense models)
 you will need this lib to calc sentence embedding
 ```pip install sentence-transformers ```
 3- fiass (for dense models)
  for searching nearest nabors in embedding space
 ```pip install faiss-cpu ```
 you can also use conda to install this lib



## usage:
you can find most relevent document in just 3 line of code!
```python
from retriver import Retriever
docs = [
    'hello world',
    'how are you woooorld',
    'i am fine ',
    'this is a junk sentence!',
    'this is a simlar word with a typo',
    "it's time to find most similar documents",
]
retriever = Retriever(method='tf_idf_cfg_1')
retriever.add_doc(docs)
results = retriever.find_similars('similar term!', top_k=4)
```