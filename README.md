

# retrival and text similarity tools
### A simple warpper around common retrival tools
fast and typo telorant tf-idf and multilingual dence retrival models

this repo is for educational porposes so the code is very basic and super readable (no fancy abstraction)


## editable install
1. clone repository from github 
2. cd to repository folder
3. install with pip
```pip install -e .```

## usage:
you can find most relevent document in just 3 line of code!
```python
from retriever import Retriever
docs = [
    'hello world',
    'how are you woooorld',
    'i am fine ',
    'this is a junk sentence!',
    'this is a siiiiimilar word with a typo',
    "it's time to find most similar documents",
]
retriever = Retriever(method='tf_idf_cfg_1')
retriever.add_doc(docs)
results = retriever.find_similars('similar term!', top_k=4)
```