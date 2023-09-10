
import numpy as np
import torch
import json
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from ..faiss_util import vectorDB
from ..utils import average_pool

class DenseRetriever:
    '''
    basic warper aroud sklearn TfidfVectorizer usage
    DIP (Dependency Inversion Principle) is not followed here because I don't want make  complicated
    '''
    def __init__(
        self,
        model='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
        size=384,
        id_only=False,
        normalize=True,
        sentence_transformer=True,
        load_path=None
    ):

        if sentence_transformer:
            self.model = SentenceTransformer(model)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model)
            self.model = AutoModel.from_pretrained(model)

        if load_path:
            self.vec_db = vectorDB(load_path=load_path)
            with open(load_path+'docs.json','r') as f:
                self.docs = json.load(f)
        else:
            self.vec_db = vectorDB(normalize=normalize,dim=size)
            self.docs = []

        # in id_only mode, doc_ids are returned instead of docs
        self.id_only = id_only


    def get_emb (self,sentences): 
        if isinstance(self.model,SentenceTransformer):
            with torch.no_grad():
                return self.model.encode(sentences, convert_to_tensor=True)#,device=device)
        else:
            batch = self.tokenizer(sentences, max_length=512, padding=True, truncation=True, return_tensors="pt")
            with torch.no_grad():
                outputs = self.model(**batch)
                embeddings = average_pool(outputs.last_hidden_state, batch['attention_mask'])
                return embeddings

    def add_doc(self,doc):
        if not self.id_only:
            assert len(self.docs) == self.vec_db.last_idx
            self.docs.append(doc)
        self.vec_db.insert_batch(self.get_emb([doc]),[self.vec_db.last_idx])
       
    def add_doc_batch(self,docs):
        if len(docs) > 1000:
            print('warning: inserting more than 1000 docs at once can be very slow because of ram problem')
        if not self.id_only:
            assert len(self.docs) == self.vec_db.last_idx
            self.docs.extend(docs)
        s = self.vec_db.last_idx 
        new_ids = list(range(s,s+len(docs)))
        self.vec_db.insert_batch(self.get_emb(docs),new_ids)

    def get_doc(self,doc_id):
        if self.id_only:
            return doc_id
        else:
            return self.docs[doc_id]

    def find_similars(self,query:str, top_k=20):
        
        distance, doc_ids = self.vec_db.find_top_k(self.get_emb(query),top_k)
        score = (2 - distance**2) / 2 # calculating cosine similarity from distance
        return [(self.get_doc(doc_id), score[i]) for i,doc_id in enumerate(doc_ids)]

    def save(self,path,overwite_ok=False):
        if path[-1] != '/':
            path += '/'
        self.vec_db.save_BD(path,overwite_ok)
        with open(path+'docs.json','w') as f:
            json.dump(self.docs,f,ensure_ascii=False,indent=2)


