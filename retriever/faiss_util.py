#  pip install faiss-cpu

import faiss
import pickle as pkl
import os


class vectorDB:
    def __init__(self, dim=768, normalize=False, load_path=False):
        self.normalize = normalize

        if load_path:
            if os.path.isdir(load_path + "/fiass-BD"):
                load_path = load_path + "/fiass-BD"
            self.index = faiss.read_index(load_path + "/db.fiass")
            with open(load_path + "/metadata.pkl", "rb") as f:
                meta = pkl.load(f)
                self.idx_to_key = meta["idx_to_key"]
                self.normalize = meta["normalize"]

            self.key_to_idx = {v: k for k, v in self.idx_to_key.items()}
            self.last_idx = len(self.idx_to_key)
            assert self.last_idx == self.index.ntotal

        else:
            self.index = faiss.index_factory(dim, "HNSW32")
            self.idx_to_key = {}
            self.key_to_idx = {}  # reverse map
            self.last_idx = 0

    def insert_batch(self, embeddings, uids):
        if self.normalize:
            embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)
        self.index.add(embeddings.cpu().numpy())
        for uid in uids:
            self.idx_to_key[self.last_idx] = uid
            self.key_to_idx[uid] = self.last_idx
            self.last_idx += 1

        assert self.last_idx == self.index.ntotal

    def reconstruct_vector(self, uid):
        return self.index.reconstruct(self.key_to_idx[uid])

    def find_top_k(self, embedding, k):
        if self.normalize:
            embedding = embedding / embedding.norm()

        D, Indexs = self.index.search(embedding.reshape(1, -1), k)
        if (D > 2000).any():
            print("very large distance detected (>2000) this could be a fiass bug")
            D[D >= 2000] = D[D < 2000].max()
            print("large values replaced !!\n")
        return D[0], [self.idx_to_key[idx] for idx in Indexs[0]]

    def save_BD(self, save_path, overwite_ok=False):
        os.makedirs(save_path + "/fiass-BD", exist_ok=overwite_ok)
        faiss.write_index(self.index, save_path + "/fiass-BD/db.fiass")
        with open(save_path + "/fiass-BD/metadata.pkl", "wb") as f:
            pkl.dump({"idx_to_key": self.idx_to_key, "normalize": self.normalize}, f)


"""  usage

# testing util with xlmr model
# pip install sentence_transformers

from sentence_transformers import SentenceTransformer
model = SentenceTransformer('sentence-transformers/paraphrase-xlm-r-multilingual-v1')

def get_emb (sentences):
    return model.encode(sentences, convert_to_tensor=True)#,device=device)

samples = [
    "میوه تازه",
    "شیر",
    "ماست",
    "بیسکوئیت",
]

db = vectorDB(normalize=True)
db.insert_batch(get_emb(samples),samples)
print(db.find_top_k(get_emb('دوغ'),4))

db.save_BD('./')

print(db.reconstruct_vector('شیر'))

del db

loaded_db = vectorDB(load_path='./')


"""
