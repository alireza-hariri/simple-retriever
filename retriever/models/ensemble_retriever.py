from .. import retriever_factory
from ..utils import top_k_argsort
from copy import deepcopy
import numpy as np


class EnsembleRetriever:
    def __init__(
        self,
        model_dict=None,
        load_path=None,
    ):
        if load_path:
            # # read meta file
            # with open(load_path+'meta.json','r') as f:
            #     meta = json.load(f)
            # # read docs
            # with open(load_path+'docs.json','r') as f:
            #     self.docs = json.load(f)
            # # TODO: load retreiver models from folders !!!!!!!!!!!
            # if model_dict:
            #     raise ValueError('model_list should be None when load_path is not None')

            raise NotImplementedError("loading is not implemented yet")

        elif model_dict:
            self.meta = deepcopy(model_dict)
            self.docs = []

            def get_params(model_meta):
                params = model_meta.get("params", {})
                params["id_only"] = True
                return params

            self.models = {
                name: retriever_factory(model["method"], **get_params(model))
                for name, model in model_dict.items()
            }
            self.model_weights = {
                name: model["weight"] for name, model in model_dict.items()
            }

    def add_doc(self, doc):
        for model_name, model in self.models.items():
            model.add_doc(doc)
        self.docs.append(doc)

    def add_doc_batch(self, docs):
        for model_name, model in self.models.items():
            model.add_doc_batch(docs)
        self.docs.extend(docs)

    def find_similars(self, query, top_k=5):
        model_weights = self.model_weights
        all_condidates = {}
        worst = {}
        for name, ret_model in self.models.items():
            results = ret_model.find_similars(
                query, top_k=min(top_k, len(self.docs) - 1)
            )
            for key, score in results:
                if key not in all_condidates:
                    all_condidates[key] = {}
                all_condidates[key][name] = score * model_weights[name]
            worst[name] = score * model_weights[name]

        # TODO: normalize scores of each model
        for k in all_condidates:
            all_condidates[k]["ensemble"] = np.mean(
                [all_condidates[k].get(m, worst[m]) for m in self.models]
            )
        all_keys = list(all_condidates.keys())
        top_k_idx = top_k_argsort(
            [all_condidates[k]["ensemble"] for k in all_keys], top_k
        )
        return [
            (self.docs[all_keys[i]], all_condidates[all_keys[i]]["ensemble"])
            for i in top_k_idx
        ]
