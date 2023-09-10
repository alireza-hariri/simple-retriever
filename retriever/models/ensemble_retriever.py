from .. import retriever_factory
from ..utils import top_k_argsort
from copy import deepcopy
import numpy as np


def calc_mean_std(values):
    values = np.array(values)
    return values.mean(), (values.std() + 0.05)


class EnsembleRetriever:
    def __init__(
        self,
        model_dict=None,
        load_path=None,
        normalize=True,
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
            self.normalize = normalize
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
        model_mean_std = {}
        for name, ret_model in self.models.items():
            results = ret_model.find_similars(
                query, top_k=min(top_k, len(self.docs) - 1)
            )
            model_mean_std[name] = calc_mean_std([v for _, v in results])

            for key, score in results:
                if key not in all_condidates:
                    all_condidates[key] = {}
                all_condidates[key][name] = score
            worst[name] = score

        all_keys = list(all_condidates.keys())

        def get_val(condid, name):
            m, std = model_mean_std[name]
            return (condid.get(name, worst[name])-m) / std * model_weights[name]

        for k in all_keys:
            all_condidates[k]["ensemble"] = np.mean(
                [get_val(all_condidates[k], name) for name in self.models]
            ) + 0.5

        top_k_idx = top_k_argsort(
            [all_condidates[k]["ensemble"] for k in all_keys], top_k
        )
        return [
            (self.docs[all_keys[i]], all_condidates[all_keys[i]]["ensemble"])
            for i in top_k_idx
        ]
