from .. import retriever_factory, DenseRetriever, TFIDF_Retriever
from ..utils import top_k_argsort
from copy import deepcopy
import numpy as np
import json


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
            if load_path[-1] != "/":
                load_path += "/"
            # read meta file
            with open(load_path + "meta.json", "r") as f:
                self.meta = json.load(f)
            assert self.meta["method"] == "ensemble"
            # read docs
            with open(load_path + "docs.json", "r") as f:
                self.docs = json.load(f)
            if model_dict:
                raise ValueError("model_list should be None when load_path is not None")

            self.models = {}
            # read models from folders
            for name, model_conf in self.meta["model_dict"].items():
                model_path = load_path + name + "/"
                with open(model_path + "meta.json", "r") as f:
                    model_meta = json.load(f)
                if model_meta["method"] == "dense":
                    self.models[name] = DenseRetriever(load_path=model_path)
                elif model_meta["method"] == "tf_idf":
                    self.models[name] = TFIDF_Retriever(load_path=model_path)
                else:
                    raise ValueError("Unknown method: " + model_meta["method"])
            self.normalize = self.meta["normalize"]

        elif model_dict:
            self.meta = {
                "method": "ensemble",
                "model_dict": deepcopy(model_dict),
                "normalize": normalize,
            }
            self.docs = []

            def get_params(model_conf):
                params = model_conf.get("params", {})
                params["id_only"] = True
                return params

            self.models = {
                name: retriever_factory(model["method"], **get_params(model))
                for name, model in model_dict.items()
            }
            self.normalize = normalize

        self.model_weights = {
            name: model["weight"] for name, model in self.meta["model_dict"].items()
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
            if self.normalize:
                return (condid.get(name, worst[name]) - m) / std * model_weights[name]
            else:
                return condid.get(name, worst[name]) * model_weights[name]

        for k in all_keys:
            all_condidates[k]["ensemble"] = (
                np.mean([get_val(all_condidates[k], name) for name in self.models])
                + 0.5
            )

        top_k_idx = top_k_argsort(
            [all_condidates[k]["ensemble"] for k in all_keys], top_k
        )
        return [
            (self.docs[all_keys[i]], all_condidates[all_keys[i]]["ensemble"])
            for i in top_k_idx
        ]

    def save(self, path, overwite_ok=False):
        if path[-1] != "/":
            path += "/"
        for name, model in self.models.items():
            model.save(path + name + "/", overwite_ok)
        with open(path + "meta.json", "w") as f:
            json.dump(self.meta, f)
        with open(path + "docs.json", "w") as f:
            json.dump(self.docs, f, ensure_ascii=False, indent=2)
