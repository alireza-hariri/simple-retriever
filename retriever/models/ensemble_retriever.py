

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
        
            raise NotImplementedError('loading is not implemented yet')

        elif model_dict:
            self.meta = model_dict
            self.docs = []
            # model weights !!!!!!!!!
            # models !!!!!!!
    
    def add_doc(self,doc):
        for model_name,model in self.models.items():
            model.add_doc(doc)
        self.docs.append(doc)

    def add_doc_batch(self,docs):
        for model_name,model in self.models.items():
            model.add_doc_batch(docs)
        self.docs.extend(docs)

    def find_similars(self,query,top_k=5):
        model_weights = {} # ???
        all_condidates = {}
        worst = {}
        for name,ret_model in models.items():
            results = ret_model.find_similars(s, top_k=2*k)
            for key,score in results:
                if key not in all_condidates:
                    all_condidates[key] = {}
                all_condidates[key][name] = score*model_coef[name]
            worst[model] = score*model_coef[name] # last-sample score is lowest score
        
        # TODO: normalize scores of each model
        for k in all_condidates:
            all_condidates[k]['ensemble'] = np.mean([all_condidates[k].get(m,worst[m]) for m in models])
