
from retriever import DenseRetriever
from retriever import retriever_factory
from easytimer import tick, tock


test_docs = [
    'hello world',
    'how are you woooorld',
    'i am fine ',
    'in this term i will go to school',
    'this is a junk sample',
    'this is a siiiiimilar term with typo',
    "these terms are alike",
]



def test_DenseRetriever():
    for id_only in [False,True]:
        tick("loding model")
        retriever = DenseRetriever(
            # model = "intfloat/multilingual-e5-base",
            # model = "sentence-transformers/LaBSE",
            # model = "sentence-transformers/paraphrase-xlm-r-multilingual-v1",
            # size=768,
            # sentence_transformer=False
            id_only=id_only
        )
        tick("calculating embeddings")
        retriever.add_doc_batch(test_docs)
        tick("retrieving")
        results = retriever.find_similars('no similar word!', top_k=5)
        tick("saving")
        retriever.save('DenseRetriever_test/',overwite_ok=True)

        tick("loading")
        retriever = DenseRetriever(
            id_only=id_only,
            load_path='DenseRetriever_test/'
        )
        tick("adding more docs")
        retriever.add_doc_batch([
            'a different sentence',
            'a similar sentence',
        ])
        tick("retrieving again")
        results = retriever.find_similars('no similar word!', top_k=5)
        for item in results:
            print(item)
        tick('saving again')
        retriever.save('DenseRetriever_test-2/',overwite_ok=True)
        assert retriever.vec_db.last_idx == len(test_docs) + 2
        tock()





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


def compare_models():
    tick("loding model")
    #del retriever1,retriever2,retriever3
    retriever1 = retriever_factory('dense_LaBSE')
    retriever2 = retriever_factory('dense_multilingual_e5')
    retriever3 = retriever_factory('dense_MiniLM')

    #tick("calculating embeddings")
    retriever1.add_doc_batch(samples)
    retriever2.add_doc_batch(samples)
    retriever3.add_doc_batch(samples)
    tick("retrieving")
    
    def print_models(s):
        print(' *** ',s,' *** \n')
        for r in [retriever1,retriever2,retriever3]:
            results = r.find_similars(s, top_k=7)
            for item in results:
                print(item)
            print('-----\n')
            
    print_models("خسته شدم از وضعیت دیگه نمیتونیم اینجا دووم بیاریم این سرویس شما خیلی بی کیفیت هست")
    print_models("فعالیت بدنی")
    print_models("نوشیدنی")
    print_models("هزینه های پنهان")
    print_models("کلاه برداری کردن")
    print_models("دادگاه بین المللی")
    tock()

if __name__ == '__main__':
    # test_DenseRetriever()
    compare_models()