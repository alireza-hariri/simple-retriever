from retriever import retriever_factory, EnsembleRetriever
from easytimer import tick, tock

test_docs = [
    "hello world",
    "how are you woooorld",
    "i am fine ",
    "in this term i will go to school",
    "this is a junk sample",
    "this is a siiiiimilar term with typo",
    "these terms are alike",
]

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


def test_ensemble():
    tick("loading model")
    retriever = retriever_factory("ensemble_cfg_1")
    tick("adding docs")
    retriever.add_doc_batch(samples)
    tick("testing")
    retriever0 = retriever

    def top_1(s):
        results = retriever0.find_similars(s, top_k=5)
        return results[0][0]

    assert (
        top_1(
            "خسته شدم از وضعیت دیگه نمیتونیم اینجا دووم بیاریم این سرویس شما خیلی بی کیفیت هست"
        )
        == "نت ضعیفه"
    )
    assert top_1("فعالیت بدنی") == "ورزش صبحگاه"
    assert top_1("نوشیدنی") == "دوغ و نوشابه"
    assert top_1("هزینه های پنهان") == "سربار مالیاتی"
    assert top_1("کلاه برداری کردن") == "دروغ گفتن"
    assert top_1("دادگاه بین المللی") == "مبارزه ی مدنی"
    assert top_1("مبارزه با فساد") == "مبارزه با فساد"

    tick("saving model")
    # save and load
    retriever.save("test_ensemble", overwite_ok=True)
    tock()
    print("test passed!")


def load_saved_model():
    tick("loading saved model")
    retriever2 = EnsembleRetriever(load_path="test_ensemble")
    tick("adding more docs")
    retriever2.add_doc_batch(test_docs)
    tick("testing again")

    def top_1(s):
        results = retriever2.find_similars(s, top_k=5)
        return results[0][0]

    assert top_1("فعالیت بدنی") == "ورزش صبحگاه"
    assert top_1("نوشیدنی") == "دوغ و نوشابه"
    assert top_1("هزینه های پنهان") == "سربار مالیاتی"
    assert top_1("کلاه برداری کردن") == "دروغ گفتن"
    assert top_1("دادگاه بین المللی") == "مبارزه ی مدنی"
    assert top_1("مبارزه با فساد") == "مبارزه با فساد"
    tock()
    print("test passed!")


if __name__ == "__main__":
    load_saved_model()
