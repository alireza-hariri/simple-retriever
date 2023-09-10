from retriever import retriever_factory

# from easytimer import tick, tock

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
    retriever = retriever_factory("ensemble_cfg_1")
    retriever.add_doc_batch(test_docs)
    results = retriever.find_similars("no similar word!", top_k=4)
    for item in results:
        print(item)
    retriever.add_doc_batch(samples)

    def print_results(s):
        print("\nquery:", s)
        results = retriever.find_similars(s, top_k=4)
        for item in results:
            print(item)

    print_results(
        "خسته شدم از وضعیت دیگه نمیتونیم اینجا دووم بیاریم این سرویس شما خیلی بی کیفیت هست"
    )
    print_results("فعالیت بدنی")
    print_results("نوشیدنی")
    print_results("هزینه های پنهان")
    print_results("کلاه برداری کردن")
    print_results("دادگاه بین المللی")
