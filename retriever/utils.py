import numpy as np


def top_k_argsort(scores, k):
    # using argpartition is faster than argsort [ O(n) vs O(n*log(n)) ]
    # https://stackoverflow.com/questions/6910641/how-do-i-get-indices-of-n-maximum-values-in-a-numpy-array
    top_k_idxs = np.argpartition(scores, -k)[-k:]
    scores = np.array(scores)
    top_k_idxs = top_k_idxs[np.argsort(-scores[top_k_idxs])]
    return top_k_idxs


def average_pool(last_hidden_states, attention_mask):
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def test_top_k_argsort():
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    # assert will raise error if its argument is false
    assert np.array_equal(top_k_argsort(scores, 3), np.argsort(-scores)[:3])
    # you can use pytest to see passed tests instead of printing results !
    print("test_top_k_argsort passed")


if __name__ == "__main__":
    # you can use pytest -s to run all functions that starts with
    #  'test_' instead of calling them this way !
    test_top_k_argsort()
