import torch
import torchhd


d = 1500
digits = torchhd.random(11, d, "MAP")
print(digits[0].dtype, digits[0].shape, digits[0].element_size())

import sys
arr = [1] * 1000
print("1000 1s size:", sys.getsizeof(arr))

def similarity(value, vectors):
    s = torchhd.cosine_similarity(value, vectors)
    v, i = torch.max(s, dim=0)
    #print("Closest:", i.item())
    return i.item(), s

def remove_val(hv, value):
    return torchhd.permute(hv + torchhd.negative(value), shifts=-1)

def test_3_digits():
    result = digits[2] + torchhd.permute(digits[1] + torchhd.permute(digits[0] + torchhd.permute(digits[10])))
    print(result)
    i, s = similarity(result, digits)
    result = torchhd.permute(result, shifts=-1)
    i, s = similarity(result, digits)
    result = torchhd.permute(result, shifts=-1)
    i, s = similarity(result, digits)
    result = torchhd.permute(result, shifts=-1)
    i, s = similarity(result, digits)

#test_3_digits()

def test_bundles():
    hv = digits[-1]
    digit_count = len(digits) - 1
    print("Testing", digit_count, "keys")
    how_many = 100
    for i in range(how_many):
        hv = digits[i % digit_count] + torchhd.permute(hv)

    error_count = 0
    for j in range(how_many - 1, -1, -1):
        index, s = similarity(hv, digits)
        if not j % digit_count == index:
            error_count += 1
            #print("Did not find a match for", j, j % 10, "!=", index) 
        #hv = torchhd.permute(hv, shifts=-1)
        hv = remove_val(hv, digits[index])

    print("encountered", error_count, "errors")

test_bundles()
