import torch
import torchhd
from sklearn.cluster import KMeans, DBSCAN
import numpy as np

dimensions = 100

def encode_num(number, value_hvs, position_hvs):
    output = torchhd.structures.HashTable(dimensions)
    position = 0
    val = number
    while val >= 10:
        digit = int(val % 10)
        output.add(position_hvs(torch.tensor([position])), value_hvs(torch.tensor([digit])))
        #print(val, digit, "@", position)

        val = val // 10
        position += 1

    if val > 0:
        #print("Adding last digit", val, position)
        output.add(position_hvs(torch.tensor([position])), value_hvs(torch.tensor([val])))

    #print("Encoded", number, "with shape", output.value.shape)
    return output

# Doesn't use the weights from the cosine similarity 
# just tries to decode x number of digits based on what the answer should be
def greedy_decode(hv, value, value_hvs, position_hvs):
    result = 0
    digits = []
    for p in range(10):
        val_hv = hv.get(position_hvs(torch.tensor([p])))
        similarity = torchhd.cosine_similarity(val_hv, value_hvs.weight)
        v, i = torch.max(similarity, dim=1)
        digits.append(i)
    #print("digits", digits)
    # count how many digits
    digit_count = len(str(value))
    for i in range(digit_count):
        if i == 0:
            result += digits[i].item()
            #print("Adding", digits[i].item())
        else:
            result += digits[i].item() * 10 ** i
            #print("Adding", digits[i].item() * 10 ** i)
    return result

def decode_num(hv, value_hvs, position_hvs):
    result = 0
    weights = []
    digits = []
    for p in range(10):
        val_hv = hv.get(position_hvs(torch.tensor([p])))
        similarity = torchhd.cosine_similarity(val_hv, value_hvs.weight)
        v, i = torch.max(similarity, dim=1)
        weights.append(v.item())
        digits.append(i)
    #print("weights", weights)
    #print("digits", digits)
    weights = np.array(weights)
    for i in range(len(weights)):
        if weights[i] > 0.299:
            if i == 0:
                result += digits[i].item()
                #print("Adding", digits[i].item())
            else:
                result += digits[i].item() * 10 ** i
                #print("Adding", digits[i].item() * 10 ** i)
        else:
            break
    return result
'''
    #km = KMeans(2, n_init=10)
    #classes = km.fit_predict(weights.reshape(-1,1))
    clustering = DBSCAN(eps=0.3, min_samples=2)
    classes = clustering.fit_predict(weights.reshape(-1,1))
    print("classes", classes)
    initial_class = classes[0]
    for i in range(len(classes)):
        if classes[i] == initial_class:
            if p == 0:
                result += digits[i].item()
                print("Adding", digits[i].item())
            else:
                result += digits[i].item() * 10 ** i
                print("Adding", digits[i].item() * 10 ** i)
        else:
          break
    return result
    '''

# Returns the weight value for the first digit that should not be counted
def get_first_negative_weight(hv, value_hvs, position_hvs, value):
    result = 0
    weights = []
    digits = []
    for p in range(10):
        val_hv = hv.get(position_hvs(torch.tensor([p])))
        similarity = torchhd.cosine_similarity(val_hv, value_hvs.weight)
        v, i = torch.max(similarity, dim=1)
        weights.append(v.item())
        digits.append(i)

    length = len(str(value))
    #print(value, "length", length, "weight", weights[length])
    for i in range(length):
        if weights[i] < weights[length]:
            print("Got valid weight with less value than invalid one")
    return weights[length]

if __name__ == "__main__":
    positions = torch.load("hvs/digit_positions.pt")
    values = torch.load("hvs/number_values.pt")
    #encode_num(500, values, positions)
    #print(decode_num(encode_num(1234320966, values, positions), values, positions))
    print(decode_num(encode_num(1000, values, positions), values, positions))
