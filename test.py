import torch
import torchhd
from number_encoding import encode_num, decode_num, greedy_decode, get_first_negative_weight

device = torch.device("cpu")
print(device)

positions = torch.load("hvs/digit_positions.pt")
values = torch.load("hvs/number_values.pt")

def mult(net, a, b):
    # output = a * b
    a_enc = encode_num(a, values, positions)
    b_enc = encode_num(b, values, positions)
    in_enc = torch.cat((a_enc.value, b_enc.value), dim=-1)
    in_enc.to(device)
    output = net(in_enc)
    result_enc = encode_num(a * b, values, positions)
    #print(in_enc)
    #print(output)
    #print(result_enc.value)
    hm = torchhd.structures.HashTable(output)
    #return decode_num(hm, values, positions)
    return greedy_decode(hm, a * b, values, positions)

if __name__ == "__main__":
    net = torch.load("model.pt")
    net.to(device)
    net.eval()
    count = 0
    correct = 0
    print("Evaluating on test data...")
    for i in range(100, 200):
        for j in range(100, 200):
            a = mult(net, j, i)
            #print(i, "*", j, "=", a)
            count += 1
            if a == j * i:
                correct += 1

    print("Out of", count, ",", correct, "correct:", (correct / count) * 100, "%")
