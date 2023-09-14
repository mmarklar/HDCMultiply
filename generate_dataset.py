import torch
from dataset import MultiDataset
from number_encoding import encode_num

positions = torch.load("hvs/digit_positions.pt")
values = torch.load("hvs/number_values.pt")

equations = torch.tensor([])
solutions = torch.tensor([])
for a in range(100, 200):
    for b in range(1, 100):
        c = a * b
        a_enc = encode_num(a, values, positions)
        b_enc = encode_num(b, values, positions)
        c_enc = encode_num(c, values, positions)
        #print(a, "*", b, "=",  c)
        #print(a_enc.value.shape, b_enc.value.shape)
        in_enc = torch.cat((a_enc.value, b_enc.value), dim=-1)
        equations = torch.cat((equations, in_enc))
        solutions = torch.cat((solutions, c_enc.value))

torch.save(equations, "hvs/equations.pt")
torch.save(solutions, "hvs/solutions.pt")

all_data = MultiDataset()
train_data, test_data = torch.utils.data.random_split(all_data, [0.8, 0.2])

torch.save(train_data, "hvs/train_data.pt")
torch.save(test_data, "hvs/test_data.pt")
