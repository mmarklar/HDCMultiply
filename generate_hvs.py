import torch
import torchhd

dimensions = 100

# HVs for 0-9
value_hvs = torchhd.embeddings.Random(10, dimensions)
# HVs for the digit position in a number. Currently support up to 100x100? so 5 digits
position_hvs = torchhd.embeddings.Random(10, dimensions)

torch.save(value_hvs, "hvs/number_values.pt")
torch.save(position_hvs, "hvs/digit_positions.pt")
