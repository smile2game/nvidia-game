import torch
outputs = torch.zeros(1,77,768, dtype=torch.float32).to("cuda")
print(outputs.reshape(-1))