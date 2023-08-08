from model.model import PhaseFIT
import torch

rand_image=torch.rand((6,3,256,256)).cuda()

model=PhaseFIT(size=256).cuda()

output=model(rand_image)
print(output.size())