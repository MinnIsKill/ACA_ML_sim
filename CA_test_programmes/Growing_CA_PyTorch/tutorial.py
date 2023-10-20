import torch

grid_0 = torch.rand(5, 5)
#print(grid_0)


###
# rules; filter we will convolve the input grid with 
# (note: it's 3x3, meaning we are only allowed to look at the immediate neighbours and the cell itself)
rules = torch.tensor([[0, 0.5, 0], [0, 0, 0], [0, 0.5, 0]])

###
# perform convolution
grid_1 = torch.nn.functional.conv2d(grid_0[None, None, ...], rules[None, None, ...], padding=1).squeeze()
#print(grid_0)