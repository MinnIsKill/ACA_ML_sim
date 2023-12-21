#gradients are essential for model optimisation, and 
#PyTorch's 'autograd' package provides tools which can 
#do all the computations for us
import torch

print("\r\n===== requires_grad=True =====\r\n")

x = torch.randn(3, requires_grad=True)
print(x)

y = x+2
print(y)

print("\r\n----------\r\n")

z = y*y*2
print(z)
z = z.mean()
print(z)

z.backward() #would produce an error if requires_grad=False
print(x.grad)

print("\r\n===== gradient argument =====\r\n")

t = y*y*2
print(t)
#t.backward() #would produce an error because element #1
              #of tensor isn't scalar (has multiple values)

z2 = y*y*2
v = torch.tensor([0.1, 1.0, 0.001], dtype=torch.float32)
z2.backward(v) #dz/dx
print(x.grad)

print("\r\n===== prevent PyTorch from tracking history and calculating grad_fn attribute =====\r\n")

x = torch.randn(3, requires_grad=True)
print(x)
# OPTION 1:  x.requires_grad(False)
# OPTION 2:  x.detach()
# OPTION 3:  with torch.no_grad():

print("\r\n----------\r\n")

x.requires_grad_(False) #trailing '_' means var will be modified in-place
print(x)

print("\r\n----------\r\n")

x = torch.randn(3, requires_grad=True)
y = x.detach()
print(y)

print("\r\n----------\r\n")

x = torch.randn(3, requires_grad=True)
with torch.no_grad():
    y = x + 2
    print(y)

print("\r\n===== dummy training example =====\r\n")

weights1 = torch.ones(4, requires_grad=True)
weights2 = torch.ones(4, requires_grad=True)

for epoch in range(2):
    model_output = (weights1*3).sum()

    model_output.backward() #loop #2:  the second backward 
            #call will again accumulate the values and write 
            #them into the grad attribute, resulting in 
            #tensor([6., 6., 6., 6.])

    print(weights1.grad)
            #loop #1:  tensor([3., 3., 3., 3.])
            #loop #2:  tensor([6., 6., 6., 6.])

print("\r\n----------\r\n")
#to avoid previous issue, do:

for epoch in range(3):
    model_output = (weights2*3).sum()

    model_output.backward()

    print(weights2.grad)

    weights2.grad.zero_() #this (zero out the grad attribute)

print("\r\n===== PyTorch built-in optimizer =====\r\n")

weights3 = torch.ones(4, requires_grad=True)

optimizer = torch.optim.SGD([weights3], lr=0.01) #SGD = Stochastic Gradient Descent
                                              #lr = learning rate
optimizer.step()
optimizer.zero_grad() #same thing as line #89 (to avoid same issue)