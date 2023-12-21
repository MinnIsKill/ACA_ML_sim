import torch
import numpy as np

print("===== create tensor =====\r\n")

#1D tensor - empty
y = torch.empty(3)
print(y)
#2D tensor - filled with zeros
z = torch.zeros(2, 3)
print(z)
#3D tensor - filled with ones
x = torch.ones(2, 2, 3)
print(x)

print("\r\n===== change data type =====\r\n")

print(x.dtype) # will be float by default
x = torch.ones(2, 2, dtype=torch.int) # make it int
print(x.dtype)
print(x.size)

print("\r\n===== alternative way to create tensor =====\r\n")

x = torch.tensor([2.5, 0.1])
print(x)

print("\r\n===== basic operations on tensors =====\r\n")

x = torch.rand(2,2)
y = torch.rand(2,2)
print(x)
print(y)
z = x + y
z = torch.mul(x,y)
print(z)

print("\r\n----------\r\n")

#in-place operation
y.add_(x)
print(y)

print("\r\n----------\r\n")

#both following lines do the same thing
z = x - y
print(z)
z = torch.sub(x, y)
print(z)

print("\r\n===== slicing operations on tensors =====\r\n")

x = torch.rand(5,3)
print(x)
print(x[1, :]) #print row #1, but all columns
print(x[1, 1]) #print element at row #1 column #1
print(x[1, 1].item()) #print actual value of said element

print("\r\n===== reshaping tensors =====\r\n")

x = torch.rand(4,4) #create a 2D tensor
print(x)
y = x.view(16) #make it a 1D vector
print(y)

print("\r\n----------\r\n")

print(x)
y = x.view(-1, 8) #pytorch automatically determines correct size
print(y.size())

print("\r\n===== numpy & torch tensors conversion (into eachother) =====\r\n")

a = torch.ones(5)
print(a)
b = a.numpy()
print(b)
print(type(b))

print("\r\n----------\r\n")

#!!! CAREFUL: if tensor is on CPU and not GPU then both objects will share 
#             the same memory location (so if we change one we change both)

print(a)
print(b)
a.add_(1)
print(a)
print(b)

print("\r\n----------\r\n")

a = np.ones(5)
print(a)
b = torch.from_numpy(a) #create torch tensor from numpy array
#b = torch.from_numpy(a, dtype=float) #you can also specify data type
print(b)

#!!! CAREFUL: if tensor is on CPU and not GPU then both objects will share 
#             the same memory location (so if we change one we change both)

a += 1
print(a)
print(b)

print("\r\n===== working with CPU vs GPU =====\r\n")

if torch.cuda.is_available(): #if we can use GPU
    device = torch.device("cuda")
    x = torch.ones(5, device=device) #this way
    y = torch.ones(5) #or this
    y = y.to(device)  #way
    print(x.device)
    print(y.device)
    
    print("\r\n----------\r\n")

    #!!! CAREFUL: numpy can only handle CPU tensors (cannot) convert GPU 
    #             tensor back to numpy)
    z = x + y
    #z.numpy()   <-- would return error
    print(z)
    print("prev device:", z.device)
    z = z.to("cpu")
    print("changed to: ", z.device)
    z.numpy()
    print(z)

print("\r\n===== gradient (more on that in next tutorial) =====\r\n")

x = torch.ones(5, requires_grad=True) #telling pytorch that it will need to
                                      #calculate the gradients for this tensor
                                      #later in our optimisation steps
print(x)