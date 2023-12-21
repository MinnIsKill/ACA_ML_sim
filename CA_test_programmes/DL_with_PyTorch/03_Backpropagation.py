import torch

x = torch.tensor(1.0)
y = torch.tensor(2.0)

w = torch.tensor(1.0, requires_grad=True) #we will need the
                        #gradient for (loss)'

#forward pass and compute the loss
y_hat = w * x         #y_hat = w * x = 1 * 1 = 1
loss = (y_hat - y)**2 #loss = s^2 = (y_hat - y)^2 = (1 - 2)^2 = 1
print(loss)

#backward pass
loss.backward() #does the whole backpropagation (see: blue and 
                #orange calculations in the attached .png)
print(w.grad)   # -2 (for how we got the number, again, see .png)

###next steps, for example: 1) update weights
#                           2) do next forward and backward pass
#                           3) repeat for a couple iterations