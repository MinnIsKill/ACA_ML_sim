import torch

#check if CUDA is available
if torch.cuda.is_available():
    print("CUDA is available.")
else:
    print("CUDA is not available.")

#check if cuDNN is available
if torch.backends.cudnn.is_available:
    print("cuDNN is available.")
else:
    print("cuDNN is not available.")
