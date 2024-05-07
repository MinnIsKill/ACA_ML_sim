"""
    File:     predictor_trainer_run.py
    Author:   Vojtěch Kališ
    VUTBR ID: xkalis03

    Brief:    PyTorch machine learning predictor trainer run implementation
"""

import sys
import torch
import torch.nn as nn
import torch.optim as optim
from SmoothLife import CL_SmoothLife
from Lenia import CL_Lenia
from predictor_trainer import Future_state_predictor

if __name__ == "__main__":
#define SmoothLife instance
    #NOTE: adjust parameters if you've changed those in SmoothLife_GUI.py as well!!!
    #      (you'll get a "size mismatch" error when trying to use a model trained on different height and width)
    if len(sys.argv) == 2: #two arguments ("predictor_trainer_run.py and one other")
        if sys.argv[1] == "SmoothLife":
            model = CL_SmoothLife(height=580, width=580, birth_range=(0.28, 0.37), survival_range=(0.27, 0.45),
                               sigmoid_widths=(0.03, 0.15), inner_radius=6.0, outer_radius_multiplier=3.0)
            #initialize grid
            model.init_space(inner_radius=6.0)
        elif sys.argv[1] == "Lenia-smooth":
            model = CL_Lenia(height=580, width=580)
            #initiate grid
            model.init_space(0)
        elif sys.argv[1] == "Lenia-orbium":
            model = CL_Lenia(height=580, width=580)
            #initiate grid
            model.init_space(1)
        else:
            print("accepted arguments:   SmoothLife | Lenia-smooth | Lenia-orbium")
            exit(1)
    else:
        print("argument missing, or too many arguments received.") 
        print("argument specifying model is required.")
        print("accepted arguments:   SmoothLife | Lenia-smooth | Lenia-orbium")
        exit(1)

#define predictor (Future_state_predictor)
    #get size of grid
    input_size = model.grid.numel()
    output_size = input_size
    #initialize predictor
    predictor = Future_state_predictor(input_size, 64, output_size)
    #initialize optimizer
    optimizer = optim.Adam(predictor.parameters(), lr=0.001)
    criterion = nn.MSELoss()

#generate training data from the respective simulation
    #number of sample data
    num_samples = 2000
    train_inputs = []
    train_targets = []

    #load train inputs and targets
    print("loading train inputs...")
    if sys.argv[1] == "SmoothLife":
        for h in range(num_samples):
            if h % 500 == 0:
                print(f"    train input #{h}")
            current_state = model.grid.clone().detach()
            model.grid = model.forward()
            future_state = model.grid.clone().detach()

            train_inputs.append(current_state.view(1, -1))
            train_targets.append(future_state.view(1, -1))
    elif sys.argv[1] == "Lenia-smooth" or sys.argv[1] == "Lenia-orbium":
        for h in range(num_samples):
            if h % 500 == 0:
                print(f"    train input #{h}")
            current_state = model.grid.clone().detach()
            model.grid = model.forward(asynchronous=True)
            future_state = model.grid.clone().detach()

            train_inputs.append(current_state.view(1, -1))
            train_targets.append(future_state.view(1, -1))
    print("done")

    #move model and data to GPU, if possible
    print("trying to acces Cuda...")
    if torch.cuda.is_available():
        print("Cuda found, model moved to GPU")
        device = torch.device("cuda")
    else:
        print("Cuda not found, model moved to CPU")
        device = torch.device("cpu")
        
    predictor.to(device)

    #stack train inputs and targets (to lighten load)
    print("stacking train inputs and targets...")
    train_inputs = torch.stack(train_inputs).to(device)
    train_targets = torch.stack(train_targets).to(device)
    print("done")

#train the model
    old_loss = 0.0
    for epoch in range(300):
        optimizer.zero_grad()

        #process data in mini-batches (for faster processing)
        for i in range(0, len(train_inputs), 20):
            """
            if i == 0:
                print(f"epoch {epoch}: {i}")
            elif i % 100 == 0:
                print(f"               {i}")
            """
            batch_inputs = train_inputs[i:i + 20].view(20, -1)
            batch_targets = train_targets[i:i + 20].view(20, -1)

            outputs = predictor(batch_inputs)
            #calculate current loss
            loss = criterion(outputs, batch_targets)
            loss.backward()

        #if loss value went up, we're encountered overfitting ==> stop training
        if old_loss < loss.item() and old_loss != 0.0:
            print(f"overfitting encountered at epoch {epoch}, training stopped.")
            break
        old_loss = loss.item()

        optimizer.step()

        #print progress
        if epoch % 10 == 0:
            print(f'epoch {epoch}, loss: {loss.item()}')

#save the trained model
    if sys.argv[1] == "SmoothLife":
        torch.save(predictor.state_dict(), "SmoothLife_predictor.pth")
    elif sys.argv[1] == "Lenia-smooth":
        torch.save(predictor.state_dict(), "Lenia-smooth_predictor.pth")
    elif sys.argv[1] == "Lenia-orbium":
        torch.save(predictor.state_dict(), "Lenia-orbium_predictor.pth")
    print("model training completed and saved.")