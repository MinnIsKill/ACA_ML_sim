"""
    File:     SmoothLife_predictor_trainer_run.py
    Author:   Vojtěch Kališ
    VUTBR ID: xkalis03

    Brief:    SmoothLife PyTorch machine learning predictor trainer run implementation
""" 

import sys
import torch
import torch.nn as nn
import torch.optim as optim
from SmoothLife import CL_SmoothLife
from Lenia import CL_Lenia
from predictor_trainer import FutureStatePredictor

if __name__ == "__main__":
#define SmoothLife instance
    #NOTE: adjust height and width if you've changed those in SmoothLife.py as well!!!
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
        print("Argument missing, or too many arguments received.") 
        print("Argument specifying model is required.")
        print("Accepted arguments:   SmoothLife | Lenia-smooth | Lenia-orbium")
        exit(1)

#define predictor (FutureStatePredictor)
    #get size of grid
    input_size = model.grid.numel()
    output_size = input_size
    #initialize predictor
    predictor = FutureStatePredictor(input_size, 64, output_size)
    #initialize optimizer
    optimizer = optim.Adam(predictor.parameters(), lr=0.001)
    criterion = nn.MSELoss()

#generate training data from SmoothLife simulation
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

            train_inputs.append(current_state.view(0, 1))
            train_targets.append(future_state.view(0, 1))
    elif sys.argv[1] == "Lenia-smooth" or sys.argv[1] == "Lenia-orbium":
        num_steps = 3 #simulate Lenia's evolution for 3 steps
        for h in range(num_samples):
            if h % 500 == 0:
                print(f"    Train input #{h}")

            # Record the initial state
            current_state = model.grid.clone().detach()
            input_sequence = [current_state]

            # Simulate Lenia's evolution for multiple steps
            for _ in range(num_steps):
                model.grid = model.forward(asynchronous=True)
                current_state = model.grid.clone().detach()
                input_sequence.append(current_state)

            # Generate input-output pairs
            for i in range(len(input_sequence) - 1):
                train_inputs.append(input_sequence[i].view(1, -1))
                train_targets.append(input_sequence[i + 1].view(1, -1))
    print("done")

    #move model and data to GPU, if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
            print(f"Overfitting encountered at epoch {epoch}, training stopped.")
            break
        old_loss = loss.item()

        optimizer.step()

        #print progress
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')

#save the trained model
    if sys.argv[1] == "SmoothLife":
        torch.save(predictor.state_dict(), "SmoothLife_predictor.pth")
    elif sys.argv[1] == "Lenia-smooth":
        torch.save(predictor.state_dict(), "Lenia-smooth_predictor.pth")
    elif sys.argv[1] == "Lenia-orbium":
        torch.save(predictor.state_dict(), "Lenia-orbium_predictor.pth")
    print("Model training completed and saved.")