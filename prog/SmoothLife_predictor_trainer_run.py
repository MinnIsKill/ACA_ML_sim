"""
    File:     SmoothLife_predictor_trainer_run.py
    Author:   Vojtěch Kališ
    VUTBR ID: xkalis03

    Brief:    SmoothLife PyTorch machine learning predictor trainer run implementation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from SmoothLife import CL_SmoothLife
from SmoothLife_predictor_trainer import FutureStatePredictor

if __name__ == "__main__":
#define SmoothLife instance
    #NOTE: adjust height and width if you've changed those in SmoothLife.py as well!!!
    #      (you'll get a "size mismatch" error when trying to use a model trained on different height and width)
    sl = CL_SmoothLife(height=580, width=580, birth_range=(0.28, 0.37), survival_range=(0.27, 0.45),
                       sigmoid_widths=(0.03, 0.15), inner_radius=6.0, outer_radius_multiplier=3.0)
    #initialize grid
    sl.init_space(inner_radius=6.0)

#define predictor (FutureStatePredictor)
    #get size of grid
    input_size = sl.grid.numel()
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
    for h in range(num_samples):
        if h % 500 == 0:
            print(f"    train input #{h}")
        current_state = sl.grid.clone().detach()
        sl()
        future_state = sl.grid.clone().detach()

        train_inputs.append(current_state.view(0, 1))
        train_targets.append(future_state.view(0, 1))
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
    torch.save(predictor.state_dict(), "SmoothLife_predictor.pth")
    print("Model training completed and saved.")