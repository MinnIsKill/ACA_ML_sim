"""
    File:     SmoothLife_ML_run.py
    Author:   Vojtěch Kališ
    VUTBR ID: xkalis03

    Brief:    SmoothLife PyTorch machine learning implementation
""" 

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from SmoothLife import CL_SmoothLife
from SmoothLife_predictor_trainer import FutureStatePredictor

def run_ml_simulation(predictor, initial_state, steps):
#define SmoothLife instance
    #NOTE: adjust height and width if you've changed those in SmoothLife.py as well!!!
    #      (you'll get a "size mismatch" error when trying to use a model trained on different height and width)
    sl = CL_SmoothLife(height=580, width=580, birth_range=(0.28, 0.37),
                    survival_range=(0.27, 0.45), sigmoid_widths=(0.03, 0.15),
                    inner_radius=6.0, outer_radius_multiplier=3.0)
    #initialize grid
    sl.init_space(inner_radius=6.0)

    #create plotting
    fig, ax = plt.subplots(figsize=(sl.width / 100, sl.height / 100), dpi=100)
    im = ax.imshow(sl.grid.squeeze().detach().numpy(), cmap=plt.get_cmap("gray"),
                   aspect="equal", extent=[0, sl.width, 0, sl.height])

    fig.patch.set_visible(False)
    ax.axis('off')

    #visual representation of the machine learning process
    text = ax.text(0.5, 0.95, '', transform=ax.transAxes, color='white', fontsize=12,
                ha='center', va='center', backgroundcolor='black')
    
    #print(predictor.state_dict())
    #input_tensor = torch.randn_like(initial_state)
    #predicted_output = predictor(input_tensor.view(-1)).view(initial_state.shape)
    #print(f'Input Sum: {input_tensor.sum()}, Predicted Sum: {predicted_output.sum()}')

    def animate(frame):
        #take a step forward
        sl.forward_ML(predictor)

        #convert the PyTorch tensor to a NumPy array for Matplotlib
        im.set_array(sl.grid.cpu().numpy())

        #update the visual representation
        text.set_text(f'ML Step: {frame}')

        return (im, text)

    ani = animation.FuncAnimation(fig, animate, frames=steps, interval=50, blit=True)

    #adjust layout to reduce white space
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    #save the animation as an MP4 file
    #ani.save('SmoothLife_ML.mp4', writer='ffmpeg', fps=10)

if __name__ == "__main__":
    #define initial state
    initial_state = torch.zeros((580, 580))

    #load the trained predictor
    input_size = initial_state.numel()
    output_size = input_size
    predictor = FutureStatePredictor(input_size, 64, output_size) #initialize predictor
    predictor.load_state_dict(torch.load("SmoothLife_predictor.pth", map_location=torch.device('cpu'))) #load pre-trained predictor
    predictor.eval() #put predictor into eval mode

    #move model parameters to GPU (if possible)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictor.to(device)

    #run the ML simulation
    run_ml_simulation(predictor, initial_state, steps=200)