"""
    File:     SmoothLife.py
    Author:   Vojtěch Kališ
    VUTBR ID: xkalis03
    Date:     27.1.2024

    Brief:    SmoothLife PyTorch implementation

    based on the paper: Generalization of Conway's "Game of Life" to a continuous domain - SmoothLife
                        Stephan Rafler, 2011, https://arxiv.org/pdf/1111.1567.pdf
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

#NOTE: The rules are basically 1:1 copies from the paper above (because I rushed this so it's done by midterm). 
#      It shall remain this way until I've gotten a better grasp of the underlying math, at which point they will be adjusted.
class BasicRules(torch.nn.Module):
    def __init__(self, birth_interval, survival_interval, sigmoid_widths):
        super(BasicRules, self).__init__()
        #birth intervals
        self.B1, self.B2 = birth_interval
        #death intervals
        self.D1, self.D2 = survival_interval
        #M = "inner filling(area)" of the cell, N = "outer filling(area)" of the cell
        self.N, self.M = sigmoid_widths

    def apply_rules(self, n, m):
        aliveness = 1.0 / (1.0 + torch.exp(-4.0 / self.M * (m - 0.5)))
        #if cell is dead && neighbour density between B1 and B2, cell becomes alive
        #if cell is alive && neighbour density between D1 and D2, cell stays alive
        threshold1 = (1.0 - aliveness) * self.B1 + aliveness * self.D1
        threshold2 = (1.0 - aliveness) * self.B2 + aliveness * self.D2

        sigmoid1 = 1.0 / (1.0 + torch.exp(-4.0 / self.N * (n - threshold1)))
        sigmoid2 = 1.0 - 1.0 / (1.0 + torch.exp(-4.0 / self.N * (n - threshold2)))
        
        new_aliveness = sigmoid1 * sigmoid2
        return torch.clamp(new_aliveness, 0, 1)

class Convolution(torch.nn.Module):
    def __init__(self, size, inner_radius, outer_radius_multiplier):
        super(Convolution, self).__init__()
        self.inner_radius = inner_radius
        self.outer_radius_multiplier = outer_radius_multiplier
        self.compute_transforms(size)

    def compute_transforms(self, size):
        # Convert size tuple to a tensor and then to a numpy array
        size_tensor = torch.tensor(size)
        size_np = size_tensor.numpy()

        inner_circle = self.generate_circle(size_np, self.inner_radius)
        outer_circle = self.generate_circle(size_np, self.inner_radius * self.outer_radius_multiplier)
        annulus = outer_circle - inner_circle
        inner_circle = inner_circle / torch.sum(inner_circle)
        annulus = annulus / torch.sum(annulus)
        self.M = torch.fft.fft2(inner_circle)
        self.N = torch.fft.fft2(annulus)

    def generate_circle(self, size, radius, roll=True, logres=None):
        # Convert size tuple to a tensor and then to a numpy array
        size_tensor = torch.tensor(size)

        yy, xx = torch.meshgrid(torch.arange(size[0]), torch.arange(size[1]), indexing='ij')
        radiuses = torch.sqrt((xx - size[1] / 2) ** 2 + (yy - size[0] / 2) ** 2)
        if logres is None:
            logres = torch.log2(torch.min(size_tensor).to(torch.float32))
        with torch.no_grad():
            logistic = 1 / (1 + torch.exp(logres * (radiuses - radius)))
        if roll:
            logistic = torch.roll(logistic, size[0] // 2, dims=0)
            logistic = torch.roll(logistic, size[1] // 2, dims=1)
        return logistic

class SmoothLife(torch.nn.Module):
    def __init__(self, height, width, birth_range, survival_range, sigmoid_widths, inner_radius, outer_radius_multiplier):
        super(SmoothLife, self).__init__()
        self.width = width
        self.height = height
        self.convolution = Convolution((height, width), inner_radius, outer_radius_multiplier)
        self.rules = BasicRules(birth_range, survival_range, sigmoid_widths)
        #clear
        self.field = torch.zeros((self.height, self.width))

    def forward(self):
        """
            SmoothLife step
        """
        field_transformed = torch.fft.fft2(self.field)
        M_buffer = self.apply_multiplier(field_transformed, self.convolution.M)
        N_buffer = self.apply_multiplier(field_transformed, self.convolution.N)
        M_real = torch.real(torch.fft.ifft2(M_buffer))
        N_real = torch.real(torch.fft.ifft2(N_buffer))
        self.field = self.rules.apply_rules(N_real, M_real)
        return self.field

    def apply_multiplier(self, field_transformed, multiplier):
        return field_transformed * multiplier

    def init_space(self, inner_radius, count=None, intensity=1):
        """
            populate simulation space
        """
        if count is None:
            count = int(self.width * self.height / ((inner_radius * 2) ** 2))
        for i in range(count):
            radius = int(inner_radius)
            r = torch.randint(0, self.height - radius, (1,))
            c = torch.randint(0, self.width - radius, (1,))
            self.field[r: r + radius, c: c + radius] = intensity


if __name__ == "__main__":
    #set output to 512x512 resolution
    width = 512
    height = 512

    #main simulation values - adjust for different outcomes
    birth_interval = (0.28, 0.37)    #
    survival_interval = (0.27, 0.45) #
    sigmoid_widths = (0.03, 0.15)    #
    inner_radius = 6.0               #
    outer_radius_multiplier = 3.0    #

    #create SmoothLife instance
    sl = SmoothLife(height, width, birth_interval, survival_interval, sigmoid_widths, inner_radius, outer_radius_multiplier)
    #initialize simulation space
    sl.init_space(inner_radius)

    fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
    im = ax.imshow(sl.field.squeeze().detach().numpy(), cmap=plt.get_cmap("gray"), aspect="equal", extent=[0, width, 0, height])

    #set figure frame invisible
    fig.patch.set_visible(False)
    ax.axis('off')

    def animate(frame):
        sl()
        im.set_array(sl.field.squeeze().numpy())
        return (im,)

    ani = animation.FuncAnimation(fig, animate, frames=200, interval=50, blit=True)

    #adjust layout to reduce white space
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    #save the animation as an MP4 file
    ani.save('SmoothLife.mp4', writer='ffmpeg', fps=10)