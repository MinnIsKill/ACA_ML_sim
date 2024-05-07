"""
    File:     Lenia.py
    Author:   Vojtěch Kališ
    VUTBR ID: xkalis03

    Brief:    Lenia PyTorch implementation

    based on the paper: Lenia — Biology of Artificial Life
                        Bert Wang-Chak Chan, 2019, https://arxiv.org/pdf/1812.05433
            as well as: https://github.com/OpenLenia/Lenia-Tutorial
"""

import torch
import numpy as np
#import scipy.signal

#this pattern is 20x20, we need to scale it up
orbium_pattern = torch.tensor([[0,0,0,0,0,0,0.1,0.14,0.1,0,0,0.03,0.03,0,0,0.3,0,0,0,0], 
         [0,0,0,0,0,0.08,0.24,0.3,0.3,0.18,0.14,0.15,0.16,0.15,0.09,0.2,0,0,0,0], 
         [0,0,0,0,0,0.15,0.34,0.44,0.46,0.38,0.18,0.14,0.11,0.13,0.19,0.18,0.45,0,0,0], 
         [0,0,0,0,0.06,0.13,0.39,0.5,0.5,0.37,0.06,0,0,0,0.02,0.16,0.68,0,0,0], 
         [0,0,0,0.11,0.17,0.17,0.33,0.4,0.38,0.28,0.14,0,0,0,0,0,0.18,0.42,0,0], 
         [0,0,0.09,0.18,0.13,0.06,0.08,0.26,0.32,0.32,0.27,0,0,0,0,0,0,0.82,0,0], 
         [0.27,0,0.16,0.12,0,0,0,0.25,0.38,0.44,0.45,0.34,0,0,0,0,0,0.22,0.17,0], 
         [0,0.07,0.2,0.02,0,0,0,0.31,0.48,0.57,0.6,0.57,0,0,0,0,0,0,0.49,0], 
         [0,0.59,0.19,0,0,0,0,0.2,0.57,0.69,0.76,0.76,0.49,0,0,0,0,0,0.36,0], 
         [0,0.58,0.19,0,0,0,0,0,0.67,0.83,0.9,0.92,0.87,0.12,0,0,0,0,0.22,0.07], 
         [0,0,0.46,0,0,0,0,0,0.7,0.93,1,1,1,0.61,0,0,0,0,0.18,0.11], 
         [0,0,0.82,0,0,0,0,0,0.47,1,1,0.98,1,0.96,0.27,0,0,0,0.19,0.1], 
         [0,0,0.46,0,0,0,0,0,0.25,1,1,0.84,0.92,0.97,0.54,0.14,0.04,0.1,0.21,0.05], 
         [0,0,0,0.4,0,0,0,0,0.09,0.8,1,0.82,0.8,0.85,0.63,0.31,0.18,0.19,0.2,0.01], 
         [0,0,0,0.36,0.1,0,0,0,0.05,0.54,0.86,0.79,0.74,0.72,0.6,0.39,0.28,0.24,0.13,0], 
         [0,0,0,0.01,0.3,0.07,0,0,0.08,0.36,0.64,0.7,0.64,0.6,0.51,0.39,0.29,0.19,0.04,0], 
         [0,0,0,0,0.1,0.24,0.14,0.1,0.15,0.29,0.45,0.53,0.52,0.46,0.4,0.31,0.21,0.08,0,0], 
         [0,0,0,0,0,0.08,0.21,0.21,0.22,0.29,0.36,0.39,0.37,0.33,0.26,0.18,0.09,0,0,0], 
         [0,0,0,0,0,0,0.03,0.13,0.19,0.22,0.24,0.24,0.23,0.18,0.13,0.05,0,0,0,0], 
         [0,0,0,0,0,0,0,0,0.02,0.06,0.08,0.09,0.07,0.05,0.01,0,0,0,0,0]])
#scale up the pattern to size=120, so 120x120. We use bilinear interpolation to do the job
orbium_pattern_scaled = torch.nn.functional.interpolate(orbium_pattern.unsqueeze(0).unsqueeze(0), size=200, mode='bilinear', align_corners=False)
#remove extra dimensions
orbium_pattern_scaled = orbium_pattern_scaled.squeeze()



class CL_Lenia(torch.nn.Module):
    def __init__(self, height, width):
        """ 
            Initialize necessities
        """
        super(CL_Lenia, self).__init__()
        self.width = width
        self.height = height
        """create grid"""
        self.grid_size = 64
        self.mid = self.width // 2
        self.scale = 2.2

        self.grid = torch.zeros((self.height, self.width))
        self.grid_save = self.grid.clone()
        """initialize alpha for ML"""
        self.alpha = torch.zeros((self.height, self.width))
        """initialize step counter"""
        self.step = 0
        """initialize parameters and grid"""
        self.init_space(0) #default

    def bell(self, x, m, s):
        """
            function to create a kernel for convolution where kernel has a smooth distribution,
            returns a Gaussian (or normal) distribution, centered at "m", with a spread defined by "s"
        """
        return np.exp(-((x - m) / s) ** 2 / 2)

    def growth(self, U):
        """
            determines how cells in the cellular automaton should grow or shrink based on the 
            "potential" derived from convolution with the kernel
        """
        return self.bell(U, self.mean, self.spread) * 2 - 1

    def forward(self, asynchronous):
        """
            Lenia step
        """
        if asynchronous == True:
            #randomly determine whether each cell should be updated or not
            update_mask = torch.rand_like(self.grid) > 0.3 #cells with values > 0.3 will be updated (30% chance)
        else:
            update_mask = torch.ones_like(self.grid, dtype=torch.bool) #all cells will be updated
        #compute local neighborhood using convolution
        #first compute the FFT of grid, then multiply it with pre-prepared kernel in the 
        #frequency domain and transform it back to the spatial domain using inverse FFT
        U_fft = torch.fft.fftn(self.grid)
        U_fft = torch.fft.ifftn(U_fft * self.fK)
        #now get the real part
        U = torch.real(U_fft)

        #apply rules, update only the cells where update_mask is True
        self.grid[update_mask] = torch.clamp(self.grid[update_mask] + 1 / self.update_frequency * self.growth(U)[update_mask], 0, 1)

        self.step += 1

        return self.grid
    
    def forward_ML(self, asynchronous, predictor):
        """
            Lenia step with Machine Learning
        """
        if asynchronous == True:
            #randomly determine whether each cell should be updated or not
            update_mask = torch.rand_like(self.grid) > 0.3 #cells with values > 0.3 will be updated (30% chance)
        else:
            update_mask = torch.ones_like(self.grid, dtype=torch.bool) #all cells will be updated
        #compute local neighborhood using convolution
        #first compute the FFT of grid, then multiply it with pre-prepared kernel in the 
        #frequency domain and transform it back to the spatial domain using inverse FFT
        U_fft = torch.fft.fftn(self.grid)
        U_fft = torch.fft.ifftn(U_fft * self.fK)
        #now get the real part
        U = torch.real(U_fft)

        #apply rules, update only the cells where update_mask is True
        self.grid[update_mask] = torch.clamp(self.grid[update_mask] + 1 / self.update_frequency * self.growth(U)[update_mask], 0, 1)

        #use the ML predictor to influence the state
        with torch.no_grad(): #we use no_grad to disable gradient calculation, so we don't mess up backward propagation
            predicted_state = predictor(self.grid.view(-1)).view(self.height, self.width)

        #combine original state with predicted state
        #calculate adaptive alpha (how much influence the predictor has on the outcome (0 = none, 1 = full))
        self.alpha = self.calculate_alpha(predicted_state)
        self.grid = (1 - self.alpha) * self.grid + self.alpha * predicted_state

        self.step += 1

        return self.grid
    
    def calculate_alpha(self, predicted_state, alpha_low=0.05, alpha_high=0.5):
        """
            Function to calculate the adaptive alpha
        """
        #calculate the difference between predicted state and current state
        state_difference = torch.abs(predicted_state - self.grid)

        #promote stability to next state, apply it
        stability_factor = 0.5 - state_difference
        alpha = alpha_low + (stability_factor * (alpha_high - alpha_low))

        #add a small boost to alpha for regions where current state is present
        alpha -= self.grid * 0.005  #boost to maintain existing patterns

        #apply a smoothing operation to avoid abrupt transitions
        smoothed_alpha = torch.nn.functional.conv2d(
            alpha.unsqueeze(0).unsqueeze(0),
            weight=torch.ones((1, 1, 3, 3)) / 9,  #3x3 averaging kernel for smoothing
            stride=1,
            padding=1,
        ).squeeze()

        #clamp alpha to ensure it stays within the allowed range
        alpha = torch.clamp(smoothed_alpha, alpha_low, alpha_high)

        return alpha

    def init_space(self, case):
        """
            populate simulation space
        """
        if case == 0: #Randomized (Smooth)
            self.kernel_radius = 20 * self.scale
            self.update_frequency = 10
            self.mean = 0.135
            self.spread = 0.015
            self.grid = torch.rand(self.height, self.width)

            self.create_kernel()
            
        elif case == 1: #Orbium
            self.kernel_radius = 58 * self.scale
            self.update_frequency = 10
            self.mean = 0.15
            self.spread = 0.015

            #get the dimensions of the Orbium pattern
            orbium_height, orbium_width = orbium_pattern_scaled.shape

            #calculate the starting positions to place the Orbium pattern in the center of the grid
            start_row = (self.height - orbium_height) // 2
            start_col = (self.width - orbium_width) // 2

            #create temporary grid as blank canvas
            temp_grid = torch.zeros((self.height, self.width))
            #place the Orbium pattern in the center of the grid
            temp_grid[start_row:start_row+orbium_height, start_col:start_col+orbium_width] = orbium_pattern_scaled
            #copy resulting grid to the real grid
            self.grid = temp_grid.clone()
            #make a copy (clone) of grid
            self.grid_save = self.grid.clone()

            self.create_kernel()

        elif case == 2: #Gaussian spot in the middle
            self.kernel_radius = 58 * self.scale
            self.update_frequency = 10
            self.mean = 0.135
            self.spread = 0.015
            self.grid = torch.rand(self.height, self.width)

            grid_tmp = np.ones((self.height, self.width))
            radius = 36
            y, x = np.ogrid[-self.height//2:self.height//2, -self.width//2:self.width//2]
            grid_tmp = np.exp(-0.5 * (x*x + y*y) / (radius*radius))
            self.grid = torch.tensor(grid_tmp)

            self.create_kernel()
        
        else: #???
            self.grid = torch.zeros((self.height,self.width))
        
    def create_kernel(self):
        """
            defines the kernel and prepares it for convolution by transforming it into 
            the frequency domain using the Fast Fourier Transform
        """
        #create two 2D grid tensors representing the x and y coordinates
        x_indices, y_indices = torch.meshgrid(torch.arange(-self.mid, self.mid), torch.arange(-self.mid, self.mid), indexing='ij')
        #compute the Euclidean distance from the origin for each point on the grid
        self.distances = torch.sqrt(x_indices**2 + y_indices**2) / self.kernel_radius
        #create the kernel
        self.kernel = (self.distances < 1) * self.bell(self.distances, 0.5, 0.15)

        #pre-calculate FFT of kernel
        self.fK = torch.fft.fftn(torch.fft.fftshift(self.kernel.clone().detach() / torch.sum(self.kernel.clone().detach())))