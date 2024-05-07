"""
    File:     SmoothLife.py
    Author:   Vojtěch Kališ
    VUTBR ID: xkalis03

    Brief:    SmoothLife PyTorch implementation

    based on the paper: Generalization of Conway's "Game of Life" to a continuous domain - SmoothLife
                        Stephan Rafler, 2011, https://arxiv.org/pdf/1111.1567.pdf
"""

import torch
import math

"""
    SmoothLife class
"""
class CL_SmoothLife(torch.nn.Module):
### INITIALIZATION ###
    def __init__(self, height, width, birth_range, survival_range, sigmoid_widths, inner_radius, outer_radius_multiplier):
        """ 
            Initialize necessities
        """
        super(CL_SmoothLife, self).__init__()
        self.width = width
        self.height = height
        """convolution"""
        self.inner_radius = inner_radius
        self.outer_radius_multiplier = outer_radius_multiplier
        self.compute_transforms((height, width))
        """rules"""
        #birth intervals
        self.birth_low, self.birth_high = birth_range
        #death intervals
        self.death_low, self.death_high = survival_range
        #M = "inner filling(area)" of the cell, N = "outer filling(area)" of the cell
        self.inner_filling, self.outer_filling = sigmoid_widths
        """clear"""
        self.grid = torch.zeros((self.height, self.width))
        self.grid_save = self.grid.clone()
        """initialize alpha for ML"""
        self.alpha = torch.zeros((self.height, self.width))
        """initialize step counter"""
        self.step = 0

### CONVOLUTION ###
    def compute_transforms(self, size):
        """
            Function to generate inner and outer transforms

            based on the math from: https://0fps.net/2012/11/19/conways-game-of-life-for-curved-surfaces-part-1/
        """
        #create two circles; the outer and inner one
        smaller_circle = self.generate_circle(size, self.inner_radius)
        bigger_circle = self.generate_circle(size, self.inner_radius * self.outer_radius_multiplier)

        #create the final areas
        #normalize the circles using torch.sum, then apply fft to transform them into the
        #frequency domain. Create the "ring" outer area by subtracting the smaller circle
        #from the bigger one
        self.inner_area = torch.fft.fft2(smaller_circle / torch.sum(smaller_circle))
        self.outer_area = torch.fft.fft2((bigger_circle - smaller_circle) / torch.sum(bigger_circle - smaller_circle))

    def generate_circle(self, size, radius):
        """
            Generate a circular mask with a given radius

            based on the math from: https://0fps.net/2012/11/19/conways-game-of-life-for-curved-surfaces-part-1/
        """
        #calculate coordinates
        y_coord, x_coord = torch.meshgrid(torch.arange(size[0]), torch.arange(size[1]), indexing='ij')
        #compute distances from the center
        distances = torch.sqrt((x_coord - size[1] / 2) ** 2 + (y_coord - size[0] / 2) ** 2)
        
        #create circle with 1s inside the radius and 0s outside
        circle = (distances <= radius).float()

        #roll the circle, centering it at extremes (this is better for convolution)
        circle = torch.roll(circle, size[0] // 2, dims=0)
        circle = torch.roll(circle, size[1] // 2, dims=1)

        #result will be a 2D grid with a circle drawn onto it with 1s
        return circle

### RULES APPLICATION ###
    def apply_rules(self, n, m):
        """ 
            Rules application as per the paper by Stephan Rafler
        """
        new_aliveness = self.sigma_2(n, self.sigma_m(self.birth_low, self.death_low, m), self.sigma_m(self.birth_high, self.death_high, m))

        return torch.clamp(new_aliveness, 0, 1)
    
    def sigma_1(self, x, a, filling):
        #x == m | n
        #a == 0.5 | thresholds (sigma_m results)
        #filling == self.N | self.M
        return (1.0 / (1.0 + torch.exp(-4.0 / filling * (x - a))))
    
    def sigma_2(self, x, a, b):
        #x == n
        #a, b == thresholds (sigma_m results)
        return (self.sigma_1(x, a, self.inner_filling) * (1.0 - self.sigma_1(x, b, self.inner_filling)))
    
    def sigma_m(self, x, y, m):
        #x == self.birth_high|low
        #y == self.death_high|low
        #m == m
        return  (x * (1.0 - self.sigma_1(m, 0.5, self.outer_filling)) + y * self.sigma_1(m, 0.5, self.outer_filling))
    
### GRID INITIALIZATION ###
    def init_space(self, inner_radius):
        """
            Populate simulation space with patches of 1s
        """
        grid_area = self.height * self.width
        #inner radius = size of patch, so for example inner radius = 6 is a 6x6 patch,
        #meaning the area of the patch is 6^2
        patch_area = int(inner_radius) ** 2
        #cover 30% of the grid space
        coverage_goal = 0.3
        #calculate how many patches of 1s to create
        estimated_patches_cnt = (grid_area / patch_area) * coverage_goal
        #round the final number of patches up
        patches_cnt = math.ceil(estimated_patches_cnt)

        for _ in range(patches_cnt):
            #get a random starting point for the patch
            start_row = torch.randint(0, self.height - int(inner_radius), (1,))
            start_col = torch.randint(0, self.width - int(inner_radius), (1,))
            #calculate the ending point for the patch
            end_row = start_row + int(inner_radius)
            end_col = start_col + int(inner_radius)

            #fill the patch on the grid
            self.grid[start_row:end_row, start_col:end_col] = 1
        self.grid_save = self.grid.clone()

### FORWARD PASS ###
    def forward(self):
        """
            SmoothLife step
        """
        #convert grid from the spatial domain to the frequency domain for convolution
        grid_transformed = torch.fft.fft2(self.grid)
        #calculate inner and outer area (do convolution)
        M_real = torch.real(torch.fft.ifft2(self.apply_multiplier(grid_transformed, self.inner_area)))
        N_real = torch.real(torch.fft.ifft2(self.apply_multiplier(grid_transformed, self.outer_area)))

        #apply SmoothLife rules with sigmoid smoothing
        self.grid = self.apply_rules(N_real, M_real)

        self.step += 1

        return self.grid
    
    def forward_ML(self, predictor):
        """
            SmoothLife step with MachineLearning predictor influence
        """
        #create a new grid
        grid_transformed = torch.fft.fft2(self.grid)
        #calculate inner and outer filling
        M_real = torch.real(torch.fft.ifft2(self.apply_multiplier(grid_transformed, self.inner_area)))
        N_real = torch.real(torch.fft.ifft2(self.apply_multiplier(grid_transformed, self.outer_area)))

        #apply SmoothLife rules without ML for the first part of the step
        self.grid = self.apply_rules(N_real, M_real)

        #use the ML predictor to influence the state
        with torch.no_grad(): #we use no_grad to disable gradient calculation, so we don't mess up backward propagation
            predicted_state = predictor(self.grid.view(-1)).view(self.height, self.width)

        '''
        avg_current_state_tensor = self.grid.mean().item()
        print(f"curr state avg value: {avg_current_state_tensor}")
        avg_predicted_state = predicted_state.mean().item()
        print(f"predictor avg value: {avg_predicted_state}")
        '''

        #combine original state with predicted state
        #calculate adaptive alpha (how much influence the predictor has on the outcome (0 = none, 1 = full))
        self.alpha = self.calculate_alpha(predicted_state)
        self.grid = (1 - self.alpha) * self.grid + self.alpha * predicted_state

        self.step += 1

        return self.grid
    
    def calculate_alpha(self, predicted_state, alpha_low=0.05, alpha_high=0.5):
        """
            Function to calculate alpha, meaning influence of Machine Learning on the next state
        """
        #calculate the difference between predicted state and current state
        state_difference = torch.abs(predicted_state - self.grid)

        #promote stability to next state, apply it
        stability_factor = 0.5 - state_difference
        alpha = alpha_low + (stability_factor * (alpha_high - alpha_low))

        #add a small boost to alpha for regions where current state is present
        alpha -= self.grid * 0.05  #boost to maintain existing patterns

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

    def apply_multiplier(self, grid_transformed, multiplier):
        return grid_transformed * multiplier