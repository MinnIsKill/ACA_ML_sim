"""
    File:     SmoothLife.py
    Author:   Vojtěch Kališ
    VUTBR ID: xkalis03

    Brief:    SmoothLife PyTorch implementation

    based on the paper: Generalization of Conway's "Game of Life" to a continuous domain - SmoothLife
                        Stephan Rafler, 2011, https://arxiv.org/pdf/1111.1567.pdf
"""

import sys
from PyQt5.QtWidgets import QWidget, QGraphicsScene, QGraphicsView, QPushButton, QVBoxLayout, QHBoxLayout, QFrame, QApplication, QCheckBox
from PyQt5.QtGui import QImage, QPixmap, QColor
from PyQt5.QtCore import QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from SmoothLife_predictor_trainer import FutureStatePredictor

""" 
    Class for basic rules (not advanced ones)
"""
#NOTE: The rules are basically 1:1 copies from the paper above. 
#      It shall remain this way until I've gotten a better grasp of the underlying math, at which point they will (maybe) be adjusted.
class Basic_rules(torch.nn.Module):

    def __init__(self, birth_interval, survival_interval, sigmoid_widths):
        """ 
            Initialize necessities
        """
        super(Basic_rules, self).__init__()
        #birth intervals
        self.B1, self.B2 = birth_interval
        #death intervals
        self.D1, self.D2 = survival_interval
        #M = "inner filling(area)" of the cell, N = "outer filling(area)" of the cell
        self.N, self.M = sigmoid_widths

    def apply_rules(self, n, m):
        """ 
            Basic rules application
        """
        #get current degree of being alive based on the sigmoid function
        aliveness = 1.0 / (1.0 + torch.exp(-4.0 / self.M * (m - 0.5)))
        #if cell is dead && neighbour density between B1 and B2, cell becomes alive
        #if cell is alive && neighbour density between D1 and D2, cell stays alive
        threshold1 = (1.0 - aliveness) * self.B1 + aliveness * self.D1
        threshold2 = (1.0 - aliveness) * self.B2 + aliveness * self.D2

        sigmoid1 = 1.0 / (1.0 + torch.exp(-4.0 / self.N * (n - threshold1)))
        sigmoid2 = 1.0 - 1.0 / (1.0 + torch.exp(-4.0 / self.N * (n - threshold2)))
        
        new_aliveness = sigmoid1 * sigmoid2
        return torch.clamp(new_aliveness, 0, 1)

""" 
    Convolution class
"""
class Convolution(torch.nn.Module):
    
    def __init__(self, size, inner_radius, outer_radius_multiplier):
        """ 
            Initialize necessities
        """
        super(Convolution, self).__init__()
        self.inner_radius = inner_radius
        self.outer_radius_multiplier = outer_radius_multiplier
        self.compute_transforms(size)

    def compute_transforms(self, size):
        #convert size tuple to a tensor and then to a numpy array
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
        #convert size tuple to a tensor
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

"""
    SmoothLife class
"""
class CL_SmoothLife(torch.nn.Module):

    def __init__(self, height, width, birth_range, survival_range, sigmoid_widths, inner_radius, outer_radius_multiplier):
        """ 
            Initialize necessities
        """
        super(CL_SmoothLife, self).__init__()
        self.width = width
        self.height = height
        self.convolution = Convolution((height, width), inner_radius, outer_radius_multiplier)
        self.rules = Basic_rules(birth_range, survival_range, sigmoid_widths)
        #clear
        self.grid = torch.zeros((self.height, self.width))
        self.grid_save = self.grid.clone()

    def forward(self):
        """
            SmoothLife step
        """
        #create a new grid
        grid_transformed = torch.fft.fft2(self.grid)
        #calculate inner and outer filling
        M_real = torch.real(torch.fft.ifft2(self.apply_multiplier(grid_transformed, self.convolution.M)))
        N_real = torch.real(torch.fft.ifft2(self.apply_multiplier(grid_transformed, self.convolution.N)))

        #apply SmoothLife rules
        self.grid = self.rules.apply_rules(N_real, M_real)

        return self.grid
    
    def forward_ML(self, predictor):
        """
            SmoothLife step with MachineLearning predictor influence
        """
        #create a new grid
        grid_transformed = torch.fft.fft2(self.grid)
        #calculate inner and outer filling
        M_real = torch.real(torch.fft.ifft2(self.apply_multiplier(grid_transformed, self.convolution.M)))
        N_real = torch.real(torch.fft.ifft2(self.apply_multiplier(grid_transformed, self.convolution.N)))

        #apply SmoothLife rules without ML for the first part of the step
        self.grid = self.rules.apply_rules(N_real, M_real)

        #use the ML predictor to influence the state
        with torch.no_grad(): #we use no_grad to disable gradient calculation, so we don't mess up backward propagation
            predicted_state = predictor(self.grid.view(-1)).view(self.height, self.width)

        #combine original state with predicted state
        alpha = 0.05  #how much influence the predictor has on the outcome (0 = none, 1 = full)
        self.grid = (1 - alpha) * self.grid + alpha * predicted_state

        return self.grid

    def apply_multiplier(self, grid_transformed, multiplier):
        return grid_transformed * multiplier

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
            self.grid[r: r + radius, c: c + radius] = intensity
        self.grid_save = self.grid.clone()

class SmoothLife_GUI(QWidget):

    def __init__(self, parent=None):
        """ 
            Initialize necessities
        """
        super(SmoothLife_GUI, self).__init__(parent)

        #set output resolution
        width = 580
        height = 580

        #main simulation values - adjust for different outcomes
        self.birth_interval = (0.28, 0.37)    #
        self.survival_interval = (0.27, 0.45) #
        self.sigmoid_widths = (0.03, 0.15)    #
        self.inner_radius = 6.0               #
        self.outer_radius_multiplier = 3.0    #

        #create SmoothLife instance
        self.sl = CL_SmoothLife(height, width, self.birth_interval, self.survival_interval, self.sigmoid_widths, self.inner_radius, self.outer_radius_multiplier)
        self.sl.init_space(self.inner_radius)

        #get size of simulation
        input_size = self.sl.grid.numel()
        output_size = input_size
        #initialize predictor
        self.predictor = FutureStatePredictor(input_size, 64, output_size)
        #load pre-trained predictor
        self.predictor.load_state_dict(torch.load("SmoothLife_predictor.pth", map_location=torch.device('cpu')))
        #put predictor into eval mode
        self.predictor.eval()
        #variable for checking if Machine Learning should be used or not (user input checkbox)
        self.ml_use = False

        #create the layout for the SmoothLife widget
        self.layout = QVBoxLayout(self)

        self.setLayout(self.layout)
        self.setWindowTitle('SmoothLife GUI')

        self.fig, self.ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
        self.im = self.ax.imshow(self.sl.grid.squeeze().detach().numpy(), cmap=plt.get_cmap("gray"), aspect="equal", extent=[0, width, 0, height])

        self.fig.patch.set_visible(False)
        self.ax.axis('off')

        #create the graphics canvas
        self.canvas = FigureCanvas(self.fig)
        #set the size of the simulation window (FigureCanvas)
        self.canvas.setFixedSize(width, height)
        #add simulation window to main window (layout)
        self.layout.addWidget(self.canvas)

        #initialize timer for simulation updates
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_board)

        #create a container widget for the buttons
        button_container = QFrame(self)
        button_container.setFrameShape(QFrame.NoFrame)
        button_container.setMaximumWidth(width)

        #create the main buttons layout
        self.button_layout = QVBoxLayout(button_container)

        #create two horizontal layouts for buttons
        self.horizontal_layout1 = QHBoxLayout()
        self.horizontal_layout2 = QHBoxLayout()
        #add the horizontal button layouts to the main buttons layout
        self.button_layout.addLayout(self.horizontal_layout1)
        self.button_layout.addLayout(self.horizontal_layout2)

        #call a function to create the buttons
        self.create_smoothlife_buttons()

        #add the button container to the layout
        self.layout.addWidget(button_container)
        #add the button layout to the main layout
        self.layout.addLayout(self.button_layout)

    def create_smoothlife_buttons(self):
        """ 
            Function for buttons creation
        """
        self.start_button = QPushButton("START", self)
        self.start_button.clicked.connect(self.start)

        self.stop_button = QPushButton("STOP", self)
        self.stop_button.clicked.connect(self.stop)

        self.restart_button = QPushButton("RESTART", self)
        self.restart_button.clicked.connect(self.restart)

        self.reshuffle_button = QPushButton("RESHUFFLE", self)
        self.reshuffle_button.clicked.connect(self.reshuffle)

        self.ml_checkbox = QCheckBox('Use Machine Learning', self)
        self.ml_checkbox.stateChanged.connect(self.ml_checkbox_handler)

        self.horizontal_layout1.addWidget(self.start_button)
        self.horizontal_layout1.addWidget(self.stop_button)
        self.horizontal_layout1.addWidget(self.restart_button)
        self.horizontal_layout1.addWidget(self.reshuffle_button)
        
        self.horizontal_layout2.addWidget(self.ml_checkbox)

    def ml_checkbox_handler(self, state):
        """ 
            Function to handle checkbox functionality (gets called when checkbox is clicked)
        """
        if state == 2:
            self.ml_use = True
        else:
            self.ml_use = False

    def start(self):
        """
            Start the simulation
        """
        self.timer.start(100)  # Adjust the interval as needed (in milliseconds)


    def stop(self):
        """
            Stop the simulation
        """
        self.timer.stop()

    def restart(self):
        """
            Restart the simulation (and stop it until Start is used)
        """
        self.stop()
        self.sl.grid = self.sl.grid_save.clone()
        self.update_board()

    def reshuffle(self):
        """
            Restart the simulation, randomize it (and stop it until Start is used)
        """
        self.stop()
        self.sl.grid = torch.zeros((self.sl.height, self.sl.width))
        self.sl.init_space(self.inner_radius)
        self.sl.grid_save = self.sl.grid.clone()
        self.update_board()

    def update_board(self):
        """
            Function for board state update handling
        """
        #if 'use Machine Learning' checkbox is on True
        if self.ml_use == True:
            #make SmoothLife step with ML
            self.sl.forward_ML(self.predictor)
        else:
            #else make regular SmoothLife step
            self.sl.forward()

        #update the image
        self.im.set_array(self.sl.grid.squeeze().numpy())
        self.fig.canvas.draw()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = SmoothLife_GUI()
    window.show()
    sys.exit(app.exec_())