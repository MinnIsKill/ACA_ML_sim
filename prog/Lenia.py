"""
    File:     Lenia.py
    Author:   Vojtěch Kališ
    VUTBR ID: xkalis03

    Brief:    Lenia PyTorch implementation
"""

#TODO: MACHINE LEARNING WILL REQUIRE ASYNCHRONOUS TO BE ON (or off?)!!!

import sys
from PyQt5.QtWidgets import QWidget, QGraphicsScene, QGraphicsView, QPushButton, QVBoxLayout, QHBoxLayout, QFrame, QApplication, QCheckBox, QComboBox
from PyQt5.QtGui import QImage, QPixmap, QColor
from PyQt5.QtCore import QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from predictor_trainer import FutureStatePredictor
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
        super(CL_Lenia, self).__init__()
        self.width = width
        self.height = height

        self.grid_size = 64
        self.mid = self.width // 2
        self.scale = 2.2

        self.grid = torch.zeros((self.height, self.width))
        self.grid_save = self.grid.clone()

        self.init_space(0) #default

    def bell(self, x, m, s):
        return np.exp(-((x - m) / s) ** 2 / 2)

    def growth(self, U):
        return self.bell(U, self.mean, self.spread) * 2 - 1

    def forward(self, asynchronous):
        """
            Lenia step
        """
        if asynchronous == True:
            # Randomly determine whether each cell should be updated or not
            update_mask = torch.rand_like(self.grid) > 0.3 #cells with values > 0.05 will be updated
        else:
            update_mask = torch.rand_like(self.grid) >= 0 #all cells will be updated
        # Compute local neighborhood using convolution
        U_fft = torch.fft.fftn(self.grid)
        U_fft = torch.fft.ifftn(U_fft * self.fK)
        U = torch.real(U_fft)
        # Update only the cells where update_mask is True
        self.grid[update_mask] = torch.clamp(self.grid[update_mask] + 1 / self.update_frequency * self.growth(U)[update_mask], 0, 1)

        return self.grid

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

            self.param_recalc()
            
        elif case == 1: #Orbium
            self.kernel_radius = 58 * self.scale
            self.update_frequency = 10
            self.mean = 0.15
            self.spread = 0.015

            # Get the dimensions of the Orbium pattern
            orbium_height, orbium_width = orbium_pattern_scaled.shape

            # Calculate the starting positions to place the Orbium pattern in the center of the grid
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

            self.param_recalc()

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

            self.param_recalc()
        
        else: #???
            self.grid = torch.zeros((self.height,self.width))




        '''
        if case == "smooth":
            self.R = 10
            self.T = 10
            self.m = 0.135
            self.s = 0.015
            self.A = torch.rand(self.grid_size, self.grid_size)
        elif case == "orbium":
            self.R = 13
            self.T = 10
            self.m = 0.15
            self.s = 0.015
            self.cx, self.cy = 20, 20
            self.cells = [[0,0,0,0,0,0,0.1,0.14,0.1,0,0,0.03,0.03,0,0,0.3,0,0,0,0], 
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
                          [0,0,0,0,0,0,0,0,0.02,0.06,0.08,0.09,0.07,0.05,0.01,0,0,0,0,0]]
            
            self.A = torch.zeros([self.grid_size, self.grid_size])
            self.cells = torch.tensor(self.cells)
            self.C = self.cells
            self.C = torch.from_numpy(scipy.ndimage.zoom(self.C.numpy(), self.scale, order=0))
            self.R *= self.scale
            self.A[self.cx:self.cx + self.C.shape[0], self.cy:self.cy + self.C.shape[1]] = self.C
            '''
        
    def param_recalc(self):
        x_indices, y_indices = torch.meshgrid(torch.arange(-self.mid, self.mid), torch.arange(-self.mid, self.mid), indexing='ij')
        self.D = torch.sqrt(x_indices**2 + y_indices**2) / self.kernel_radius

        self.K = (self.D < 1) * self.bell(self.D, 0.5, 0.15)

        #pre-calculate FFT of kernel
        self.fK = torch.fft.fftn(torch.fft.fftshift(self.K.clone().detach() / torch.sum(self.K.clone().detach())))


"""
    Lenia GUI class
"""
class Lenia_GUI(QWidget):

    def __init__(self, parent=None):
        """ 
            Initialize necessities
        """
        super(Lenia_GUI, self).__init__(parent)

        #set output resolution
        width = 580
        height = 580

        #variable for checking if Machine Learning should be used or not (user input checkbox)
        self.ml_use = False
        #variable for checking if asynchronicity should be used or not (user input checkbox)
        self.asynchronous = True
        #variable for checking init board state
        #0 = default, 1 = orbium, 2 = ???
        self.init_state = 0

        #create Lenia instance
        self.lenia = CL_Lenia(height, width)
        self.lenia.init_space(self.init_state)

        #get size of simulation
        """!!!
        input_size = self.lenia.grid.numel()
        output_size = input_size

        #initialize predictor
        self.predictor = FutureStatePredictor(input_size, 64, output_size)
        #load pre-trained predictor
        self.predictor.load_state_dict(torch.load("Lenia_predictor.pth", map_location=torch.device('cpu')))
        #put predictor into eval mode
        self.predictor.eval()
        """

        #create the layout for the Lenia widget
        self.layout = QVBoxLayout(self)

        self.setLayout(self.layout)
        self.setWindowTitle('Lenia GUI')

        self.fig, self.ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
        self.im = self.ax.imshow(self.lenia.grid.squeeze().detach().numpy(), cmap=plt.get_cmap("gray"), aspect="equal", extent=[0, width, 0, height])

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
        self.create_Lenia_buttons()

        #add the button container to the layout
        self.layout.addWidget(button_container)
        #add the button layout to the main layout
        self.layout.addLayout(self.button_layout)

    def create_Lenia_buttons(self):
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

        """!!!
        self.ml_checkbox = QCheckBox('Use Machine Learning', self)
        self.ml_checkbox.stateChanged.connect(self.ml_checkbox_handler)
        """
        self.async_checkbox = QCheckBox('Use Asynchronous Updating', self)
        self.async_checkbox.stateChanged.connect(self.async_checkbox_handler)

        #create dropdown menu
        self.initstate_dd = QComboBox(self)
        #add items to dd menu
        self.initstate_dd.addItem("Randomized")
        self.initstate_dd.addItem("Orbium")
        self.initstate_dd.addItem("Gaussian")
        #set initial dd state
        self.initstate_dd.setCurrentIndex(0)
        #connect function to be called on selection change
        self.initstate_dd.currentIndexChanged.connect(self.init_state_dropdown_handler)

        self.horizontal_layout1.addWidget(self.start_button)
        self.horizontal_layout1.addWidget(self.stop_button)
        self.horizontal_layout1.addWidget(self.restart_button)
        self.horizontal_layout1.addWidget(self.reshuffle_button)
        
        """!!!
        self.horizontal_layout2.addWidget(self.ml_checkbox)
        """
        self.horizontal_layout2.addWidget(self.async_checkbox)
        self.horizontal_layout2.addWidget(self.initstate_dd)

    """
        Checkbox handlers
    """
    def async_checkbox_handler(self, state):
        if state == 2:
            self.asynchronous = True
        else:
            self.asynchronous = False

    """!!!
    def ml_checkbox_handler(self, state):

            #Function to handle checkbox functionality (gets called when checkbox is clicked)

        if state == 2:
            self.ml_use = True
        else:
            self.ml_use = False
    """
    
    def init_state_dropdown_handler(self, state):
        """
            Init board state dropdown handler
        """
        if state == 0:
            self.init_state = 0
        elif state == 1:
            self.init_state = 1
        elif state == 2:
            self.init_state = 2
        #else:
        self.reshuffle()

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
        self.lenia.grid = self.lenia.grid_save.clone()
        self.update_board()

    def reshuffle(self):
        """
            Restart the simulation, randomize it (and stop it until Start is used)
        """
        self.stop()
        self.lenia.grid = torch.zeros((self.lenia.height, self.lenia.width))
        self.lenia.init_space(self.init_state)
        self.lenia.grid_save = self.lenia.grid.clone()
        self.update_board()

    def update_board(self):
        """
            Function for board state update handling
        """
        """!!!
        #if 'use Machine Learning' checkbox is on True
        if self.ml_use == True:
            #make Lenia step with ML
            self.lenia.forward_ML(self.predictor)
        else:
            #else make regular Lenia step
            self.lenia.forward()
        """
        #make regular Lenia step, DELETE THIS WHEN YOU UNCOMMENT THE ABOVE!!!
        self.lenia.forward(self.asynchronous)

        #update the image
        self.im.set_array(self.lenia.grid.squeeze().numpy())
        self.fig.canvas.draw()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Lenia_GUI()
    ex.show()
    sys.exit(app.exec_())