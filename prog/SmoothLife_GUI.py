"""
    File:     SmoothLife_GUI.py
    Author:   Vojtěch Kališ
    VUTBR ID: xkalis03

    Brief:    SmoothLife GUI PyQt5 implementation
"""

import torch
import os
import sys
from PyQt5.QtWidgets import QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QFrame, QApplication, QCheckBox, QLabel, QSpacerItem, QSizePolicy
from PyQt5.QtCore import Qt
from PyQt5.QtCore import QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from predictor_trainer import Future_state_predictor
from SmoothLife import CL_SmoothLife

"""
    SmoothLife GUI class
"""
class SmoothLife_GUI(QWidget):

    def __init__(self, parent=None):
        """ 
            Initialize necessities
        """
        super(SmoothLife_GUI, self).__init__(parent)

        #set output resolution
        self.width_s = 580
        self.height_s = 580

        #main simulation values - adjust for different outcomes
        #NOTE: the enclosed ML model was trained using these values and will likely need them to remained the same
        self.birth_interval = (0.28, 0.37)    #
        self.survival_interval = (0.27, 0.45) #
        self.sigmoid_widths = (0.03, 0.15)    #
        self.inner_radius = 6.0               #
        self.outer_radius_multiplier = 3.0    #

        #create SmoothLife instance
        self.sl = CL_SmoothLife(self.height_s, self.width_s, self.birth_interval, self.survival_interval, self.sigmoid_widths, self.inner_radius, self.outer_radius_multiplier)
        self.sl.init_space(self.inner_radius)

        #get size of simulation
        input_size = self.sl.grid.numel()
        output_size = input_size
        #initialize predictor
        self.predictor = Future_state_predictor(input_size, 64, output_size)
        #load pre-trained predictor
        if os.path.isfile("SmoothLife_predictor.pth"):
            if torch.cuda.is_available():
                self.predictor.load_state_dict(torch.load("SmoothLife_predictor.pth", map_location=torch.device('cuda')))
            else:
                self.predictor.load_state_dict(torch.load("SmoothLife_predictor.pth", map_location=torch.device('cpu')))
        else:
            print("ERROR: SmoothLife_predictor.pth not found - please train a Machine Learning model first.")
            print("       The functionality won't be available. Refer to the enclosed README for further information.")
        #put predictor into eval mode
        self.predictor.eval()
        #variable for checking if Machine Learning should be used or not (user input checkbox)
        self.ml_use = False

    #the actual GUI stuff
        self.setWindowTitle('SmoothLife GUI')
        self.setGeometry(100, 100, 1200, 600)
        #create the main layout for the SmoothLife widget
        self.main_layout = QHBoxLayout(self)

        #create the left and right layouts
        self.left_widget = QWidget()
        self.right_widget = QWidget()
        self.left_layout = QVBoxLayout(self.left_widget)
        self.right_layout = QVBoxLayout(self.right_widget)

        self.create_left_window()
        self.create_right_window()

# Left side (Simulation window)
    def create_left_window(self):
        """
            Function to create left side of the simulation window
        """
    #figure (simulation)
        self.fig, self.ax = plt.subplots(figsize=(self.width_s / 100, self.height_s / 100), dpi=100)
        self.im = self.ax.imshow(self.sl.grid.squeeze().detach().numpy(), cmap=plt.get_cmap("gray"), aspect="equal", extent=[0, self.width_s, 0, self.height_s])

        self.fig.patch.set_visible(False)
        self.ax.axis('off')
        #create the graphics canvas
        self.canvas = FigureCanvas(self.fig)
        #set the size of the simulation window (FigureCanvas)
        self.canvas.setFixedSize(self.width_s, self.height_s)
        #add simulation window to main window (layout)
        self.left_layout.addWidget(self.canvas)

        #initialize timer for simulation updates
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_board)

    #buttons
        #create a container widget for the buttons
        button_container = QFrame(self)
        button_container.setFrameShape(QFrame.NoFrame)
        button_container.setMaximumWidth(self.width_s)

        #create the main buttons layout
        self.left_button_layout = QVBoxLayout(button_container)

        #create two horizontal layouts for buttons
        self.left_horizontal_layout1 = QHBoxLayout()
        self.left_horizontal_layout2 = QHBoxLayout()
        #add the horizontal button layouts to the main buttons layout
        self.left_button_layout.addLayout(self.left_horizontal_layout1)
        self.left_button_layout.addLayout(self.left_horizontal_layout2)

        #call a function to create the buttons
        self.create_left_smoothlife_buttons()

        #add the button container to the layout
        self.left_layout.addWidget(button_container)
        #add the button layout to the main layout
        self.left_layout.addLayout(self.left_button_layout)

        self.main_layout.addWidget(self.left_widget)

    def create_right_window(self):
        """
            Function to create right side of the simulation window
        """
        #create a container widget for the buttons
        button_container = QFrame(self)
        button_container.setFrameShape(QFrame.NoFrame)
        button_container.setMaximumWidth(self.width_s)

        #create the main buttons layout
        self.right_button_layout = QVBoxLayout(button_container)
        #adjust spacing between horizontal layouts
        self.right_button_layout.setSpacing(10)

        #spacers for buttons centering
        spacer_top = QSpacerItem(10, 70, QSizePolicy.Expanding, QSizePolicy.Minimum)  # Vertical space
        spacer_bottom = QSpacerItem(10, 40, QSizePolicy.Expanding, QSizePolicy.Minimum)  # Vertical space
        #add spacers and other layouts to the main layout
        #first add top spacer to the layour, then the button layout, then bottom spacer
        self.right_layout.addItem(spacer_top)
        self.right_layout.addLayout(self.right_button_layout)
        self.right_layout.addItem(spacer_bottom)
        #set margins for the main layout
        self.right_layout.setContentsMargins(10, 10, 10, 10)
        #set alignment
        self.right_layout.setAlignment(Qt.AlignTop)

        #create two horizontal layouts for buttons
        self.right_horizontal_layout1 = QHBoxLayout()
        self.right_horizontal_layout2 = QHBoxLayout()
        #add the horizontal button layouts to the main buttons layout
        self.right_button_layout.addLayout(self.right_horizontal_layout1)
        self.right_button_layout.addLayout(self.right_horizontal_layout2)

        #call a function to create the buttons
        self.create_right_smoothlife_buttons()

        #add the button container to the layout
        self.right_layout.addWidget(button_container)
        #add the button layout to the main layout
        self.right_layout.addLayout(self.right_button_layout)

        self.main_layout.addWidget(self.right_widget)

    def create_left_smoothlife_buttons(self):
        """ 
            Function for left buttons creation
        """
        self.start_button = QPushButton("START", self)
        self.start_button.clicked.connect(self.start)

        self.stop_button = QPushButton("STOP", self)
        self.stop_button.clicked.connect(self.stop)

        self.restart_button = QPushButton("RESTART", self)
        self.restart_button.clicked.connect(self.restart)

        self.reshuffle_button = QPushButton("RESHUFFLE", self)
        self.reshuffle_button.clicked.connect(self.reshuffle)

        self.left_horizontal_layout1.addWidget(self.start_button)
        self.left_horizontal_layout1.addWidget(self.stop_button)
        self.left_horizontal_layout1.addWidget(self.restart_button)
        self.left_horizontal_layout1.addWidget(self.reshuffle_button)

    def create_right_smoothlife_buttons(self):
        """ 
            Function for right buttons creation
        """
        self.ml_checkbox = QCheckBox('Use Machine Learning', self)
        if os.path.isfile("SmoothLife_predictor.pth"):
            self.ml_checkbox.setEnabled(True)
        else:
            self.ml_checkbox.setEnabled(False)
        self.ml_checkbox.stateChanged.connect(self.ml_checkbox_handler)

        self.alpha_label = QLabel("Alpha: 0.00000", self)
        self.step_label = QLabel("Step: 0", self)

        self.right_horizontal_layout1.addWidget(self.ml_checkbox)

        self.right_horizontal_layout2.addWidget(self.step_label)
        self.right_horizontal_layout2.addWidget(self.alpha_label)

    def update_alpha_label(self):
        """
            Function for updating displayed alpha value
        """
        #extract the value from the tensor and convert it to a float
        #get the scalar value from the tensor
        avg_alpha_value = self.sl.alpha.mean().item()
        #format as a float with 5 decimal places
        self.alpha_label.setText(f"Alpha: {avg_alpha_value:.5f}")

    def update_step_label(self):
        """
            Function for updating displayed step number
        """
        self.step_label.setText(f"Step: {self.sl.step}")

    def ml_checkbox_handler(self, state):
        """ 
            Function to handle checkbox functionality (gets called when checkbox is clicked)
        """
        if state == 2:
            self.ml_use = True
        else:
            self.ml_use = False
            self.sl.alpha = torch.zeros_like(self.sl.alpha)
            self.update_alpha_label()

    def start(self):
        """
            Start the simulation
        """
        self.timer.start(100)


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
        #reset stats
        self.sl.alpha = torch.zeros((self.sl.height, self.sl.width))
        self.update_alpha_label()
        self.sl.step = 0
        self.update_step_label()

    def reshuffle(self):
        """
            Restart the simulation, randomize it (and stop it until Start is used)
        """
        self.stop()
        self.sl.grid = torch.zeros((self.sl.height, self.sl.width))
        self.sl.init_space(self.inner_radius)
        self.sl.grid_save = self.sl.grid.clone()
        self.update_board()
        #reset stats
        self.sl.alpha = torch.zeros((self.sl.height, self.sl.width))
        self.update_alpha_label()
        self.sl.step = 0
        self.update_step_label()

    def update_board(self):
        """
            Function for board state update handling
        """
        #if 'use Machine Learning' checkbox is on True
        if self.ml_use == True:
            #make SmoothLife step with ML
            self.sl.forward_ML(self.predictor)
            self.update_alpha_label()
            self.update_step_label()
        else:
            #else make regular SmoothLife step
            self.sl.forward()
            self.update_step_label()

        #update the image
        self.im.set_array(self.sl.grid.squeeze().numpy())
        self.fig.canvas.draw()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = SmoothLife_GUI()
    window.show()
    sys.exit(app.exec_())