"""
    File:     Lenia_GUI.py
    Author:   Vojtěch Kališ
    VUTBR ID: xkalis03

    Brief:    Lenia GUI PyQt5 implementation
"""

import torch
import os
import sys
from PyQt5.QtWidgets import QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QFrame, QApplication, QCheckBox, QComboBox, QLabel, QSpacerItem, QSizePolicy
from PyQt5.QtCore import Qt
from PyQt5.QtCore import QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from predictor_trainer import Future_state_predictor
from Lenia import CL_Lenia

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
        self.width_s = 580
        self.height_s = 580

        #variable for checking if Machine Learning should be used or not (user input checkbox)
        self.ml_use = False
        #variable for checking if asynchronicity should be used or not (user input checkbox)
        self.asynchronous = True
        #variable for checking init board state
        #0 = default, 1 = orbium, 2 = ???
        self.init_state = 0

        #create Lenia instance
        self.lenia = CL_Lenia(self.height_s, self.width_s)
        self.lenia.init_space(self.init_state)

        #get size of simulation

        input_size = self.lenia.grid.numel()
        output_size = input_size

    #ML initializations
        if os.path.isfile("Lenia-smooth_predictor.pth"):
            #initialize smooth predictor
            self.predictor_smooth = Future_state_predictor(input_size, 64, output_size)
            #load pre-trained smooth predictor
            if torch.cuda.is_available():
                self.predictor_smooth.load_state_dict(torch.load("Lenia-smooth_predictor.pth", map_location=torch.device("cuda")))
            else:
                self.predictor_smooth.load_state_dict(torch.load("Lenia-smooth_predictor.pth", map_location=torch.device("cpu")))
            #put smooth predictor into eval mode
            self.predictor_smooth.eval()
        else:
            print("ERROR: Lenia-smooth_predictor.pth not found - please train a Machine Learning model first.")
            print("       The functionality won't be available. Refer to the enclosed README for further information.")
        
        if os.path.isfile("Lenia-orbium_predictor.pth"):
            #initialize orbium predictor
            self.predictor_orbium = Future_state_predictor(input_size, 64, output_size)
            #load pre-trained orbium predictor
            if torch.cuda.is_available():
                self.predictor_orbium.load_state_dict(torch.load("Lenia-orbium_predictor.pth", map_location=torch.device("cuda")))
            else:
                self.predictor_orbium.load_state_dict(torch.load("Lenia-orbium_predictor.pth", map_location=torch.device("cpu")))
            #put orbium predictor into eval mode
            self.predictor_orbium.eval()
        else:
            print("ERROR: Lenia-orbium_predictor.pth not found - please train a Machine Learning model first.")
            print("       The functionality won't be available. Refer to the enclosed README for further information.")

    #the actual GUI stuff
        self.setWindowTitle('Lenia GUI')
        self.setGeometry(100, 100, 1200, 600)
        #create the main layout for the Lenia widget
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
    #figure (simulation)
        self.fig, self.ax = plt.subplots(figsize=(self.width_s / 100, self.height_s / 100), dpi=100)
        self.im = self.ax.imshow(self.lenia.grid.squeeze().detach().numpy(), cmap=plt.get_cmap("gray"), aspect="equal", extent=[0, self.width_s, 0, self.height_s])

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
        self.create_left_Lenia_buttons()

        #add the button container to the layout
        self.left_layout.addWidget(button_container)
        #add the button layout to the main layout
        self.left_layout.addLayout(self.left_button_layout)

        self.main_layout.addWidget(self.left_widget)

    def create_right_window(self):
        #create a container widget for the buttons
        button_container = QFrame(self)
        button_container.setFrameShape(QFrame.NoFrame)
        button_container.setMaximumWidth(self.width_s)

        #create the main buttons layout
        self.right_button_layout = QVBoxLayout(button_container)
        #adjust spacing between horizontal layouts
        self.right_button_layout.setSpacing(10)

        #spacers for buttons centering
        spacer_top = QSpacerItem(10, 40, QSizePolicy.Expanding, QSizePolicy.Minimum)  # Vertical space
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

        #create horizontal layouts for buttons
        self.right_horizontal_layout1 = QHBoxLayout()
        self.right_horizontal_layout2 = QHBoxLayout()
        self.right_horizontal_layout3 = QHBoxLayout()
        #add the horizontal button layouts to the main buttons layout
        self.right_button_layout.addLayout(self.right_horizontal_layout1)
        self.right_button_layout.addLayout(self.right_horizontal_layout2)
        self.right_button_layout.addLayout(self.right_horizontal_layout3)

        #call a function to create the buttons
        self.create_right_Lenia_buttons()

        #add the button container to the layout
        self.right_layout.addWidget(button_container)
        #add the button layout to the main layout
        self.right_layout.addLayout(self.right_button_layout)

        self.main_layout.addWidget(self.right_widget)

    def create_left_Lenia_buttons(self):
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

        self.left_horizontal_layout1.addWidget(self.start_button)
        self.left_horizontal_layout1.addWidget(self.stop_button)
        self.left_horizontal_layout1.addWidget(self.restart_button)
        self.left_horizontal_layout1.addWidget(self.reshuffle_button)

    def create_right_Lenia_buttons(self):
        """ 
            Function for buttons creation
        """
        self.ml_checkbox = QCheckBox('Use Machine Learning', self)
        if os.path.isfile("Lenia-smooth_predictor.pth"):
            self.ml_checkbox.setEnabled(True)
        else:
            self.ml_checkbox.setEnabled(False)
        self.ml_checkbox.stateChanged.connect(self.ml_checkbox_handler)

        self.async_checkbox = QCheckBox('Use Asynchronous Updating', self)
        self.async_checkbox.setChecked(True)
        self.async_checkbox.stateChanged.connect(self.async_checkbox_handler)

        #create dropdown menu
        self.initstate_dd = QComboBox(self)
        #add items to dropdown menu
        self.initstate_dd.addItem("Smooth (Randomized)")
        self.initstate_dd.addItem("Orbium")
        self.initstate_dd.addItem("Gaussian")
        #set initial dropdown state
        self.initstate_dd.setCurrentIndex(0)
        #connect function to be called on selection change
        self.initstate_dd.currentIndexChanged.connect(self.init_state_dropdown_handler)

        self.alpha_label = QLabel("Alpha: 0.00000", self)
        self.step_label = QLabel("Step: 0", self)

        #first row
        self.right_horizontal_layout1.addWidget(self.initstate_dd)
        #second row
        self.right_horizontal_layout2.addWidget(self.ml_checkbox)
        self.right_horizontal_layout2.addWidget(self.async_checkbox)
        #third row
        self.right_horizontal_layout3.addWidget(self.step_label)
        self.right_horizontal_layout3.addWidget(self.alpha_label)

    def update_alpha_label(self):
        """
            Function for updating displayed alpha value
        """
        #extract the value from the tensor and convert it to a float
        #get the scalar value from the tensor
        avg_alpha_value = self.lenia.alpha.mean().item()
        #format as a float with 5 decimal places
        self.alpha_label.setText(f"Alpha: {avg_alpha_value:.5f}")

    def update_step_label(self):
        """
            Function for updating displayed step number
        """
        self.step_label.setText(f"Step: {self.lenia.step}")

    """
        Checkbox handlers
    """
    def async_checkbox_handler(self, state):
        if state == 2:
            self.asynchronous = True
        else:
            self.asynchronous = False

    def ml_checkbox_handler(self, state):
        if state == 2:
            self.ml_use = True
        else:
            self.ml_use = False
            self.lenia.alpha = torch.zeros_like(self.lenia.alpha)
            self.update_alpha_label()


    
    def init_state_dropdown_handler(self, state):
        """
            Init board state dropdown handler
        """
        #Smooth (randomized)
        if state == 0:
            self.init_state = 0
            if os.path.isfile("Lenia-smooth_predictor.pth"):
                self.ml_checkbox.setEnabled(True)
            else:
                self.ml_checkbox.setEnabled(False)

            #machine learning disabled by default
            self.ml_checkbox.setChecked(False)
            self.ml_use = False
            #asynchronous updating enabled by default
            self.async_checkbox.setChecked(True)
            self.asynchronous = True
        #Orbium
        elif state == 1: 
            self.init_state = 1
            if os.path.isfile("Lenia-orbium_predictor.pth"):
                self.ml_checkbox.setEnabled(True)
            else:
                self.ml_checkbox.setEnabled(False)

            #machine learning disabled by default
            self.ml_checkbox.setChecked(False)
            self.ml_use = False
            #asynchronous updating enabled by default
            self.async_checkbox.setChecked(True)
            self.asynchronous = True
        #Gaussian
        elif state == 2:
            self.init_state = 2
            self.ml_checkbox.setEnabled(False)

            #machine learning disabled by default
            self.ml_checkbox.setChecked(False)
            self.ml_use = False
            #asynchronous updating enabled by default
            self.async_checkbox.setChecked(True)
            self.asynchronous = True
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
        #reset stats
        self.lenia.alpha = torch.zeros((self.lenia.height, self.lenia.width))
        self.update_alpha_label()
        self.lenia.step = 0
        self.update_step_label()

    def reshuffle(self):
        """
            Restart the simulation, randomize it (and stop it until Start is used)
        """
        self.stop()
        self.lenia.grid = torch.zeros((self.lenia.height, self.lenia.width))
        self.lenia.init_space(self.init_state)
        self.lenia.grid_save = self.lenia.grid.clone()
        self.update_board()
        #reset stats
        self.lenia.alpha = torch.zeros((self.lenia.height, self.lenia.width))
        self.update_alpha_label()
        self.lenia.step = 0
        self.update_step_label()

    def update_board(self):
        """
            Function for board state update handling
        """
        #if 'use Machine Learning' checkbox is on True
        if self.ml_use == True:
            #make Lenia step with ML
            if self.init_state == 0: #Smooth
                self.lenia.forward_ML(self.asynchronous, self.predictor_smooth)
                self.update_alpha_label()
                self.update_step_label()
            elif self.init_state == 1: #Orbium
                self.lenia.forward_ML(self.asynchronous, self.predictor_orbium)
                self.update_alpha_label()
                self.update_step_label()
        else:
            #else make regular Lenia step
            self.lenia.forward(self.asynchronous)
            self.update_step_label()

        #update the image
        self.im.set_array(self.lenia.grid.squeeze().numpy())
        self.fig.canvas.draw()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Lenia_GUI()
    ex.show()
    sys.exit(app.exec_())