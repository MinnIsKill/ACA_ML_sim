"""
    File:     Game_of_Life.py
    Author:   Vojtěch Kališ
    VUTBR ID: xkalis03

    Brief:    Conway's Game of Life PyTorch implementation

    Rules:    1. If a live cell has two or three live neighbours, it stays alive.
              2. If a live cell has fewer than two live neighbours, it dies (underpopulation).
              3. If a live cell has more than three live neighbours, it dies (overpopulation).
              4. If a dead cell has exactly three live neighbours, it becomes a live cell (reproduction).
"""

import numpy as np
from PyQt5.QtWidgets import QWidget, QGraphicsScene, QGraphicsView, QPushButton, QVBoxLayout, QHBoxLayout, QFrame, QApplication
from PyQt5.QtGui import QImage, QPixmap, QColor
from PyQt5.QtCore import QTimer
import sys
import torch

"""
    Game of Life class
""" 
class CL_Game_of_Life(QWidget):

    def __init__(self, parent=None):
        """ 
            Initialize necessities
        """
        #call the QObject's __init__ method
        super(CL_Game_of_Life, self).__init__(parent)

        #create the layout for the Game of Life widget
        self.layout = QVBoxLayout(self)

        #create the graphics scene and view
        self.scene = QGraphicsScene(self)
        self.view = QGraphicsView(self.scene)

        #set the size of the simulation window (QGraphicsView)
        self.view.setFixedSize(580, 580)

        #add the QGraphicsView to the layout
        self.layout.addWidget(self.view)

        #create a container widget for the buttons
        button_container = QFrame(self)
        button_container.setFrameShape(QFrame.NoFrame)
        button_container.setMaximumWidth(580)

        #create the main buttons layout
        self.button_layout = QVBoxLayout(button_container)

        #create two horizontal layouts for buttons
        self.horizontal_layout1 = QHBoxLayout()
        self.horizontal_layout2 = QHBoxLayout()
        #add the horizontal button layouts to the main buttons layout
        self.button_layout.addLayout(self.horizontal_layout1)
        self.button_layout.addLayout(self.horizontal_layout2)

        #call a function to create the buttons
        self.create_game_of_life_buttons()

        #add the button container to the layout
        self.layout.addWidget(button_container)
        #add the button layout to the main layout
        self.layout.addLayout(self.button_layout)

        #inititalize running attribute and set to False (to start simulation stopped)
        self.running = False
        #set update interval
        self.update_interval = 100
        #create convolution filter (Moore neighbourhood)
        self.neighbour_filter = torch.tensor([[1, 1, 1], 
                                              [1, 0, 1], 
                                              [1, 1, 1]]
                                            ).reshape(1, 1, 3, 3).float() #reshape the 3x3 matrix into a 4D tensor (1, 1, 3, 3)
                                                                          #(where the two 3s are height and width)
        #create grid
        self.grid = torch.randint(0, 2, (50, 50))
        #create grid clone as a savepoint
        self.grid_save = self.grid.clone()

        #initialize current_step var (for moving forwards and backwards in simulation through buttons)
        self.current_step = 0

        #initialize timer for simulation updates
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_board)

        #set cell size for simulation output
        self.cell_size = 11.55

    def create_game_of_life_buttons(self):
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

        self.step_forward_button = QPushButton("Step Forward", self)
        self.step_forward_button.clicked.connect(self.step_forward)

        self.step_backward_button = QPushButton("Step Backward", self)
        self.step_backward_button.clicked.connect(self.step_backward)

        self.horizontal_layout1.addWidget(self.start_button)
        self.horizontal_layout1.addWidget(self.stop_button)
        self.horizontal_layout1.addWidget(self.restart_button)
        self.horizontal_layout1.addWidget(self.reshuffle_button)

        self.horizontal_layout2.addWidget(self.step_forward_button)
        self.horizontal_layout2.addWidget(self.step_backward_button)

    def start(self):
        """
            Start the simulation
        """
        self.running = True
        self.update_board()

    def stop(self):
        """
            Stop the simulation
        """
        self.running = False

    def restart(self):
        """
            Restart the simulation (and stop it until Start is used)
        """
        self.stop()
        self.grid = self.grid_save.clone()
        self.image_update()

        #we need to reset current_step counter
        self.current_step = 0

    def reshuffle(self):
        """
            Restart the simulation, randomize it (and stop it until Start is used)
        """
        self.stop()
        self.grid = torch.randint(0, 2, (50, 50))
        self.current_step = 0
        self.grid_save = self.grid.clone()
        self.image_update()

    def step_forward(self):
        """
            Make one step forward in simulation
        """
        #stop the simulation loop
        self.running = False

        self.conway_step()
        self.image_update()

    def step_backward(self):
        """
            Make one step backward in simulation
        """
        #stop the simulation loop
        self.running = False
        
        if self.current_step > 0:
            #execute steps until the desired step
            desired_step = self.current_step - 1
            #revert to the previous step by restarting the simulation 
            #(the "restart" function will also handle resetting the current_step counter)
            self.restart()
            #move to the desired step
            for _ in range(desired_step):
                self.conway_step()
            self.image_update()

    def update_board(self):
        """
            Function for board state update handling
        """
        if self.running:
            #make one time step
            self.conway_step()

            #update visuals
            self.image_update()

            #use the .timer function and set interval to create a recursive loop
            self.timer.start(self.update_interval)

    def image_update(self):
        """
            Update processed image based on current grid state
        """
        #convert tensor to numpy array (and scale to 0-255 for display)
        img = np.array(self.grid) * 255
        #convert single channel to three channels
        img = np.stack((img,) * 3, axis=-1)
        #make img black-and-white (by converting values above 128 to 255 (white), and below or equal to 128 to 0 (black))
        img = np.where(img > 128, 255, 0)

        #get the height and width of the image
        height, width = img.shape[:2]
        #calculate the number of bytes per line in the image
        bytes_per_line = 3 * width
        #create a QImage from the NumPy array (in RGB888 format)
        qimage = QImage(img.data, width, height, bytes_per_line, QImage.Format_RGB888)
        #create a QPixmap from the QImage
        pixmap = QPixmap.fromImage(qimage)

        #clear the QGraphicsScene to prepare for new items
        self.scene.clear()

        #iterate through each cell in the grid
        for x in range(width):
            for y in range(height):
                #get the value of the cell in the grid
                cell_value = int(self.grid[y, x].item()) * 255
                #create a QColor using the cell value as the intensity for each RGB channel
                color = QColor(cell_value, cell_value, cell_value)
                #fill the pixmap with the color
                pixmap.fill(color)
                #create a QGraphicsPixmapItem and add it to the scene
                item = self.scene.addPixmap(pixmap)
                #set the position of the item based on the cell size
                item.setPos(x * self.cell_size, y * self.cell_size)

        #set the scene rectangle size to cover the entire grid
        self.view.setSceneRect(0, 0, width * self.cell_size, height * self.cell_size)
        #set the scene for the QGraphicsView
        self.view.setScene(self.scene)

    def conway_step(self):
        """
            Make one Conway Time Step

            Rules:
                1. If a live cell has two or three live neighbours, it stays alive.
                2. If a live cell has fewer than two live neighbours, it dies (underpopulation).
                3. If a live cell has more than three live neighbours, it dies (overpopulation).
                4. If a dead cell has exactly three live neighbours, it becomes a live cell (reproduction).

            NOTE: due to the nature of our updating (creating a new grid to compute on), 
                  we only really need to compute on old grid and fill the new one with cells which 
                  should stay/become alive (so, implement rules #1 and #4)
        """
        #further computations require the board contain float values
        self.grid = self.grid.float()

        #obtain count of alive neighbours for each individual cell using convolution (slight PyTorch magic involved)
        alive_neighbours_cnt = torch.nn.functional.conv2d(self.grid.unsqueeze(0), self.neighbour_filter, padding=1)
        #convert self.grid into a new tensor where each value is represented as a byte to ensure that
        #all elements in self.grid strictly conform to being 0 or 1 to avoid unexpected behavior
        alive = self.grid.byte()

        #implement Rule #1 (so, check if a cell is alive and has 2 or 3 alive neighbours)
        new_grid = (alive & (torch.eq(alive_neighbours_cnt.squeeze(), 2) |
                             torch.eq(alive_neighbours_cnt.squeeze(), 3))).float()
        #implement Rule #4 (more PyTorch magic)
        new_grid += (~alive & torch.eq(alive_neighbours_cnt.squeeze(), 3)).float()

        #update our grid, use squeeze() to remove any added dimensions (make it a 3x3 tensor again)
        self.grid = new_grid.clone().squeeze()

        #we just moved by one time step, increment number of steps taken
        self.current_step += 1


if __name__ == "__main__":
    app = QApplication(sys.argv)
    game_of_life = CL_Game_of_Life()
    game_of_life.show()
    sys.exit(app.exec_())