"""
    File:     GoL.py
    Author:   Vojtěch Kališ
    VUTBR ID: xkalis03
    Date:     21.12.2023

    Brief:    Conway's Game of Life PyTorch implementation

    Rules:    1. If a live cell has two or three live neighbours, it stays alive.
              2. If a live cell has fewer than two live neighbours, it dies (underpopulation).
              3. If a live cell has more than three live neighbours, it dies (overpopulation).
              4. If a dead cell has exactly three live neighbours, it becomes a live cell (reproduction).
"""

# run XLaunch
#$ export DISPLAY=$(awk '/nameserver / {print $2; exit}' /etc/resolv.conf 2>/dev/null):0
#$ python GoL.py

import numpy as np

from PIL import Image, ImageTk #image convertion for TkInter
import tkinter as tk  #TkInter for visualization

import torch   #PyTorch, self-explanatory

import imageio #video creation
import io

"""
    Game of Life class
"""
class Game_of_Life:
    def __init__(self, root):
        """ 
            Initialize necessities
        """
        #initialize root
        self.root = root
        self.root.title("Conway's Game of Life")

        #prepare the TkInter canvas
        self.tk_canvas_prepare()

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

        #! ONLY FOR VIDEO GENERATION (comment the next line out otherwise)
        #self.frames = []

    def tk_canvas_prepare(self):
        """
            Prepare the TkInter canvas
        """
        #create the canvas and pack it (make it visible)
        self.canvas = tk.Canvas(self.root, width=500, height=500)
        self.canvas.pack()

        #create frame for buttons (so we can position them the way we want)
        self.button_frame = tk.Frame(self.root)
        self.button_frame.pack()

        #create the buttons
        self.start_button = tk.Button(self.button_frame, text="START", command=self.start)
        self.stop_button = tk.Button(self.button_frame, text="STOP", command=self.stop)
        self.restart_button = tk.Button(self.button_frame, text="RESTART", command=self.restart)

        #pack the buttons (make them visible)
        self.start_button.pack(side=tk.LEFT, padx=5)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        self.restart_button.pack(side=tk.LEFT, padx=5)

    def image_update(self):
        """
            Update processed image based on current grid state
        """
        #convert tensor to numpy array (and scale to 0-255 for display)
        img = np.array(self.grid) * 255

        #we need to convert the NumPy image into a PIL image for TkInter,
        pil_img = Image.fromarray(img.astype(np.uint8))
        #resize it to fit our canvas (using NEAREST interpolation for sharpness),
        pil_img = pil_img.resize((500, 500), Image.Resampling.NEAREST)
        #then convert the PIL image into TkInter image
        photo = ImageTk.PhotoImage(image=pil_img)

        #update displayed image
        self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        self.canvas.image = photo

    def conway_step(self):
        """
            Make one Conway Time Step

            Rules:
                1. If a live cell has two or three live neighbours, it stays alive.
                2. If a live cell has fewer than two live neighbours, it dies (underpopulation).
                3. If a live cell has more than three live neighbours, it dies (overpopulation).
                4. If a dead cell has exactly three live neighbours, it becomes a live cell (reproduction).

            NOTE: due to the nature of our updating (creating a new grid to computate on), 
                  we only really need to computate on old grid and fill the new one with cells which 
                  should stay/become alive (so, implement rules #1 and #4)
        """
        #further computations require the board contain float values
        self.grid = self.grid.float()

        #obtain count of alive neighbours for each individual cell using convolution (slight PyTorch magic involved)
        alive_neighbours_cnt = torch.nn.functional.conv2d(self.grid.unsqueeze(0), self.neighbour_filter, padding=1)
        #convert self.grid into a new tensor where each value is represented as a byte
        alive = self.grid.byte()

        #implement Rule #1 (more PyTorch magic)
        new_grid = (alive & (torch.eq(alive_neighbours_cnt.squeeze(), 2) | 
                             torch.eq(alive_neighbours_cnt.squeeze(), 3))).float()
        #implement Rule #4 (more PyTorch magic)
        new_grid += (~alive & torch.eq(alive_neighbours_cnt.squeeze(), 3)).float()

        #update our grid, use squeeze() to remove any added dimensions (make it a 3x3 tensor again)
        self.grid = new_grid.clone().squeeze()

    def update_board(self):
        """
            Recursive loop for simulation run
        """
        if self.running:
            #make one time step
            self.conway_step()

            #update visuals
            self.image_update()

            #! ONLY FOR VIDEO GENERATION (comment the following section out otherwise)
            """
            #convert canvas to PIL Image
            ps = self.canvas.postscript(colormode="color")
            img = Image.open(io.BytesIO(ps.encode('utf-8')))
            #convert the PIL Image to a numpy array
            frame = np.array(img)
            self.frames.append(frame)
            """

            #use the .after function to create a recursive loop
            self.root.after(self.update_interval, self.update_board)

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
        self.grid = torch.randint(0, 2, (50, 50))

        self.image_update()


if __name__ == "__main__":
    root = tk.Tk()
    GoL = Game_of_Life(root)
    GoL.stop() #start paused
    root.mainloop()

    #! ONLY FOR VIDEO GENERATION (comment the next line out otherwise)
    #imageio.mimsave('GoL.mp4', GoL.frames, fps=10)