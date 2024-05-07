<!-- NECESSITIES --->
# Requirements
Install required libraries by running:
```bash
$ pip install -r requirements.txt
```
NOTE: this doesn't involve Python and PyTorch; for those, please refer to the "Installing Python and PyTorch" section of the thesis.

<!-- Simulations --->
# Running the Simulations
All simulations can be run separately, or using the main HUB. To run the main HUB, use:
```bash
$ python main.py
```
To run them separately, use:
```bash
$ python Game_of_Life.py
  or
$ python SmoothLife_GUI.py
  or
$ python Lenia_GUI.py
```

<!-- Machine Learning --->
# Training the Machine Learning model
To access the Machine Learning functionality, respective `.pth`, or trained models, need to exist. The repository comes with already pre-trained versions of these, but should you want to train them yourself, you can run the training process using:
```bash
$ python predictor_trainer_run.py SmoothLife
  or
$ python predictor_trainer_run.py Lenia-smooth
  or
$ python predictor_trainer_run.py Lenia-orbium
```
NOTE: The program will require a specific naming convention for these files. `predictor_trainer.py` handles this already, but in any case, the conventions are: `SmoothLife_predictor.pth`, `Lenia-smooth_predictor.pth` and `Lenia-orbium_predictor.pth`

<!-- FILES DESCRIPTION --->
# Contents
`main.py` - main HUB window, contains all simulations

---

`check_cuda.py` - script that checks whether CUDA and cuDNN are installed/accessible

`requirements.txt` - (partial) library requirements

---

`Game_of_Life.py` - contains Conway's Game of Life PyTorch implementation with PyQt5 GUI

`Lenia.py` - contains Lenia PyTorch implementation

`Lenia_GUI.py` - contains Lenia PyQt5 GUI implementation

`SmoothLife.py` - contains SmoothLife PyTorch implementation

`SmoothLife_GUI.py` - contains SmoothLife PyQt5 GUI implementation

---

`Lenia-orbium_predictor.pth` - pre-trained machine learning model for Lenia with orbium initial configuration

`Lenia-smooth_predictor.pth` - pre-trained machine learning model for Lenia with randomized initial configuration

`SmoothLife_predictor.pth` - pre-trained machine learning model for SmoothLife