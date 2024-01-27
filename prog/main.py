"""
    File:     main.py
    Author:   Vojtěch Kališ
    VUTBR ID: xkalis03

    Brief:    ACA & ML Cellular Behaviour Simulations Hub implementation
"""

import sys
import torch
from PyQt5.QtWidgets import QApplication, QMainWindow, QTabWidget, QVBoxLayout, QWidget, QLabel, QPushButton
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from Game_of_Life import CL_Game_of_Life
from SmoothLife import SmoothLife_GUI


class MainApp(QMainWindow):
    def __init__(self):
        super(MainApp, self).__init__()

        self.setWindowTitle("ACA & ML Cellular Behaviour Simulations Hub - In Progress")
        self.setFixedSize(1200, 800)

        self.central_widget = QTabWidget(self)
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout(self.central_widget)

        # Add the Game of Life widget as a tab
        self.game_of_life_widget = CL_Game_of_Life()
        self.central_widget.addTab(self.game_of_life_widget, "Game of Life")

        # Add the SmoothLife widget as a tab
        self.smoothlife_widget = SmoothLife_GUI()
        self.central_widget.addTab(self.smoothlife_widget, "SmoothLife")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_app = MainApp()
    main_app.show()
    sys.exit(app.exec_())