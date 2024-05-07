"""
    File:     main.py
    Author:   Vojtěch Kališ
    VUTBR ID: xkalis03

    Brief:    ACA & ML Cellular Behaviour Simulations Hub implementation
""" 

import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QTabWidget, QVBoxLayout

from Game_of_Life import CL_Game_of_Life
from SmoothLife_GUI import SmoothLife_GUI
from Lenia_GUI import Lenia_GUI


class Main_app(QMainWindow):
    def __init__(self):
        super(Main_app, self).__init__()

        self.setWindowTitle("ACA & ML Cellular Behaviour Simulations Hub")
        self.setGeometry(100, 100, 1200, 600)

        self.central_widget = QTabWidget(self)
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout(self.central_widget)
        self.central_widget.setLayout(self.layout)  # Set the layout for the central widget

        # Add the Game of Life widget as a tab
        self.game_of_life_widget = CL_Game_of_Life()
        self.central_widget.addTab(self.game_of_life_widget, "Game of Life")

        # Add the SmoothLife widget as a tab
        self.smoothlife_widget = SmoothLife_GUI()
        self.central_widget.addTab(self.smoothlife_widget, "SmoothLife")

        # Add the Lenia widget as a tab
        self.lenia_widget = Lenia_GUI()
        self.central_widget.addTab(self.lenia_widget, "Lenia")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_app = Main_app()
    main_app.show()
    sys.exit(app.exec_())