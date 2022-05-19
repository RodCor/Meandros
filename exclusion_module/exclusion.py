import cv2
import numpy as np
import generals.windows_App as windows_App

class Exclusion(windows_App.App):

    def __init__(self, filename=None):

        self.img = cv2.imread(filename)
        self.filename = filename
        self.ctrlPoints = []
        self.finalpoints = []
        self.INSERT_FLAG = False  # key: 'i'
        self.MOVE_FLAG = False  # key: 'm'
        self.ACTIVE_MOV_FLAG = False
        self.SHOW_LINES = False
        self.ind = None
        self.roi = None
