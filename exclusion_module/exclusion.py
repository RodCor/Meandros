import cv2
import numpy as np
import generals.windows_App as windows_App


class Exclusion(windows_App.App):

    def __init__(self, filename=None):
        # Initialize the parent class first
        super().__init__(filename=filename, parent=None)
        
        # Override with specific values for Exclusion
        self.img = cv2.imread(filename)
