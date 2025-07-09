import generals.bezierPD as bezierPD
import cv2
import numpy as np


class Landmarks(bezierPD.BezierPD):

    def __init__(self, filename=None, ctrl_points=[]):
        """
        Landmarks class is a child class of BezierPD class. It is used
        to draw landmarks on the image and return the control points.
        Args:
            filename: str, path to the image file
            ctrl_points: list, list of control points
        """
        # Initialize the parent class first
        super().__init__(filename=filename, roi=None, ctrlPoints=ctrl_points)
        
        # Override with specific values for Landmarks
        self.img = cv2.imread(filename)
        self.SHOW_LINES = False

    def circle_worker(self, ctrl, k):
        for p in self.ctrlPoints:
            cv2.circle(self.img2, (p[0], p[1]), self.thickness, self.RED, -1)

    def return_worker(self):
        return np.asarray(self.ctrlPoints)
