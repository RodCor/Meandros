import generals.bezierPD as bezierPD
import cv2
import numpy as np


class Landmarks(bezierPD.BezierPD):

    def __init__(self, filename=None, ctrl_points = []):  # probar pasar como arg el objeto, no el path,
        # para ver si puedo actualizar el axis mientras grafico
        self.img = cv2.imread(filename)
        self.filename = filename
        self.ctrlPoints = ctrl_points
        self.INSERT_FLAG = False  # key: 'i'
        self.MOVE_FLAG = False  # key: 'm'
        self.ACTIVE_MOV_FLAG = False
        self.ind = None
        self.SHOW_LINES = False

    def circle_worker(self, ctrl, k):
        for p in self.ctrlPoints:
            cv2.circle(self.img2, (p[0], p[1]), self.thickness, self.RED, -1)

    def return_worker(self):
        return np.asarray(self.ctrlPoints)       


