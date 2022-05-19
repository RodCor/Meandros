import cv2
import numpy as np
import generals.windows_App as windows_App


class RoiSelection(windows_App.App):

    def __init__(self, filename=None, approx = None):

        self.img = cv2.imread(filename)
        self.filename = filename
        self.ctrlPoints = approx
        self.finalpoints = []
        self.INSERT_FLAG = False  # key: 'i'
        self.MOVE_FLAG = False  # key: 'm'
        self.ACTIVE_MOV_FLAG = False
        self.ind = None
        self.roi = None
        self.SHOW_LINES = None

    def return_worker(self):
        mask2 = np.transpose(np.nonzero(cv2.fillPoly(np.zeros(self.img2.shape[:2], np.uint8),
                                                    [self.roi], (255, 0, 0))))
        result = set(map(tuple, mask2[:, [1, 0]]))
        return result, self.ctrlPoints

