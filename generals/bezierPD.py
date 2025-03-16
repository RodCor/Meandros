import cv2
import numpy as np
import bezier
import generals.math_aux as math_aux
import math
import generals.windows_App as windows_App


class BezierPD(windows_App.App):
    """
    BezierPD class
    This class is used to create a Bezier curve and calculate the perpendicular distance of the curve to the ROI.
    Args:
        filename: string, image path
        roi: numpy array, region of interest
        ctrlPoints: list, control points
    Output:
        pd: numpy array, perpendicular distance
        intersection: list, intersection points
    """

    thickness = 5
    epsilon = 10

    def __init__(self, filename=None, roi=None, ctrlPoints=[]):

        self.img = cv2.imread(filename)
        self.filename = filename
        self.ctrlPoints = ctrlPoints
        self.INSERT_FLAG = False  # key: 'i'
        self.MOVE_FLAG = False  # key: 'm'
        self.ACTIVE_MOV_FLAG = False
        self.ind = None
        self.pd = None
        self.SHOW_LINES = False  # key: 'r'
        self.roi = roi
        self.intersection = []
        self.thickness = 8

    def line_control(self, k):
        nodes = np.asarray(self.ctrlPoints).T
        curve = bezier.Curve(nodes, (len(self.ctrlPoints) - 1))
        pd = curve.evaluate_multi(np.linspace(0.0, 1.0, 5000)).T
        pd = np.unique(pd.astype(int), axis=0)

        if self.ctrlPoints[0][0] < self.img2.shape[0]:
            self.pd = pd
        else:
            self.pd = pd[np.argsort(pd[:, 0])][::-1]
        try:
            for n in pd:
                self.img2[int(n[1]), int(n[0])] = [0, 250, 0]
        except IndexError:
            pass
        if k == ord("r"):

            self.mouse_flags(False, False, True)

            for k in range(0, len(pd), 20):
                y = []
                a = []
                try:
                    if k < 20:
                        for i in range(0, k + 20):
                            y.append([pd[i][1]])
                            a.append([1, pd[i][0]])
                    elif k >= 20 and abs(len(pd) - k) >= 20:
                        for i in range(k - 20, k + 20):
                            y.append([pd[i][1]])
                            a.append([1, pd[i][0]])
                    else:
                        for i in range(k - 20, len(pd)):
                            y.append([pd[i][1]])
                            a.append([1, pd[i][0]])
                except IndexError:
                    print("Index Error", k)
                slope = np.linalg.lstsq(y, a, rcond=None)[1][1]
                # slope = math_aux.min_cuadr(y, a)[1]
                # angle = math.atan(-1 / slope) * float(57.2958)
                # b = math_aux.min_cuadr(y, a)[0]
                b = np.linalg.lstsq(y, a, rcond=None)[0][0][0]
                pt_y = b + slope * (pd[k][0])
                intersection = math_aux.f(pd[k][0], int(pt_y), -(1 / slope)) & self.roi
                intersection_list = list(intersection)
                self.intersection.append(intersection_list)

                try:
                    for i in range(len(intersection_list) - 1):
                        pt1 = (intersection_list[i][0], intersection_list[i][1])
                        pt2 = (intersection_list[i + 1][0], intersection_list[i + 1][1])
                        # Dibuja una línea más gruesa
                        cv2.polylines(
                            self.img2,
                            [np.array([pt1, pt2])],
                            isClosed=False,
                            color=[18, 156, 243],
                            thickness=self.thickness,
                            lineType=cv2.LINE_AA,
                        )
                except IndexError:
                    print("Index Error")

    def return_worker(self):
        return (self.pd, self.intersection)
