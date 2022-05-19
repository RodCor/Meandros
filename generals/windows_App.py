"""

=============================================================

Documentación:

1- Para agregar un punto de control presionar la tecla 'i' (insert)
   y clickear (left buttom) sobre la imagen donde desee colocarlo.
   (realizar esta tarea tantas veces como puntos de control desee agregar)

2- Para mover los puntos de control presionar la tecla 'm' (move)
   y manteniendo apretado el left buttom sobre el punto de control, desplazarse
   hasta la nueva ubicación.

3- Para salir presione ESC

=============================================================

"""


import cv2
import numpy as np


class App:

    RED = [0, 0, 255]
    WHITE = [255, 255, 255]
    BLUE = [255, 0, 0]
    ORANGE = [255, 165, 0]
    YELLOW = [255, 255, 0]
    GREEN = [0, 255, 0]
    COLORS = [RED, WHITE, BLUE, YELLOW, GREEN]


    thickness = 5
    epsilon = 10

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
        self.contour_points = None
        

    def onmouse(self, event, x, y, flags, param):

        if event == cv2.EVENT_MOUSEWHEEL:
            pass

        if event == cv2.EVENT_LBUTTONDOWN:
            if self.INSERT_FLAG:
                cv2.circle(self.img2, (x, y), self.thickness, self.RED)  # -1 for filled
                self.ctrlPoints.append([x, y])
            elif self.MOVE_FLAG:
                self.ACTIVE_MOV_FLAG = True
                self.ind = self.get_id_under_point(x, y)
                cv2.circle(self.img2, (x, y), self.thickness, self.WHITE)
                try:
                    self.ctrlPoints[self.ind] = [x, y]
                except TypeError:
                    pass
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.ACTIVE_MOV_FLAG:
                cv2.circle(self.img2, (x, y), self.thickness, self.WHITE)
                try:
                    self.ctrlPoints[self.ind] = [x, y]
                except TypeError:
                    pass
        elif event == cv2.EVENT_LBUTTONUP:
            self.INSERT_FLAG = False
            if self.MOVE_FLAG:
                try:
                    self.ctrlPoints[self.ind] = [x, y]
                    cv2.circle(self.img2, (x, y), self.thickness, self.RED)
                except TypeError:
                    pass

                self.ACTIVE_MOV_FLAG = False

    def get_id_under_point(self, x, y):
        xy = np.asarray(self.ctrlPoints)
        xt, yt = xy[:, 0], xy[:, 1]
        d = np.hypot(xt - x, yt - y)
        indseq, = np.nonzero(d == d.min())
        ind = indseq[0]
        if d[ind] >= self.epsilon:
            ind = None
        return ind

    def mouse_flags(self, MOVE_FLAG, INSERT_FLAG, SHOW_LINES):

        self.MOVE_FLAG = MOVE_FLAG
        self.INSERT_FLAG = INSERT_FLAG
        self.SHOW_LINES = SHOW_LINES

    def save_points(self):

        self.img = self.img2
        self.finalpoints.append(tuple(self.ctrlPoints))
        self.ctrlPoints.clear()

    def line_control(self,k):
        alpha = 0.4
        self.roi = np.array(self.ctrlPoints)
        cv2.fillPoly(self.img2, [self.roi], (255, 0, 0))

    def circle_worker(self, ctrl, k):

        for p in self.ctrlPoints:
            cv2.circle(self.img2, (p[0], p[1]), self.thickness, self.RED, -1)

        if len(self.ctrlPoints) > 1:
            self.line_control(k)

        if len(ctrl) > 1:
            for n in range(len(ctrl) - 1):
                cv2.line(self.img2, (ctrl[n][0], ctrl[n][1]), (ctrl[n + 1][0], ctrl[n + 1][1]), self.ORANGE, 1)
    

    
    def return_worker(self):

        result = np.asarray(self.finalpoints)
        return result

    def run(self, windows_title = None):
        
        if self.filename:
            self.img2 = self.img.copy()
            try:
                cv2.namedWindow(windows_title, cv2.WINDOW_GUI_EXPANDED)
                cv2.setMouseCallback(windows_title, self.onmouse)

            except cv2.error:
                return cv2.destroyAllWindows()

            while 1:
                cv2.imshow(windows_title, self.img2)
                k = cv2.waitKey(1)

                if k == ord('i'):

                    self.mouse_flags(False, True, False)

                elif k == ord('m'):

                    self.mouse_flags(True, False, False)

                elif k == ord('d'):

                    self.ctrlPoints.pop()

                elif k == ord('n'):

                    self.save_points()

                if not self.SHOW_LINES:
                    self.img2 = self.img.copy()

                ctrl = self.ctrlPoints
                self.circle_worker(ctrl, k)
                
                if k == 27:
                    
                    cv2.destroyAllWindows()
                    return self.return_worker()
        else:
            return