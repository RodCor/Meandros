# aca va el scrip de analisis
import matplotlib.pyplot as plt
import math
import numpy as np


def f_corr(x, area):
    """esta función toma valores de ángulos(grados) y áreas como input y devuelve
        valores corregidos de las áreas. El factor de corrección lleva el valor
        del área dado, al correspondiente a una lectura a ángulo cero, que es
        donde los pixeles están completamente ordenados.

        f=area0/areatheta

        
        """
    if x > 0:
        f = (9.98874036e-01 - 1.54202234e-02 * x + 1.18599685e-03 * x ** 2 - 8.96081730e-05 * x ** 3 +
             4.04462433e-06 * x ** 4 - 9.94985379e-08 * x ** 5 + 1.33253976e-09 * x ** 6 - 9.15458430e-12 * x ** 7 +
             2.53133094e-14 * x ** 8)

        return f * area
    elif x < 0:
        x = -x
        f = (9.98874036e-01 - 1.54202234e-02 * x + 1.18599685e-03 * x ** 2 - 8.96081730e-05 * x ** 3 +
             4.04462433e-06 * x ** 4 - 9.94985379e-08 * x ** 5 + 1.33253976e-09 * x ** 6 - 9.15458430e-12 * x ** 7 +
             2.53133094e-14 * x ** 8)

        return f * area


def f(x0, y0, m):
    pts_posibles = set()
    if abs(m) >= 150 or math.isinf(m) == True:
        for y in range(y0 - 300,
                       y0 + 300):  # ESTE NUMERO ES UNO SUFICIENTEMENTE GRANDE COMO PARA TRAZAR UNA RECTA VERTICAL QUE SOBREPASE LA MASCARA
            #Cuadrados
            pts_posibles.add((x0, int(y)))
            pts_posibles.add((x0 + 1, int(y)))
            pts_posibles.add((x0 - 1, int(y)))
            pts_posibles.add((x0, int(y) + 1))
            pts_posibles.add((x0, int(y) - 1))
            pts_posibles.add((x0 + 1, int(y) + 1))
            pts_posibles.add((x0 - 1, int(y) - 1))
            pts_posibles.add((x0 - 1, int(y) + 1))
            pts_posibles.add((x0 + 1, int(y) - 1))
    elif 1 < abs(m) < 150:
        # Invierto para Y las lecturas al tener una pendiente muy empinada
        for y in range(y0 - 300, y0 + 300):
            x = x0 + (1 / m) * (y - y0)
            #Cuadrados
            pts_posibles.add((int(x), int(y)))
            pts_posibles.add((int(x) + 1, int(y)))
            pts_posibles.add((int(x) - 1, int(y)))
            pts_posibles.add((int(x), int(y) + 1))
            pts_posibles.add((int(x), int(y) - 1))
            pts_posibles.add((int(x) + 1, int(y) + 1))
            pts_posibles.add((int(x) - 1, int(y) - 1))
            pts_posibles.add((int(x) - 1, int(y) + 1))
            pts_posibles.add((int(x) + 1, int(y) - 1))
    else:
        for x in range(x0 - 300, x0 + 300):
            y = y0 + m * (x - x0)
            #Cuadrados
            pts_posibles.add((int(x), int(y)))
            pts_posibles.add((int(x) + 1, int(y)))
            pts_posibles.add((int(x) - 1, int(y)))
            pts_posibles.add((int(x), int(y) + 1))
            pts_posibles.add((int(x), int(y) - 1))
            pts_posibles.add((int(x) + 1, int(y) + 1))
            pts_posibles.add((int(x) - 1, int(y) - 1))
            pts_posibles.add((int(x) - 1, int(y) + 1))
            pts_posibles.add((int(x) + 1, int(y) - 1))
    return pts_posibles


# def f_(x0, y0, m):
#     pts_posibles = set()
#     if abs(m) >= 150 or math.isinf(m) == True:
#         for y in range(y0 - 200, y0 + 200):  # ESTE NUMERO ES UNO SUFICIENTEMENTE GRANDE COMO PARA TRAZAR UNA RECTA VERTICAL QUE SOBREPASE LA MASCARA
#             pts_posibles.add((x0, int(y)))
#             #pts_posibles.add((int(y), x0 + 1))
#             #pts_posibles.add((int(y), x0 - 1))
#             #pts_posibles.add((int(y) + 1, x0))
#             #pts_posibles.add((int(y) - 1, x0))
#             #pts_posibles.add((int(y) + 1, x0 + 1))
#             #pts_posibles.add((int(y) - 1, x0 - 1))
#             #pts_posibles.add((int(y) + 1, x0 - 1))
#             #pts_posibles.add((int(y) - 1, x0 + 1))
#     elif 1 < abs(m) < 150:
#         for y in range(y0 - 200, y0 + 200):
#             x = x0 + (1 / m) * (y - y0)
#             pts_posibles.add((int(x), int(y)))
#             #pts_posibles.add((int(y), int(x) + 1))
#             #pts_posibles.add((int(y), int(x) - 1))
#             #pts_posibles.add((int(y) + 1, int(x)))
#             #pts_posibles.add((int(y) - 1, int(x)))
#             #pts_posibles.add((int(y) + 1, int(x) + 1))
#             #pts_posibles.add((int(y) - 1, int(x) - 1))
#             #pts_posibles.add((int(y) + 1, int(x) - 1))
#             #pts_posibles.add((int(y) - 1, int(x) + 1))
#     else:
#         for x in range(x0 - 200, x0 + 200):
#             y = y0 + m * (x - x0)
#             pts_posibles.add((int(x), int(y)))
#             #pts_posibles.add((int(y), int(x) + 1))
#             #pts_posibles.add((int(y), int(x) - 1))
#             #pts_posibles.add((int(y) + 1, int(x)))
#             #pts_posibles.add((int(y) - 1, int(x)))
#             #pts_posibles.add((int(y) + 1, int(x) + 1))
#             #pts_posibles.add((int(y) - 1, int(x) - 1))
#             #pts_posibles.add((int(y) + 1, int(x) - 1))
#             #pts_posibles.add((int(y) - 1, int(x) + 1))
#     return pts_posibles


def min_cuadr(y, a):

    """Normal equation"""

    Y = np.asarray(y)
    A = np.asarray(a)
    At = np.transpose(A) # transpuesta de A
    AtxA = np.dot(At, A) #producto punto de AtxA
    AtxA_inv = np.linalg.inv(AtxA)# inversa de AtxA
    AtxA_invxAt = np.dot(AtxA_inv, At)#inversa de AtxA punto At
    AtxA_invxAtxY = np.dot(AtxA_invxAt, Y)#lo anterior punto Y
    b = AtxA_invxAtxY[0][0]
    slope = AtxA_invxAtxY[1][0]
    return [b, slope]


