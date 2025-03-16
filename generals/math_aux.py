# aca va el scrip de analisis
import matplotlib.pyplot as plt
import math
import numpy as np


def f_corr(x, area):
    """
    This function takes angle values (in degrees) and areas as input and returns corrected area values.
    The correction factor adjusts the given area value to correspond to a reading at zero degrees, where the pixels are fully aligned.
    The correction factor is calculated as:
    f = frac{{area}_0}{{area}_theta}

    Where:
    - {area}_0: Area value at zero degrees.
    - {area}_theta: Area value at the given angle.

    Args:
        x: float
            Angle value in degrees.
        area: float
            Area value.
    Output:
        f * area: float
            Corrected area value
    """
    if x > 0:
        f = (
            9.98874036e-01
            - 1.54202234e-02 * x
            + 1.18599685e-03 * x**2
            - 8.96081730e-05 * x**3
            + 4.04462433e-06 * x**4
            - 9.94985379e-08 * x**5
            + 1.33253976e-09 * x**6
            - 9.15458430e-12 * x**7
            + 2.53133094e-14 * x**8
        )

        return f * area
    elif x < 0:
        x = -x
        f = (
            9.98874036e-01
            - 1.54202234e-02 * x
            + 1.18599685e-03 * x**2
            - 8.96081730e-05 * x**3
            + 4.04462433e-06 * x**4
            - 9.94985379e-08 * x**5
            + 1.33253976e-09 * x**6
            - 9.15458430e-12 * x**7
            + 2.53133094e-14 * x**8
        )

        return f * area


def f(x0, y0, m):
    pts_posibles = set()
    if abs(m) >= 150 or math.isinf(m) == True:
        for y in range(y0 - 300, y0 + 300):

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

        for y in range(y0 - 300, y0 + 300):
            x = x0 + (1 / m) * (y - y0)

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


def min_cuadr(y, a):
    """Normal equation"""

    Y = np.asarray(y)
    A = np.asarray(a)
    At = np.transpose(A)
    AtxA = np.dot(At, A)
    AtxA_inv = np.linalg.inv(AtxA)
    AtxA_invxAt = np.dot(AtxA_inv, At)
    AtxA_invxAtxY = np.dot(AtxA_invxAt, Y)
    b = AtxA_invxAtxY[0][0]
    slope = AtxA_invxAtxY[1][0]
    return [b, slope]
