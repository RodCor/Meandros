import cv2
import numpy as np
import _utils
import math
import matplotlib.pyplot as plt
import pandas as pd


def load_exclusion(path, exclude):
    img = cv2.fillPoly(cv2.imread(path), [np.asarray(n) for n in exclude], (255, 0, 0))
    return img


def analysis(p_d, conj_mask, ap, image, channel, threshold, landmarks):
    img = image
    img_r = image.copy()
    img1 = image.copy()

    pixels = []
    f_w = []
    f_w_corr = []
    areas = []
    areas_corr = []
    rango = 0
    region = []
    elbow_landmark = {tuple(landmarks[0])}
    wrist_landmark = {tuple(landmarks[1])}
    elbow = []
    wrist = []

    for k in range(len(p_d)):
        y = []
        a = []
        try:
            if k < 20:
                for i in range(0, k + 20):
                    y.append([p_d[i][1]])
                    a.append([1, p_d[i][0]])
            elif k >= 20 and abs(len(p_d) - k) >= 20:
                for i in range(k - 20, k + 20):
                    y.append([p_d[i][1]])
                    a.append([1, p_d[i][0]])
            else:
                for i in range(k - 20, len(p_d)):
                    y.append([p_d[i][1]])
                    a.append([1, p_d[i][0]])
        except IndexError as k:
            print(k)

        # REGRESSION LINEAL
        slope = _utils.min_cuadr(y, a)[1]
        angulo = math.atan(-1 / slope) * float(57.2958)

        b = _utils.min_cuadr(y, a)[0]
        pt_y = b + slope * (p_d[k][0])
        intersection = _utils.f(p_d[k][0], int(pt_y), -(1 / slope)) & conj_mask
        if intersection == set():
            break
        intersection_list = list(intersection)
        area = len(intersection_list)
        area_corr = _utils.f_corr(angulo, area)

        if intersection & ap == set():
            suma = 0
            for i in range(len(intersection_list)):
                img_r[intersection_list[i][1], intersection_list[i][0]] = [0, 255, 0]
                px = img[intersection_list[i][1], intersection_list[i][0]]
                if px[channel] > threshold:  # THRESHOLD
                    suma += 1
                    img1[intersection_list[i][1], intersection_list[i][0]] = [0, 0, 255]
            rango += 1
            areas.append(area)

            areas_corr.append(area_corr)
            av_corr = suma / (area_corr + 1)
            av = suma / (area + 1)
            f_w.append(av)
            f_w_corr.append(av_corr)
            pixels.append(suma)

            if intersection & elbow_landmark != set():
                print('eje_elbow', rango)
                elbow.append(rango)

            elif intersection & wrist_landmark != set():
                print('eje_wrist', rango)
                wrist.append(rango)
        elif intersection & ap != set():
            region.append(k)

    #print(region, rango, len(p_d))

    maximo_f = max(pixels)
    f_f_max_100 = list(map(lambda x: (x / maximo_f) * 100, pixels))

    cv2.imshow('prueba.png', img_r)
    cv2.imshow('lectura', img1)
    cv2.imshow('PISTA BORRADA', img)

    print('--------------------------\n')
    print('OUTPUT')

    plt.style.use('ggplot')

    p_d = np.linspace(- region[0] / (rango - region[0]) * 100, 100, rango)
    width = list(map(lambda x: (x / 3) * 5.16, areas_corr))   # ACA DIVIDO POR 3 PORQUE LA LINEA DE LECTURA TIENE 3
    # DE GROSOR (EN PROMEDIO, VER ESTO, OJO!!) 5.16 ES LA RELACION UM=PIXEL

    fig = plt.figure()
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(p_d, f_f_max_100, color='#E74C3C', label=r'$\mathcal{f}\ / \mathcal{f}_{max}$')
    ax1.axvline(p_d[elbow[0]], ls=':', color='gray')
    ax1.axvline(p_d[wrist[0]], ls=':', color='gray')
    ax1.set_ylabel(r'$\mathcal{f}\ / \mathcal{f}_{max}$ (%)', fontsize=20, labelpad=30)
    ax1.tick_params(axis='both', which='major', labelsize=14)
    ax1.legend(fontsize=14)
    plt.grid(True)

    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(p_d, width, color='#E74C3C', label=r'$\mathcal{width}$')
    ax2.axvline(p_d[elbow[0]], ls=':', color='gray')
    ax2.axvline(p_d[wrist[0]], ls=':', color='gray')
    ax2.set_xlabel('PD position (%)', fontsize=20, labelpad=30)
    ax2.set_ylabel(r'$\mathcal{width}$', fontsize=20, labelpad=30)
    ax2.tick_params(axis='both', which='major', labelsize=14)
    ax2.legend(fontsize=14)
    plt.grid(True)

    data = {'PD': p_d, 'f_f_max': f_f_max_100, 'width': width}
    print(pd.DataFrame(data), 'profile en get_profile')
    # plt.show()
    return pd.DataFrame(data)
