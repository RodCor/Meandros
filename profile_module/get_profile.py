import cv2
import numpy as np
import generals.math_aux as ut
import math
import matplotlib.pyplot as plt
import pandas as pd


def load_exclusion(path, exclude=None):
    # output = cv2.imread(path)
    if exclude is not None:
        return cv2.fillPoly(cv2.imread(path), [np.asarray(n, dtype=np.int32) for n in exclude if len(n)], (255, 0, 0))
    else:
        return cv2.imread(path)


def analysis(p_d, conj_mask, ap, image, channel, threshold, landmarks=None):
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
    if landmarks is not None:
        elbow_landmark = {tuple(landmarks[1])}  # {tuple(landmarks[0])}  generalizar esto para multiples landmarks
        wrist_landmark = {tuple(landmarks[0])}
    else:
        elbow_landmark = None
        wrist_landmark = None

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
        except IndexError:
            print('no se puede', k)

        # REGRESSION LINEAL
        slope = ut.min_cuadr(y, a)[1]
        angulo = math.atan(-1 / slope) * float(57.2958)

        b = ut.min_cuadr(y, a)[0]
        pt_y = b + slope * (p_d[k][0])
        intersection = ut.f(p_d[k][0], int(pt_y), -(1 / slope)) & set(conj_mask)
        if intersection == set():
            break
        intersection_list = list(intersection)
        area = len(intersection_list)
        area_corr = ut.f_corr(angulo, area)

        #  if intersection & ap == set():
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

        if elbow_landmark is not None and intersection & elbow_landmark != set():
            print('eje_elbow', rango)
            elbow.append(rango)

        elif wrist_landmark is not None and intersection & wrist_landmark != set():
            print('eje_wrist', rango)
            wrist.append(rango)
        if intersection & ap != set():
            region.append(k)

    print(region, rango, len(p_d))

    maximo_f = max(pixels)
    f_f_max_100 = list(map(lambda x: (x / maximo_f) * 100, pixels))

    # cv2.imshow('prueba.png', img_r)
    # cv2.imshow('lectura', img1)
    # cv2.imshow('PISTA BORRADA', img)

    print('--------------------------\n')
    print('OUTPUT')

    plt.style.use('ggplot')

    p_d = np.linspace(- region[int(len(region)/2)] / (rango - region[int(len(region)/2)]) * 100, 100, rango)
    width = list(map(lambda x: (x / 3) * 5.16, areas_corr))   # ACA DIVIDO POR 3 PORQUE LA LINEA DE LECTURA TIENE 3
    # DE GROSOR (EN PROMEDIO, VER ESTO, OJO!!) 5.16 ES LA RELACION UM=PIXEL

    # df.to_csv('D:/backup_8-7-20/compu hernan/Desktop/analisis de imagenes/axolotl/trabajo actual/4-8/DA analysis,
    # set 1/TIG1-P53/' + files['name%d' % name], header=True, index=False)

    fig = plt.figure()
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(p_d, f_f_max_100, color='#E74C3C', label=r'$\mathcal{f}\ / \mathcal{f}_{max}$')
    ax1.axvline(p_d[region[int(len(region)/2)]], ls=':', color='gray')
    if elbow:
        ax1.axvline(p_d[elbow[0]], ls=':', color='gray')
    if wrist:
        ax1.axvline(p_d[wrist[0]], ls=':', color='gray')
    ax1.set_ylabel(r'$\mathcal{f}\ / \mathcal{f}_{max}$ (%)', fontsize=20, labelpad=30)
    ax1.tick_params(axis='both', which='major', labelsize=14)
    ax1.legend(fontsize=14)
    plt.grid(True)

    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(p_d, width, color='#E74C3C', label=r'$\mathcal{width}$')
    # ax2.axvline(p_d[elbow[0]], ls=':', color='gray')
    # ax2.axvline(p_d[wrist[0]], ls=':', color='gray')
    ax2.set_xlabel('PD position (%)', fontsize=20, labelpad=30)
    ax2.set_ylabel(r'$\mathcal{width}$', fontsize=20, labelpad=30)
    ax2.tick_params(axis='both', which='major', labelsize=14)
    ax2.legend(fontsize=14)
    plt.grid(True)

    data = {'PD': p_d, 'f_f_max': f_f_max_100, 'width': width}
    pd.DataFrame(data).to_csv("reports_intensity.csv", index=False)
    print(pd.DataFrame(data))
    plt.ion()
    plt.show()
    return pd.DataFrame(data), p_d[elbow[0]], p_d[wrist[0]] ## guardar en el self.File