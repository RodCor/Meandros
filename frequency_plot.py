import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import pandas as pd
import math
import bezier


def data_collection(pd_axis, intersection_list, pixels_th, value_th, color_th):
    index_point = 0
    j = 0
    list_points_pd = []
    list_leng_pd = []
    dist_list = []
    x_lab = []
    y_lab = []
    z_lab = []
    if color_th == 2:
        color_thr = 0
    else:
        color_thr = 1

    
    for inte in intersection_list:
        points = list(set(tuple(x) for x in inte) & set(tuple(x) for x in pd_axis))
        if points: 
            bpx = int(points[0][1])
            bpy = int(points[0][0])
            for py,px in inte:
                dist = math.sqrt((bpx - px)**2 + (bpy - py)**2)
                intense = pixels_th[bpy, bpx]
                dist_list.append((index_point, math.floor(dist), intense[color_thr]))
            index_point = index_point +1
            list_points_pd.append((bpy, bpx))
    
    while j<= len(list_points_pd)-1:
        if j+1<= len(list_points_pd)-1:
            leng = bezier.Curve([list_points_pd[j], list_points_pd[j+1]], 1).length
            list_leng_pd.append(leng)
        j = j+1

    for index_point, dist, inten in dist_list:
        if index_point < len(list_leng_pd):
            x_lab.append(list_leng_pd[index_point])
            y_lab.append(dist)
            z_lab.append(inten)
    
    np_x = np.array(x_lab)
    np_y = np.array(y_lab)
    np_z = np.array(z_lab)
    
    superplot = plt.figure(figsize =(14, 9))
    ax_superplot = plt.axes(projection ='3d')
    ax_superplot.scatter3D(np_x,np_y ,np_z , c = np_z<=value_th)
    ax_superplot.set_xlabel('Curve Lenght')
    ax_superplot.set_ylabel('Distance')
    ax_superplot.set_zlabel('Intensity');
    plt.show()