import numpy as np
from mpl_toolkits import mplot3d
import matplotlib
import matplotlib.pyplot as plt
import math
import bezier



def data_collection(pd_axis, ap_axis, intersection_list, pixels_th, value_th, color_th):

    index_point = 0
    j = 0
    list_points_pd = []
    list_leng_pd = []
    dist_list = []
    x_lab = []
    y_lab = []
    z_lab = []
    max_leng = 0
    min_leng = 0
    max_dist = 0
    min_dist = 0
    if color_th == 2:
        color_thr = 0
        color_name = "RFP"
    else:
        color_thr = 1
        color_name = "GFP"

    
    ap_points = set(tuple(x) for x in ap_axis)
    zero_zero = list(set(tuple(x) for x in pd_axis) & ap_points)
    for inte in intersection_list:
        points = list(set(tuple(x) for x in inte) & set(tuple(x) for x in pd_axis))
        if points: 
            bpx = int(points[0][1])
            bpy = int(points[0][0])
            for py,px in inte:
                dist = math.sqrt((bpx - px)**2 + (bpy - py)**2)
                if bpy < py:
                    dist = dist * -1
                if max_dist < dist:
                    max_dist = dist
                if dist < min_dist :
                    min_dist = dist
                intense = pixels_th[bpy, bpx]
                dist_list.append((index_point, math.floor(dist), intense[color_thr]))
            index_point = index_point +1
            list_points_pd.append((bpy, bpx))
    
    while j<= len(list_points_pd)-1:
        if j+1<= len(list_points_pd)-1:
            leng = bezier.Curve([list_points_pd[j], list_points_pd[j+1]], 1).length
            # Arbitrary determination of the first point as zero for the consideration in terms of positive or negative
            if list_points_pd[j][0] < zero_zero[0][0]:
                leng = leng*-1
            if max_leng < leng:
                max_leng = leng
            if leng < min_leng :
                min_leng = leng
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

    pd_plane_x = [max_leng,max_leng,0,min_leng,min_leng]
    pd_plane_z = [0,255,255,0,255]
    xx, zz = np.meshgrid(pd_plane_x, pd_plane_z)
    yy = xx*0


    ap_plane_y = [min_dist,min_dist,max_dist,max_dist]
    ap_plane_z = [0,255,0,255]
    yyy, zzz = np.meshgrid(ap_plane_y, ap_plane_z)
    xxx = yyy*0
    
    superplot = plt.figure(figsize =(14, 9))
    ax_superplot = plt.axes(projection ='3d')
    ax_superplot.scatter3D(np_x,np_y ,np_z , c = np_z<=value_th, alpha = .3, linewidth = 0, antialiased=True, cmap= matplotlib.colors.ListedColormap([(.0, .2, 0), (.0, .79, .34)]), zorder=-1)
    ax_superplot.plot_surface(xx, yy, zz, color='red', alpha = .1, linewidth = 0, antialiased=True, zorder=1)
    ax_superplot.plot_surface(xxx, yyy, zzz, color='blue', alpha = .1, linewidth = 0, antialiased=True, zorder=2)
    ax_superplot.set_xlabel('PD Axis')
    ax_superplot.set_ylabel('AP Axis')
    ax_superplot.set_zlabel(f"""Intensity ({color_name})""");
    plt.show()