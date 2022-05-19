import pandas as pd
import numpy as np


def approx_line(ctr_points, class_id):
    df_contour =  pd.DataFrame(ctr_points, columns =['X', 'Y'])
    if class_id == 0:
        axis_bin = 'Y'
        axis_max = 1
    else:
        axis_bin = 'X'
        axis_max = 0
    right = df_contour.nlargest(1, [axis_bin])
    left = df_contour.nsmallest(1, [axis_bin])
    cr = right.values.tolist()[0]
    cl = left.values.tolist()[0]
    max_cot = round(cr[axis_max],-2)
    aux_list = df_contour[["X","Y"]].sort_values(by=axis_bin).values.tolist()
    a_side = half_lines(aux_list, cr)
    bux_list = [a_side[0]]
    bux_list = [i for i in aux_list if not i in a_side or a_side.remove(i)]
    b_side = half_lines(bux_list, cr)
    b_side = [i for i in b_side if i is not None]
    a_df = pd.DataFrame(data=a_side, columns=["X", "Y"])
    b_df = pd.DataFrame(data=b_side, columns=["X", "Y"])
    binarization(a_df,max_cot)
    binarization(b_df,max_cot)
    max_min(a_df)
    max_min(b_df)
    line_aprox = line_generator(a_df, b_df,cr,cl)
    print(line_aprox)
    return line_aprox


def binarization (contour,max_cot):
    contour["bin"] = None
    i = 0
    last_bin = 0
    for limit_bin in range(0,max_cot,25):
        if i == 0:
            contour["bin"] = np.where((contour["Y"]>= min(contour["Y"])) & (contour["Y"]<limit_bin), i+1,contour["bin"])
        else:
            contour["bin"] =  np.where((contour["Y"]>=last_bin) & (contour["Y"]<limit_bin),i+1,contour["bin"])
        i = i+1
        last_bin = limit_bin


def max_min(contour):
    contour["max"] = 0
    contour["min"] = 0
    for b in contour["bin"].unique():
        if b:
            point_max = contour[contour["bin"]==b].nlargest(1,"X")
            point_min = contour[contour["bin"]==b].nsmallest(1,"X")
            contour["max"] = np.where((contour["Y"]==point_max["Y"].values[0]) & (contour["X"] == point_max["X"].values[0]),1,contour["max"])
            contour["min"] = np.where((contour["Y"]==point_min["Y"].values[0]) & (contour["X"] == point_min["X"].values[0]),1,contour["min"])

def line_generator(a_df, b_df,cr,cl):
    line_aprox = pd.DataFrame()
    line_aprox = line_aprox.append(pd.DataFrame([[cl[0],cl[1]]], columns=["X","Y"]), ignore_index=True)
    for b in a_df["bin"].unique():
        if b in b_df["bin"].unique() and b:
            round_y = round((a_df[(a_df["bin"] == b) & (a_df["min"]==1)]["Y"].values[0] + b_df[(b_df["bin"] == b) & (b_df["min"]==1)]["Y"].values[0])/2)
            round_x = round((a_df[(a_df["bin"] == b) & (a_df["min"]==1)]["X"].values[0] + b_df[(b_df["bin"] == b) & (b_df["min"]==1)]["X"].values[0])/2)
            line_aprox = line_aprox.append(pd.DataFrame([[round_x,round_y]], columns=["X","Y"]), ignore_index=True)
    line_aprox = line_aprox.append(pd.DataFrame([[cr[0],cr[1]]], columns=["X","Y"]), ignore_index=True)
    return line_aprox


def closest_p(point, list_points):
    min_p = 99999
    closest_point = None
    for lp in list_points:
        dist_min = np.sqrt(pow(lp[0] - point[0],2) + pow(lp[1] - point[1],2))
        if dist_min < min_p and dist_min>0:
            closest_point = lp
            min_p = dist_min
    return closest_point


def half_lines(list_point, cr):
    po = list_point[0]
    aux_temp = list_point
    aux_temp.remove(po)
    result_list = []
    px = po
    while px and px != cr:
        px = closest_p(po, aux_temp)
        if px and px != cr:
            result_list.append(px)
            po = px
            aux_temp.remove(px)
    return result_list