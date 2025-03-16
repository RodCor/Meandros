import pandas as pd
import numpy as np


def check_max_distance(df_contour):
    """
    Check the maximum distance between two points in the contour
    Args:
        df_contour: DataFrame with the contour points
    Returns:
        axis_bin: Axis with the maximum distance
    """
    axis_bin = "X"
    right = df_contour.nlargest(1, [axis_bin])
    left = df_contour.nsmallest(1, [axis_bin])
    distance_x = right[axis_bin].values.tolist()[0] - left[axis_bin].values.tolist()[0]
    axis_bin = "Y"
    right = df_contour.nlargest(1, [axis_bin])
    left = df_contour.nsmallest(1, [axis_bin])
    distance_y = right[axis_bin].values.tolist()[0] - left[axis_bin].values.tolist()[0]
    if distance_x > distance_y:
        return "X", 0
    else:
        return "Y", 1


def approx_line(ctr_points, class_id):
    """
    Approximate the axis of the contour
    Args:
        ctr_points: List with the contour points
        class_id: Class id of the contour
    Returns:
        final_line: DataFrame with the approximated axis
    """
    df_contour = pd.DataFrame(ctr_points, columns=["X", "Y"])
    axis_bin, axis_max = check_max_distance(df_contour)
    right = df_contour.nlargest(1, [axis_bin])
    left = df_contour.nsmallest(1, [axis_bin])
    cr = right.values.tolist()[0]
    cl = left.values.tolist()[0]
    max_cot = round(cr[axis_max], -2)
    aux_list = df_contour[["X", "Y"]].sort_values(by=axis_bin).values.tolist()
    a_side = half_lines(aux_list, cr)
    bux_list = [a_side[0]]
    bux_list = [i for i in aux_list if not i in a_side or a_side.remove(i)]
    b_side = half_lines(bux_list, cr)
    b_side = [i for i in b_side if i is not None]
    a_df = pd.DataFrame(data=a_side, columns=["X", "Y"])
    b_df = pd.DataFrame(data=b_side, columns=["X", "Y"])
    binarization(a_df, max_cot)
    binarization(b_df, max_cot)
    max_min(a_df)
    max_min(b_df)
    final_line = line_generator(a_df, b_df, cr, cl)
    return final_line


def binarization(contour, max_cot):
    """
    Binarize the contour points
    Args:
        contour: DataFrame with the contour points
        max_cot: Maximum value of the axis
    Returns:
        None
    """
    contour["bin"] = None
    i = 0
    last_bin = 0
    for limit_bin in range(0, max_cot, 25):
        if i == 0:
            contour["bin"] = np.where(
                (contour["Y"] >= min(contour["Y"])) & (contour["Y"] < limit_bin),
                i + 1,
                contour["bin"],
            )
        else:
            contour["bin"] = np.where(
                (contour["Y"] >= last_bin) & (contour["Y"] < limit_bin),
                i + 1,
                contour["bin"],
            )
        i = i + 1
        last_bin = limit_bin


def max_min(contour):
    """
    Mark the maximum and minimum points of the contour
    Args:
        contour: DataFrame with the contour points
    Returns:
        None
    """
    contour["max"] = 0
    contour["min"] = 0
    for b in contour["bin"].unique():
        if b:
            point_max = contour[contour["bin"] == b].nlargest(1, "X")
            point_min = contour[contour["bin"] == b].nsmallest(1, "X")
            contour["max"] = np.where(
                (contour["Y"] == point_max["Y"].values[0])
                & (contour["X"] == point_max["X"].values[0]),
                1,
                contour["max"],
            )
            contour["min"] = np.where(
                (contour["Y"] == point_min["Y"].values[0])
                & (contour["X"] == point_min["X"].values[0]),
                1,
                contour["min"],
            )


def line_generator(a_df, b_df, cr, cl):
    """
    Generate the approximated axis
    Args:
        a_df: DataFrame with the contour points in the first half
        b_df: DataFrame with the contour points in the second half
        cr: Right point of the contour
        cl: Left point of the contour
    Returns:
        line_aprox: DataFrame with the approximated axis
    """
    line_aprox = pd.DataFrame(columns=["X", "Y"])
    line_aprox = pd.concat(
        [line_aprox, pd.DataFrame([[cl[0], cl[1]]], columns=["X", "Y"])],
        ignore_index=True,
    )
    for b in a_df["bin"].unique():
        if b in b_df["bin"].unique() and b:
            round_y = round(
                (
                    a_df[(a_df["bin"] == b) & (a_df["min"] == 1)]["Y"].values[0]
                    + b_df[(b_df["bin"] == b) & (b_df["min"] == 1)]["Y"].values[0]
                )
                / 2
            )
            round_x = round(
                (
                    a_df[(a_df["bin"] == b) & (a_df["min"] == 1)]["X"].values[0]
                    + b_df[(b_df["bin"] == b) & (b_df["min"] == 1)]["X"].values[0]
                )
                / 2
            )
            line_aprox = pd.concat(
                [line_aprox, pd.DataFrame([[round_x, round_y]], columns=["X", "Y"])],
                ignore_index=True,
            )
    line_aprox = pd.concat(
        [line_aprox, pd.DataFrame([[cl[0], cl[1]]], columns=["X", "Y"])],
        ignore_index=True,
    )
    return line_aprox


def closest_p(point, list_points):
    min_p = 99999
    closest_point = None
    for lp in list_points:
        dist_min = np.sqrt(pow(lp[0] - point[0], 2) + pow(lp[1] - point[1], 2))
        if dist_min < min_p and dist_min > 0:
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
