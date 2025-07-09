import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import numpy as np
from math import pi
from skimage.measure import EllipseModel
from shapely.geometry import Polygon


def pca_index_2(A,P):
    first = (pow(P,2)/A)
    second = (4*pi)
    return (first/second)


def obtain_area(b,a):
    return pi*a*b



def obtain_perim(b,a):
    return (2*pi)*np.sqrt((pow(a,2)+pow(b,2))/2)



def PCA_index (elips_df):
    features = ['X', 'Y']
    pca = PCA(n_components=2)
    x = elips_df.loc[:, features].values
    principalComponents = pca.fit_transform(x)
    principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])
    max_1 = principalDf['principal component 1'].max()
    min_1 = principalDf['principal component 1'].min()
    dist1 = max_1-min_1
    max_2 = principalDf['principal component 2'].max()
    min_2 = principalDf['principal component 2'].min()
    dist2 = max_2-min_2
    return abs(dist1/dist2)




def elipse_index (dataset):
    a_values = dataset.loc[:, ["X","Y"]].values
    ell = EllipseModel()
    ell.estimate(a_values)
    xc, yc, a, b, theta = ell.params
    A = obtain_area(b,a)
    P = obtain_perim(b,a)
    return pca_index_2(A,P)



def PI_Index(dataset):
    values = dataset.loc[:, ["X","Y"]].values
    A = Polygon(values).area
    P = Polygon(values).length
    return pca_index_2(A,P)




full = pd.read_csv("axolotl_gastru.csv")



result = pd.DataFrame()


for f in full.Filename.unique():
    df = full[full["Filename"]==f]
    pca_indx = PCA_index (df)
    elipse_indx = elipse_index(df)
    pi_indx = PI_Index(df)
    r = {'Filename': f, 'PI Index': pi_indx, 'Elipse Index': elipse_indx, 'PCA Index': pca_indx}
    result = result._append(r, ignore_index = True)
    

result.to_csv("final_gastruloids_axis.csv", index=False)

