try:
    import numpy as np
    import pandas as pd
    import os
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    import rasterio
    import geopandas as gpd
    from sklearn.metrics import mean_squared_error
    from math import sqrt
    from scipy import stats
    import time
    
except ImportError as e:
    print(f'Problem with importing libraries, check availability, detalis: {e} \n\n Cannot continue the process...')


try:
    from preprocessingtools import *
    from atmospheric_correction import *

except ImportError as e:
    print(f'You need to download the preprocessingtools library from the repository, detalis: {e} \n\n Cannot continue the process...')





def band_extractor(raster,band_min, band_max):

    band_array_list = []

    for i in range(band_min,band_max+1):
        band = np.asarray(raster.read(i)).T
        band[band == raster.nodata] = np.nan     
        band_array_list.append(band)

    return band_array_list

def defo_bands_extractor(raster, number_of_sources):

    band_min = 1
    band_max = number_of_sources
    defo_bands = band_extractor(raster,band_min, band_max)

    return defo_bands


def incidence_bands_extractor(raster, number_of_sources):

    band_min = number_of_sources + 1
    band_max = 2 * number_of_sources
    incidence_bands = band_extractor(raster,band_min, band_max)

    return incidence_bands


def coherence_bands_extractor(raster, number_of_sources):

    band_min = 2 * number_of_sources + 1
    band_max = 3 * number_of_sources
    coherence_bands = band_extractor(raster,band_min, band_max)

    return coherence_bands


def A_matrix_constructor(x, y, incidence_bands, headings):
    A = []
    for i in range(len(incidence_bands)):
        A.append([np.cos(np.radians(incidence_bands[i][x,y])), -np.sin(np.radians(incidence_bands[i][x,y]))*np.cos(np.radians(headings[i]))])

    return np.array(A)


def L_matrix_constructor(x, y, defo_bands, number_of_sources):
    L = []
    for i in range(len(defo_bands)):
        L.append(defo_bands[i][x,y])

    return np.array(L).reshape(1,number_of_sources).T


def W_matrix_constructor(x, y, coherence_bands,number_of_sources):

    W = []
    for i in range(len(coherence_bands)):
        W.append(coherence_bands[i][x,y])
        W = np.diag(np.array(W))

    return W



def MNK_decomposition(raster, number_of_sources, headings):

    defo_bands = defo_bands_extractor(raster, number_of_sources)
    incidence_bands = incidence_bands_extractor(raster, number_of_sources)

    empty_matrix = np.empty((defo_bands[0].shape[0],defo_bands[0].shape[1],))
    empty_matrix[:] = np.nan

    up_raster = empty_matrix.copy()
    ew_raster = empty_matrix.copy()
    m0_raster = empty_matrix.copy()

    if (number_of_sources - 2) == 0:
        parr = 1
    else:
        parr = number_of_sources - 2

    for x in range(defo_bands[0].shape[0]):
        for y in range(defo_bands[0].shape[1]):
            A = A_matrix_constructor(x, y, incidence_bands, headings)
            L = L_matrix_constructor(x, y, defo_bands, number_of_sources)

            X = np.linalg.inv(A.T.dot(A)).dot(A.T).dot(L)
            v = A.dot(X) - L
            m0 = sqrt((v.T.dot(v))/(parr))

            up_raster[x,y] = X[0]
            ew_raster[x,y] = X[1]
            m0_raster[x,y] = m0



    return up_raster, ew_raster, m0_raster

def MNK_decomposition_coh(raster, number_of_sources, headings):

    defo_bands = defo_bands_extractor(raster, number_of_sources)
    incidence_bands = incidence_bands_extractor(raster, number_of_sources)
    coherence_bands = coherence_bands_extractor(raster, number_of_sources)


    empty_matrix = np.empty((defo_bands[0].shape[0],defo_bands[0].shape[1],))
    empty_matrix[:] = np.nan

    up_raster = empty_matrix.copy()
    ew_raster = empty_matrix.copy()
    m0_raster = empty_matrix.copy()

    if (number_of_sources - 2) == 0:
        parr = 1
    else:
        parr = number_of_sources - 2
  
    for x in range(defo_bands[0].shape[0]):
        for y in range(defo_bands[0].shape[1]):
            A = A_matrix_constructor(x, y, incidence_bands, headings)
            W = W_matrix_constructor(x, y, coherence_bands,number_of_sources)
            L = L_matrix_constructor(x, y, defo_bands, number_of_sources)

            X = np.linalg.inv(A.T.dot(W).dot(A)).dot(A.T).dot(W).dot(L)
            v = A.dot(X) - L

            
            m0 = sqrt((v.T.dot(W).dot(v))/(parr))

            up_raster[x,y] = X[0]
            ew_raster[x,y] = X[1]
            m0_raster[x,y] = m0


    return up_raster, ew_raster, m0_raster


def decomposition_mod(raster, number_of_sources, headings, coh_weight = bool):

    if coh_weight == True:
        print(MNK_decomposition_coh.__name__)
        up_raster, ew_raster, m0_raster = MNK_decomposition_coh(raster, number_of_sources, headings)
    else:
        print(MNK_decomposition.__name__)
        up_raster, ew_raster, m0_raster = MNK_decomposition(raster, number_of_sources, headings)

    return up_raster, ew_raster, m0_raster


def decomposition(path, out_path, number_of_sources, headings, coh_weight = bool):

    
    dinsar_data = [s for s in os.listdir(path) if s.endswith('.tif')]

    for file in dinsar_data:
        fullstart = time.time()
        filename = (path+ '\\' + file)
        raster_name = file
        raster1 = rasterio.open(filename)
        up_raster, ew_raster, m0_raster = decomposition_mod(raster1, number_of_sources, headings, coh_weight)

        start3 = time.time()

        with rasterio.open(out_path +'\\'+'up_' +raster_name, "w", driver=raster1.meta['driver'],height=raster1.meta['height'], width=raster1.meta['width'],
                            count=1, dtype=raster1.meta['dtype'], crs=raster1.meta['crs'],transform=raster1.meta['transform'],) as dest1:
            dest1.write(up_raster.T, 1)

        with rasterio.open(out_path +'\\'+'east_' + raster_name, "w", driver=raster1.meta['driver'],height=raster1.meta['height'], width=raster1.meta['width'],
                            count=1, dtype=raster1.meta['dtype'], crs=raster1.meta['crs'],transform=raster1.meta['transform'],) as dest2:
            dest2.write(ew_raster.T, 1)

        with rasterio.open(out_path +'\\'+'error_' + raster_name, "w", driver=raster1.meta['driver'],height=raster1.meta['height'], width=raster1.meta['width'],
                            count=1, dtype=raster1.meta['dtype'], crs=raster1.meta['crs'],transform=raster1.meta['transform'],) as dest3:
            dest3.write(m0_raster.T, 1)

        fullend = time.time()
        print(f'{raster_name} proces wykonano w {round((fullend-fullstart)/60,1)} m')



