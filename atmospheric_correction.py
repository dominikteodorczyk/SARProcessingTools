#   A trend is fitted to the deformation data in the form of a plane and then model values are output, 
#   which are then removed from the input data.
#
#   Version 2.0 from 04.05.2022 compiled on python = 3.7
#
#   The process requires input data and a write path. 
#   After fitting into the data planes with different degrees of complexity (from polynomial of degree 2 
#   to polynomial of degree 10 by default) using the MCDM TOPSIS method, the configuration with the highest 
#   kurtosis and the lowest errors on GNSS points and defined points in areas of probable land subsidence 
#   basins is selected.


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

except ImportError as e:
    print(f'You need to download the preprocessingtools library from the repository, detalis: {e} \n\n Cannot continue the process...')




class AtmoCorrectionModule(Raster):

    '''
    First, create an instance of AtmoCorrectionModule object and then use the atmospheric_correction() function to perform the correction
    -------------------------------------------------------
    Parametres:

    filename: str
        file name with path

    out_path: str
        file saving path

    fitting_point: str
        path to the shapefile containing the invariant point 

    pointsGNSS: str
        path to the shapefile containing the GNSS points to validate the shell fit

    pointsDEFO: str
        path to shapefile containing points In deformation basin to validate plane fit

    txt_path: str
        path to shapefile containing txt files with GNSS data 

    max_deg: int
        the maximum degree of the polynomial considered in the tests

    poli_w: float (0,1)
        the degree weight of the polynomial in the solution

    kurt_w: float (0,1)
        the degree weight of the kurtosis in the solution


    GNSS_w: float (0,1)
        the degree weight of the GNSS point RMSE in the solution


    DEFO_w: float (0,1)
        the degree weight of the DEFO point RMSE in the solution

    '''


    def __init__(self, filename, fitting_point, pointsGNSS, pointsDEFO, txt_path, max_deg, poli_w, kurt_w, GNSS_w, DEFO_w):
        super().__init__(filename)
        self.path = os.path.dirname(filename)
        self.fitting_point = fitting_point
        self.pointsGNSS = pointsGNSS
        self.pointsDEFO = pointsDEFO
        self.txt_path = txt_path
        self.max_deg = max_deg
        self.poli_w = poli_w
        self.kurt_w = kurt_w
        self.GNSS_w = GNSS_w
        self.DEFO_w = DEFO_w



    def GetInfo(self):

        #return basic data about the file
        
        return super().GetInfo()



    def ReadRasterFile(self):

        try:

            left, buttom, right, top = self.raster.bounds

            res_we, res_ns = self.raster.res
            data_array = np.asarray(self.raster.read(1)).T
            data_array[data_array == self.raster.nodata] = np.nan

            longitude  = np.linspace(left, left + (res_we * (self.raster.shape[0])), self.raster.shape[0])
            latitude = np.linspace(top, top + (res_ns * (self.raster.shape[1])), self.raster.shape[1])

            coordinates = np.meshgrid(longitude, latitude)
            L,B = np.asarray(coordinates[0]), np.asarray(coordinates[1])

            raster = None
            out_coord = np.array([B.flatten(), L.flatten(), data_array.flatten()], dtype = np.float64).T
            coordinates = out_coord[~np.isnan(out_coord).any(axis = 1), :]

        except ValueError as e:
            print(f'Something gone wrong!')
            print(f'mored details:\n{e}')
    
        return (coordinates, data_array, latitude, longitude)
    


    def list_of_points(self, longitude,latitude):

        #return a list of coordinates without values

        temp_X_val = []

        for y in latitude:
            for x in longitude:
                temp_X_val.append((x,y))
    
        temp_X_val = np.array(temp_X_val)

        return temp_X_val

    

    def fitting_value(self, diff_raster):

        #return value of raster shift after correction on invariant point 

        pointData= gpd.read_file(self.fitting_point)

        try:
            for point in pointData['geometry']:
                x = point.xy[0][0]
                y = point.xy[1][0]
                row, col = self.raster.index(x,y)
                shift_value = diff_raster[col,row]
                shift_value != np.nan

        except: 
            print(f'{ self.name } has NaN value in ref point!')
            print('Please change the point ')

        return shift_value



    def MNKcorrection(self, deg, coordinates, data_array, points_list):

        # performs least squares corrections using a polynomial of specified degree and returns an array of raster values

        shape = np.array(data_array).shape    
        result = np.where(~np.isnan(data_array))

        coordinates_test =np.column_stack((coordinates[:, 1], coordinates[:, 0]))
        polynominl = PolynomialFeatures(degree = deg)
        coordinates_test = polynominl.fit_transform(coordinates_test)

        model = LinearRegression()  
        model.fit(coordinates_test, data_array[result])

        coordinates_test = polynominl.transform(points_list)
        trend_values = model.predict(coordinates_test).reshape(shape[0],shape[1])

        diff_raster = data_array - trend_values
        fitting_raster = diff_raster - self.fitting_value(diff_raster)

        return fitting_raster



    def raster_value_reader(self):

        # returns the values of the base raster at given points

        pointData= gpd.read_file(self.pointsDEFO)
        raster_values = []

        for point in pointData['geometry']:
            x = point.xy[0][0]
            y = point.xy[1][0]
            row, col = self.raster.index(x,y)
            raster_values.append(self.raster.read(1)[row,col])

        return raster_values



    def raster_array_value_reader(self, array):

        # returns the value of points from the raster matrix after the correction
        
        pointData= gpd.read_file(self.pointsDEFO)
        raster_array_values = []

        for point in pointData['geometry']:
            x = point.xy[0][0]
            y = point.xy[1][0]
            row, col = self.raster.index(x,y)
            raster_array_values.append(array[col,row])

        return raster_array_values



    def kurtosis_value(self, raster_array):

        # returns the kurtosis value from the raster matrix
        
        return round(stats.kurtosis(raster_array[~np.isnan(raster_array)], axis=None)-3,5)



    def rmse_defo_points(self, corrected_raster_array):

        # calculates RMSE of DEFO points before and after correction 

        values_original = np.array(self.raster_value_reader())
        values_afetr_corection = np.array(self.raster_array_value_reader(corrected_raster_array))

        try:
            rmse = sqrt(mean_squared_error(values_original[~np.isnan(values_original)],
                        values_afetr_corection[~np.isnan(values_afetr_corection)]))

        except: 
            rmse = 0
                                        
        return round(rmse,3)



    def displacement_GNSS_reader(self):

        # reads station displacement values from txt files based on dates encoded in raster file name

        pointsData = gpd.read_file(self.pointsGNSS)
        GNSS_displacement_value = []

        for name in pointsData['Name']:
            x = name

            for file in os.listdir(os.fsencode(self.txt_path)):
                name = str(file)
                new_name = name.replace('b','').replace("'","").replace(".txt","")
                if new_name == x:
                    gnssdata = pd.read_csv(os.path.join(self.txt_path, str(os.fsdecode(file))))
                    try:
                        value1 = gnssdata.loc[gnssdata['Data'] == raster_period_dates(self.name)[0]]
                        value2 = gnssdata.loc[gnssdata['Data'] == raster_period_dates(self.name)[1]]
                        GNSS_displacement_value.append([new_name, 
                                    float(value2[f'LOS_{file_name_encoder(self.name)[2]}_ETRF']) 
                                    - float(value1[f'LOS_{file_name_encoder(self.name)[2]}_ETRF'])])
                    except: pass
            else: pass

        return GNSS_displacement_value



    def raster_array_value_reader_fGNNS(self, array):

        # reads the values of the G points from the raster

        pointData= gpd.read_file(self.pointsGNSS)
        raster_array_values = []

        for name,point in zip(pointData['Name'],pointData['geometry']):
            x = point.xy[0][0]
            y = point.xy[1][0]
            row, col = self.raster.index(x,y)
            raster_array_values.append([name ,array[col,row]])

        return raster_array_values



    def rmse_GNSS_points(self, corrected_raster_array):

        # calculates RMSE of GNSS points 

        x = pd.DataFrame(self.raster_array_value_reader_fGNNS(corrected_raster_array))
        y = pd.DataFrame(self.displacement_GNSS_reader())
        frames = [x,y[1]]
        result = pd.concat(frames, axis=1)
        result.columns = ['Point','ValueFromRaster','ValueFromGNSSData']
        result_clean = result.dropna()
        try:
            rmse = sqrt(mean_squared_error(result_clean['ValueFromRaster'],result_clean['ValueFromGNSSData']))
        except: 
            rmse = 0
                                        
        return [round(rmse,3), result]



    def stats_agregator(self, deg, corrected_raster_array):

        # calculates statistics for a given correction configuration and returns a list 

        kurtosis = self.kurtosis_value(corrected_raster_array)
        GNSSrsme, GNSSrsmeTable = self.rmse_GNSS_points(corrected_raster_array)
        DEFOrmse = self.rmse_defo_points(corrected_raster_array)
        stats_list = [deg, kurtosis,GNSSrsme,DEFOrmse]

        return stats_list



    def MNKpoli_fit_tester(self, points_list, coordinates, data_array):

        # module for testing all degrees of the polynomial, returns the matrix of the results of the individual solutions
        
        test_stats_results = []
        
        for deg in range(2,self.max_deg+1):
            diff_raster = self.MNKcorrection(deg, coordinates, data_array, points_list)
            stats_list = self.stats_agregator(deg, diff_raster)
            test_stats_results.append(stats_list)

        return test_stats_results



    def MCDM_module(self, test_stats_results):

        # performs TOPSIS selection of the best polynomial degree, returns the degree of the polynomial
        
        x = np.array(test_stats_results)

        sum_test = np.array([np.min(x[:,0]), 1 ,np.min(x[:,2]),np.min(x[:,3])])
        conver_test = np.array([[round(sum_test[i] / x[j, i], 7) for i in range(x.shape[1])] for j in range(x.shape[0])])
        conver_test[:,1] = x[:,1]

        weights = np.array([self.poli_w, self.kurt_w, self.GNSS_w, self.DEFO_w])

        col_sums = np.array(np.cumsum(x**2, 0)) 
        norm_x = np.array([[round(conver_test[i, j] / sqrt(col_sums[conver_test.shape[0]- 1, j]), 7) for j in range(x.shape[1])] for i in range(x.shape[0])])
        wnx = np.array([[round(weights[i] * norm_x[j, i], 7) for i in range(x.shape[1])] for j in range(x.shape[0])])

        pis = np.array([np.amax(wnx[:, :1]), np.amax(wnx[:, 1:2]), np.amax(wnx[:, 2:3]), np.amax(wnx[:, 3:4])])
        nis = np.array([np.amin(wnx[:, :1]), np.amin(wnx[:, 1:2]), np.amin(wnx[:, 2:3]), np.amin(wnx[:, 3:4])])

        b1 = np.array([[(wnx[j, i] - pis[i])**2 for i in range(x.shape[1])] for j in range(x.shape[0])])
        dpis = np.sqrt(np.sum(b1, 1))

        b2 = np.array([[(wnx[j, i] - nis[i])**2 for i in range(x.shape[1])] for j in range(x.shape[0])])
        dnis = np.sqrt(np.sum(b2, 1))

        final_solution = np.array([round(dnis[i] / (dpis[i] + dnis[i]), 5) for i in range(x.shape[0])])
        parameters = x[np.argmax(final_solution),:]
        
        return (np.argmax(final_solution) + 2), parameters



    def ATMO_MNK_Correction(self):

        #performs MCDM tests and final correction returning raster matrix corrected for atmospheric distortions 

        start = time.time()
        coordinates, data_array, latitude, longitude  = self.read_raster_file()
        points_list = self.list_of_points(longitude ,latitude)

        test_stats_results = self.MNKpoli_fit_tester(points_list,coordinates, data_array)
        poli_lvl_MCDM, self.parameters = self.MCDM_module(test_stats_results)
        corrected_raster_array = self.MNKcorrection(poli_lvl_MCDM, coordinates, data_array,points_list)

        print(f'{self.name} best solution: {poli_lvl_MCDM} in time {round((time.time()-start),1)} s')
        print(f'Parameters: {self.parameters}')

        return corrected_raster_array


    def atmospheric_correction_full(self, out_path):

        # method that performs the whole process including the step of writing to the file

        try:
            rester_corrected = np.array(self.ATMO_MNK_Correction()).astype("float32")
            with rasterio.open(out_path +'\\' + self.name, "w", **self.raster.meta) as new_raster:
                new_raster.write(rester_corrected.T, 1)

        except ValueError as e:
            print(f'Something gone wrong with {self.name} ')        
            print(f'more details: {e}')