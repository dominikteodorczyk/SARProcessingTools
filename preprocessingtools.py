from osgeo import gdal
import numpy as np
from rasterstats import zonal_stats
import os
from scipy import stats
import csv
import rasterio


def files_cliper(dir_taker, dir_saver, cliped_area):

    '''Function used to crop tiff images for further processing. Takes arguments:
    - path to the folder with the data to trim
    - path to the folder where the data is to be saved
    - the .shp file used as the cropping pattern
    Attention! All files must be in the same EPSG layout'''

    try:

        for file in os.listdir(os.fsencode(dir_taker)):

            gdal.Warp(os.path.join(dir_saver, str(os.fsdecode(file))), os.path.join(dir_taker, str(os.fsdecode(file))), 
                cutlineDSName = cliped_area, cropToCutline = True, dstNodata = np.nan)
            
            print(f'File {str(os.fsdecode(file))} cliped with success!')
            
        print(f'! {(files_cliper.__name__).upper()} successfully completed the work !')


    except ValueError as e:

        print(f'! Process {files_cliper.__name__} was terminated with an error - probably different EPSG code for each file ! MORE DETAILS: {e}')        



def exportToFileCSV(path, header, data):

    
    try:
        with open(path, mode = 'x',newline='') as file:

            writer = csv.writer(file, delimiter = ';', quoting = csv.QUOTE_MINIMAL)
            writer.writerow(header)
            writer.writerow(data)

        print(f'Data has been added to new file: {path}')


    except:

        with open(path, mode = 'a',newline='') as file:

            writer = csv.writer(file, delimiter = ';', quoting = csv.QUOTE_MINIMAL)
            writer.writerow(data)

        print(f'Data has been added to existing file {path}')



def file_name_encoder(file):

    """Function used when writing data. Allows you to decode the file name XXAZZZ_DDMMYYYY_HHLLYYYY_P, where:
    XX - imaging period number
    A - type of orbit
    ZZZ - orbit number
    DDMMYYYY - day and month when the master image was taken
    HHLLYYYY - day and month when the slave image was taken
    P - raster data type (displacement/coherence)"""

    r = []
    full_file_name = str(os.fsdecode(file))

    r.append(full_file_name[0:2])
    r.append(full_file_name[2])
    r.append(full_file_name[3:6])
    r.append(f'{full_file_name[7:9]}.{full_file_name[9:11]}.{full_file_name[11:15]}')
    r.append(f'{full_file_name[16:18]}.{full_file_name[18:20]}.{full_file_name[20:24]}')
    r.append(full_file_name[25:])

    return r


def stats_coh_doubel_std(x):
    
    return (x > np.mean(x) - (2 * np.std(x))).sum()



def above_value_02(x):

    return (x > 0.2).sum()



def coherence_statistics(direction, zone_of_interst, save_path):

    """A function that performs statistical calculations of coherence raster images. 
    Useful in the process of selecting optimal time series for further work. 
    
    Arguments of the function:
    - direction - folder path,
    - zone_of_interst - area, which we want to calculate (in .shp format - one object)
    - path of data saving in .csv format"""

    for file in os.listdir(os.fsencode(direction)):

        try:

            stat = zonal_stats(zone_of_interst, os.path.join(direction, str(os.fsdecode(file))), 
                stats = "sum mean count std median",
                    add_stats = {'stats_coh_doubel_std': stats_coh_doubel_std, 
                                 'above_value': above_value_02})

            header = ['raster_no', 'orbit_type', 'orbit', 'day_start', 'days_stop', 'file_type']
            data = file_name_encoder(file)

            for key in stat[0]:

                header.append(key)
                data.append(stat[0][key])

            exportToFileCSV(save_path, header, data)

        except ValueError as e:

            print(f'\n There is something wrong with file {str(os.fsdecode(file)).upper()}, more details:\n\n{e}')
            pass


    else: 
        
        print('{} HAS BEEN EXECUTED !'.format(coherence_statistics.__name__).upper())



def diplacement_statistcs(direction,save_path):

    """A function to calculate statistics for rasters representing displacements. 
    The function calculates the mean, median, min, max, standard deviation, kurtosis, and skewness. 
    
    Arguments of the function:
    direction- path to the folder containing the raster data set
    save_path - the path to save the resulting .csv file with statistics."""

    for file in os.listdir(os.fsencode(direction)):

        try:

            defo_map = gdal.Open(os.path.join(direction, str(os.fsdecode(file))))
            defor_map_array = defo_map.GetRasterBand(1).ReadAsArray()[~np.isnan(defo_map.GetRasterBand(1).ReadAsArray())]

            stat = {'mean' : np.nanmean(defor_map_array), 
                    'median' : np.nanmedian(defor_map_array), 
                    'min' : np.nanmin(defor_map_array),
                    'max' : np.nanmax(defor_map_array),
                    'std' : np.nanstd(defor_map_array), 
                    'kurtosis': stats.kurtosis(defor_map_array)-3,
                    'skew' : stats.skew(defor_map_array)}
            print(stat)
            header = ['raster_no', 'orbit_type', 'orbit', 'day_start', 'days_stop', 'file_type']
            data = file_name_encoder(file)

            for key in stat:

                header.append(key)
                data.append(stat[key])

            exportToFileCSV(save_path, header, data)


        except ValueError as e:

            print(f'\n There is something wrong with file {str(os.fsdecode(file)).upper()}, more details:\n\n{e}')
            pass


    else: 
        
        print(f'{(diplacement_statistcs.__name__).upper()} HAS BEEN EXECUTED !')





def shp2tif_interpolation(direction, save_direction, feature, type_of_interpolation):


    for file in os.listdir(os.fsencode(direction)):
        name = str(file)

        if name.endswith((".shp'")): 

            try:
                new_name = name[1:].replace('.shp','_iterpolation.tif').replace("'","")
                gdal.Grid(os.path.join(save_direction, new_name), 
                            os.path.join(direction, str(os.fsdecode(file))),
                                zfield=feature, algorithm = type_of_interpolation,
                                 width=3000, height=3000)

            except: ValueError

    
    else: print(f'Process {shp2tif_interpolation.__name__} complited!')




def raster_resample(direction, save_direction, resolution):


    for file in os.listdir(os.fsencode(direction)):
        name = str(file)

        if name.endswith((".tif'")):

            try:
                new_name = name[1:].replace('.tif',f'_resample{resolution}.tif').replace("'","")
                gdal.Warp(os.path.join(save_direction, new_name), 
                            os.path.join(direction, str(os.fsdecode(file))), 
                                xRes = resolution, yRes = resolution)

            except: ValueError

    
    else: print(f'Process {raster_resample.__name__} complited!')




def raster_period_dates(filename):


    x = file_name_encoder(filename)
    
    return x[3:5]



class Raster():

    def __init__(self, filename):

        self.raster = rasterio.open(filename)
        self.filename = filename
        self.name = os.path.basename(filename)
        self.type = file_name_encoder(self.name)[5].replace('.tif','')


    def GetInfo(self):

        metadate = self.raster.meta
        filenamemeta = file_name_encoder(self.name)

        print(f'{self.name}')
        print(f'Start of observation: {filenamemeta[3]}\nEnd of observation: {filenamemeta[4]}')
        if filenamemeta[1] == 'D':
            print(f'Orbit type: Descending \nOrbit number: {filenamemeta[2]}')
        else: print(f'Orbit type: Ascending \nOrbit number: {filenamemeta[2]}')
        print(f'{self.raster.crs}')
        print(f'file type: {metadate["driver"]}\t number type: {metadate["dtype"]}')
        print(f'Raster width: {metadate["width"]}px\t Raster height: {metadate["height"]}px')




