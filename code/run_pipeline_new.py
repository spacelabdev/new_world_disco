import logging
import io
import os
import csv
import datetime
import stopit
import boto3
from glob import glob
from pathlib import Path

import lightkurve as lk
import numpy as np
import pandas as pd
import requests
import math
from astropy import units as u
import warnings
from astropy.table import QTable



logger = logging.getLogger(__name__)

logger.setLevel(logging.INFO)
handler = logging.FileHandler('preprocess.log')
logger.addHandler(handler)

TESS_DATA_URL = 'https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=csv'
LOCAL_DATA_FILE_NAME = '/home/ubuntu/spaceLab/tess_data.csv'
DEFAULT_TESS_ID = '2016376984' # a working 'v-shaped' lightcurve. Eventually we'll need to run this for all lightcurves from tess
BJD_TO_BCTJD_DIFF = 2457000
OUTPUT_FOLDER = 'tess_data/' # modified to save to different output folder
subFolders = ['tic_info/', 'locGlo_flux/', 'locGlo_cent/'] #sub folders within the main output folder
# A list of all the valid authors that are supported by
# lightkurves searches and downloads
valid_authors = ["Kepler","K2", "SPOC","TESS-SPOC","QLP","TASOC","PATHOS","CDIPS","K2SFF","EVEREST","TESScut","GSFC-ELEANOR-LITE"]

# these bin numbers for TESS from Yu et al. (2019) section 2.3: https://iopscience.iop.org/article/10.3847/1538-3881/ab21d6/pdf
global_bin_width_factor = 201
local_bin_width_factor = 61

def fetch_tess_data_df():
    """
    Method to load TESS data. 

    If data does not exist locally, it will be downloaded from
    TESS_DATA_URL and saved locally.  As more TESS data comes in,
    the existing file will expand to include the newer data.
    """

    #if os.path.isfile(LOCAL_DATA_FILE_NAME):
        #return pd.read_csv(LOCAL_DATA_FILE_NAME)
    res = requests.get(TESS_DATA_URL)
    tess_data_raw = res.content
    with open(LOCAL_DATA_FILE_NAME, 'wb+') as f:
        f.write(tess_data_raw)
    return pd.read_csv(io.BytesIO(tess_data_raw))

def preprocess_tess_data(tess_id, data_df):
    """
    Method for preprocessing TESS data.

    Data preprocessing consists of 4 stages:
        1. Outliers are removed.
        2. The lightcurve is flattened and folded.
        3. A global view of the lightcurve is generated.
        4. A local view of the transit is generated.

    After preprocessing, global and local views are saved to disk.

    Input: tess_id = TESS Input Catalog (TIC) identifier.
    """

    # Download and stitch all lightcurve quarters together.

    id_string = f'TIC {tess_id}'
    
    print("Loading lightcurves")
    
    q = lk.search_lightcurve(id_string)
    
    
    # Stores all of the authors from different observations
    # For a given 
    authors_column = q.table['author']

    # Creates a boolean mask for rows where all authors are valid
    valid_authors_mask = [all(author in valid_authors for author in authors.split(','))
                    for authors in authors_column]

    # Select only the rows where all authors are valid
    # Allows us to remove rows that have unsupported authors
    search_result = q[valid_authors_mask]
    
    # Downloading the Lightkurve data
    lcs = search_result.download_all()
    #print(lcs[0])
    # TO DO: need to figure out how to work around
    # .stitch() so it does not remove the SAP columns in the data table
    # column types are incompatible: {'sap_bkg', 'sap_bkg_err', 'sap_flux'}
    #Feeds into preprocess_centroid section/function

    ###### BEGIN JULIAN'S CODE ######
    #################################
    list_of_lcs_dfs = [] #Initialize list to hold df's
    print("Converting lightcurves to dataframes")
    for i in range(len(lcs)):
        lcs_df = lcs[i].to_pandas() #Convert lightcurve at index i to pandas df
        list_of_lcs_dfs.append(lcs_df) #Append df to list
    
    new_df = ''
    if len(lcs) == 1:
        new_df = list_of_lcs_dfs[0] #If the length is 1, then use only that df
    elif len(lcs) == 2:
        new_df = pd.concat([list_of_lcs_dfs[0], list_of_lcs_dfs[1]], axis=0, join="outer") #If length is 2, concat df's together
    elif len(lcs) > 2:
        new_df = pd.concat([list_of_lcs_dfs[0], list_of_lcs_dfs[1]], axis=0, join="outer") #Initialize
        for j in range(2, len(lcs)):
            new_df = pd.concat([new_df, list_of_lcs_dfs[j]], axis=0, join="outer") #Concat each subsequent df
        
    new_df.sort_index(inplace=True) #Sort the time index
    new_df = new_df[['sap_bkg_err', 'sap_bkg', 'sap_flux', 'sap_x', 'sap_y', 'centroid_row', 'centroid_col']]

    #Note for potential optimizing:
    #Filter new_df columns to include only the following:
    #[time, flux, flux_err, cadenceno, quality, sap_bkg_err, sap_bkg, sap_flux, sap_x, sap_y, centroid_row, centroid_col]

    q_table = QTable.from_pandas(new_df, index=True) #Convert the dataframe into a QTable
    lc_temp = lk.LightCurve(data=q_table) #Convert the QTable into a LightCurve object

    ###### END JULIAN'S CODE ######
    #################################

    lc_raw = lcs.stitch()
    #stitch eliminates sap and centroid columns
    #print(lc_raw)
    #Try converting lc_raw and lc_temp to dfs, concatenate, convert back to lc objects, and normalize
    lc_raw_df = lc_raw.to_pandas()
    lc_temp_df = lc_temp.to_pandas()
    lc_temp_df.drop(columns=['flux', 'flux_err'], inplace=True)
    joined_df = lc_raw_df.join(lc_temp_df, how='outer')
    joined_df.sort_index(inplace=True)
    q_table_2 = QTable.from_pandas(joined_df, index=True)
    lc_concat = lk.LightCurve(data=q_table_2)
    
    # Fetch period and duration data from caltech exofop for tess

    # Commented this line of code out from the orginal script
    # data_df = fetch_tess_data_df()

    print("Extracting stellar parameters")

    threshold_crossing_events = data_df[data_df['TIC ID'] == int(tess_id)]

    tce_count = threshold_crossing_events.shape[0]

    for i in range(tce_count):
        
        period, duration = threshold_crossing_events['Period (days)'].iloc[i].item(),  threshold_crossing_events['Duration (hours)'].iloc[i].item()
        t0 = threshold_crossing_events['Epoch (BJD)'].iloc[i].item() - BJD_TO_BCTJD_DIFF

        # info contains: [0]tic, [1]tce, [2]period, [3]epoch, [4]duration, [5]label,
        # [6]Teff, [7]logg, [8]metallicity, [9]mass, [10]radius, [11]density
        info = np.full((12,), np.nan)

        info[0] = tess_id
        info[1] = i + 1
        info[2] = period
        info[3] = threshold_crossing_events['Epoch (BJD)'].item()
        info[4] = duration

        # if label is -1, these are unknowns for the experimental set
        if threshold_crossing_events['TFOPWG Disposition'].item() in ['KP', 'CP']:
            info[5] = 1
        elif threshold_crossing_events['TFOPWG Disposition'].item() in ['FA', 'FP']:
            info[5] = 0
        else:
            info[5] = -1

        info[6] = threshold_crossing_events['Stellar Eff Temp (K)'].item()
        info[7] = threshold_crossing_events['Stellar log(g) (cm/s^2)'].item()
        info[8] = threshold_crossing_events['Stellar Metallicity'].item()
        info[9] = threshold_crossing_events['Stellar Mass (M_Sun)'].item()
        info[10] = threshold_crossing_events['Stellar Radius (R_Sun)'].item()
        
        stellar_params_link = f'https://exofop.ipac.caltech.edu/tess/download_planet.php?id={tess_id}&output=csv'

        densities = pd.read_csv(stellar_params_link, sep='|')['Fitted Stellar Density (g/cm3)']

        if not np.all(densities.isna()):
            info[11] = pd.read_csv(stellar_params_link, sep='|')['Fitted Stellar Density (g/cm3)'].dropna().iloc[0].item()\


        print("Processing outliers")

        lc_clean = lc_concat.remove_outliers(sigma=3)

        

        # Do the hacky masking from here: https://docs.lightkurve.org/tutorials/3-science-examples/exoplanets-machine-learning-preprocessing.html
        temp_fold = lc_clean.fold(period, epoch_time=t0)
        fractional_duration = (duration / 24.0) / period
        phase_mask = np.abs(temp_fold.phase.value) < (fractional_duration * 1.5)
        transit_mask = np.in1d(lc_clean.time.value, temp_fold.time_original.value[phase_mask])


        print("Flattening lightcurve")

        lc_flat = lc_clean.flatten(mask=transit_mask)
        lc_fold = lc_flat.fold(period, epoch_time=t0)

        print("Creating global representation")
    
        lc_global = lc_fold.bin(time_bin_size=period/global_bin_width_factor).normalize() - 1
        if not (len(lc_global) == global_bin_width_factor):
            logger.info(f'{tess_id} lc_global incorrect dimension: {len(lc_global)}')
            return
        lc_global = (lc_global / np.abs(np.nanmin(lc_global.flux)) ) * 2.0 + 1

        print("Creating local representation")

        phase_mask = (lc_fold.phase > -4*fractional_duration) & (lc_fold.phase < 4.0*fractional_duration)
        lc_zoom = lc_fold[phase_mask]

        # we use 8x fractional duration here since we zoomed in on 4x the fractional duration on both sides
        lc_local = lc_zoom.bin(time_bin_size=8*fractional_duration/local_bin_width_factor).normalize() - 1
        if not (len(lc_local) == local_bin_width_factor):
            logger.info(f'{tess_id} lc_local incorrect dimension: {len(lc_local)}')
            return
        lc_local = (lc_local / np.abs(np.nanmin(lc_local.flux)) ) * 2.0 + 1


        # print(lc_local.dtype.names)
        
        # add centroid preprocessing
        print("Preprocessing centroid")
        local_cen, global_cen = preprocess_centroid(lc_local, lc_global)
        
        # export
        print("Exporting lightcurves")
        export_lightcurve(lc_local, f"{tess_id}_0{i+1}_local")
        export_lightcurve(lc_global, f"{tess_id}_0{i+1}_global")

        np.save(f"/home/ubuntu/spaceLab/{OUTPUT_FOLDER+subFolders[0]+str(tess_id)}_0{i+1}_info.npy", np.array(info))

        # export
        np.save(f"/home/ubuntu/spaceLab/{OUTPUT_FOLDER+subFolders[2]+str(tess_id)}_0{i+1}_local_cen.npy", local_cen)
        np.save(f"/home/ubuntu/spaceLab/{OUTPUT_FOLDER+subFolders[2]+str(tess_id)}_0{i+1}_global_cen.npy", global_cen)

def export_lightcurve(lc, filename):
    """
    Method to save lightcurve data as CSV and a NumPy array file (.npy) representing flux.

    Inputs: lc = lightcurve to be saved.
            folder = folder in which to save file.
            filename = name of the file.
    """

    if not os.path.isdir(f"/home/ubuntu/spaceLab/{OUTPUT_FOLDER}"):
        os.mkdir(os.path.join(os.getcwd(), f"/home/ubuntu/spaceLab/{OUTPUT_FOLDER}"))

    # Creating the subfolder, if needed
    for subfolder in subFolders:
        if not os.path.isdir(f"/home/ubuntu/spaceLab/{OUTPUT_FOLDER+subfolder}"):
            os.makedirs(os.path.join(f"/home/ubuntu/spaceLab/{OUTPUT_FOLDER}", subfolder), exist_ok=True)

#   lc.to_csv(f"./data/{filename}.csv", overwrite=True)
    np.save(f"/home/ubuntu/spaceLab/{OUTPUT_FOLDER+subFolders[1]+str(filename)}_flux.npy", np.array(lc['flux']))

### Centroid Preprocessing
# section 2.2 of https://arxiv.org/pdf/1810.13434.pdf describes how they normalized
def normalize_centroid(centroid_data):
    # normalize by subtracting median and dividing by standard deviation
    med = np.median(centroid_data)
    std = np.std(centroid_data)
    centroid_data -= med
    if std == 0:
        logger.info("Error; normalize_centroid(): std == 0")
        return
    centroid_data /= std

    # TO DO
    # "Moreover, we normalize the standard deviation of the centroid curves by that of the light curves"
    # from 2.2 not implemented


def get_mag(x, y):
    # get magnitude as: sqrt(x^2 + y^2)
    return math.sqrt(x*x + y*y)

def preprocess_centroid(lc_local, lc_global):
    """
    Method for preprocessing TESS centroid data

    Input: local and global lightcurve objects (already pre-processed)
    Output: local and global centroid position numpy arrays
    """
    sap_global_condition = 'sap_x' in lc_global.columns and 'sap_y' in lc_global.columns
    sap_local_condition = 'sap_x' in lc_local.columns and 'sap_y' in lc_local.columns
    if sap_global_condition and sap_local_condition:
        # remove the pix dimension
        global_x = np.array([float(x*u.pix/u.pix) for x in lc_global['sap_x']])
        global_y = np.array([float(y*u.pix/u.pix) for y in lc_global['sap_y']])
        local_x = np.array([float(x*u.pix/u.pix) for x in lc_local['sap_x']])
        local_y = np.array([float(y*u.pix/u.pix) for y in lc_local['sap_y']])
    else:
        # TO DO checking for centroid_row, centroid_col and performing preprocessing the data 
        '''
        centroid_global_cond = 'centroid_row' in lc_global.columns and 'centroid_col' in lc_global.columns
        centroid_local_cond = 'centroid_row' in lc_global.columns and 'centroid_col' in lc_global.columns
        if centroid_global_cond and centroid_local_cond:
            # remove the pix dimension...keeeping same name convention
            cent_row_global = np.array([float(row/u.pix) for row in lc_global['centroid_row']])
            cent_col_global = np.array([float(col/u.pix) for col in lc_global['centroid_col']])
            cent_row_local = np.array([float(row/u.pix) for row in lc_local['centroid_row']])
            cent_col_local = np.array([float(col/u.pix) for col in lc_local['centroid_col']])
        else:
        '''
        logger.info("Error: preprocess_centroid(): No handling for centroid data not stored in sap_x, sap_y or centroid_row, centroid_col")
        return

    # compute r = sqrt(x^2 + y^2) for each centroid location
    local_cen = np.array([get_mag(x,y) for x, y in zip(local_x, local_y)])
    global_cen = np.array([get_mag(x,y) for x, y in zip(global_x, global_y)])

    # normalize by subtracting mean and dividing by standard deviation
    normalize_centroid(local_cen)
    normalize_centroid(global_cen)

    return local_cen, global_cen


if __name__ == "__main__":
    tess_data = fetch_tess_data_df()

    #Initialize a dictionary to hold the TESS id's and their outcomes
    preprocess_results = {}

    for index, row in tess_data.iterrows():
        tic_id = str(row['TIC ID'])
        print("Working on TIC_ID " + tic_id)
        print("#"*30)

        with stopit.ThreadingTimeout(120) as context_manager: #Set max number of seconds to work on one lightcurve
            try:
                preprocess_tess_data(tic_id, tess_data)
            except:
                preprocess_results[tic_id] = 'ERROR' #Indicates that an error was thrown somewhere
                continue

        if context_manager.state == context_manager.EXECUTED: #Indicates preprocessing finished in under 120 seconds (2 min)
            preprocess_results[tic_id] = "COMPLETE"
        elif context_manager.state == context_manager.TIMED_OUT: #Indicates timed out or possible error 
            preprocess_results[tic_id] = "UNFINISHED"   

    print("Preprocessing finished")


    #S3 bucket to store the files
    bucket_name = "preprocessed-data-2023q1"

    print("Beginning upload to S3 bucket")
    #Path to preprocess data directory: "/home/ubuntu/spaceLab/tess_data"

    s3 = boto3.client('s3')

    #Folder and subfolders from preprocess.py file
    OUTPUT_FOLDER = 'tess_data/' # modified to save to different output folder
    subFolders = ['tic_info/', 'locGlo_flux/', 'locGlo_cent/'] #sub folders within the main output folder

    #Loop over each file in each subfolder and add to the respective folder in the bucket
    #Subfolders will be created if not already in place
    for s in subFolders:
        for file in glob(f"/home/ubuntu/spaceLab/{OUTPUT_FOLDER+s}*"):
            p = Path(file).stem+".npy" #Takes the stem of the file name and appends .npy
            s3.put_object(Bucket=bucket_name, Key=f"{OUTPUT_FOLDER+s+p}")

    print("Uploading finished")

    #Optional step: write preprocess_results dictionary to csv file on EC2 instance
    #For inspecting which TESS id's timed out or threw errors
    #Upload to S3 bucket?
    with open('/home/ubuntu/spaceLab/tess_id_results.csv', 'w') as f:
        writer = csv.writer(f)
        for k, v in preprocess_results.items():
            writer.writerow([k, v])