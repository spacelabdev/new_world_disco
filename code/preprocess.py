import logging
import io
import os

import lightkurve as lk
import numpy as np
import pandas as pd
import requests
import math
from astropy import units as u
import warnings


logger = logging.getLogger(__name__)

logger.setLevel(logging.INFO)
handler = logging.FileHandler('preprocess.log')
logger.addHandler(handler)

TESS_DATA_URL = 'https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=csv'
LOCAL_DATA_FILE_NAME = 'tess_data.csv'
DEFAULT_TESS_ID = '2016376984' # a working 'v-shaped' lightcurve. Eventually we'll need to run this for all lightcurves from tess
BJD_TO_BCTJD_DIFF = 2457000
OUTPUT_FOLDER = 'tess_data/' # modified to save to different output folder
subFolders = ['tic_info/', 'locGlo_flux/', 'locGlo_cent/']

# these bin numbers for TESS from Yu et al. (2019) section 2.3: https://iopscience.iop.org/article/10.3847/1538-3881/ab21d6/pdf
global_bin_width_factor = 201
local_bin_width_factor = 61

def fetch_tess_data_df():
    """
    Method to load TESS data. 

    If data does not exist locally, it will be downloaded from
    TESS_DATA_URL and saved locally.
    """

    if os.path.isfile(LOCAL_DATA_FILE_NAME):
        return pd.read_csv(LOCAL_DATA_FILE_NAME)
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

    print(tess_id)
    # Download and stitch all lightcurve quarters together.
    id_string = f'TIC {tess_id}'
    
    # print("Loading lightcurves")
    # This is the section of the code that is causing problems
    # the .download_all()
    q = lk.search_lightcurve(id_string)
    lcs = q.download_all()

    # print("Stitching lightcurves")

    lc_raw = lcs.stitch()

    # print("Fetching period and duration")

    # Fetch period and duration data from caltech exofop for tess
    # Commented this line of code out from the orginal script
    # data_df = fetch_tess_data_df()

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


        # print("Processing outliers")

        lc_clean = lc_raw.remove_outliers(sigma=3)

        # print("Masking hack")

        # Do the hacky masking from here: https://docs.lightkurve.org/tutorials/3-science-examples/exoplanets-machine-learning-preprocessing.html
        temp_fold = lc_clean.fold(period, epoch_time=t0)
        fractional_duration = (duration / 24.0) / period
        phase_mask = np.abs(temp_fold.phase.value) < (fractional_duration * 1.5)
        transit_mask = np.in1d(lc_clean.time.value, temp_fold.time_original.value[phase_mask])


        # print("Flattening lightcurve")

        lc_flat = lc_clean.flatten(mask=transit_mask)
        lc_fold = lc_flat.fold(period, epoch_time=t0)

        # print("Creating global representation")

        lc_global = lc_fold.bin(time_bin_size=period/global_bin_width_factor).normalize() - 1
        if not (len(lc_global) == global_bin_width_factor):
            logger.info(f'{tess_id} lc_global incorrect dimension: {len(lc_global)}')
            return
        lc_global = (lc_global / np.abs(np.nanmin(lc_global.flux)) ) * 2.0 + 1

        # print("Creating local representation")

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
        local_cen, global_cen = preprocess_centroid(lc_local, lc_global)
        
        # export
        export_lightcurve(lc_local, f"{tess_id}_0{i+1}_local")
        export_lightcurve(lc_global, f"{tess_id}_0{i+1}_global")

        np.save(f"{OUTPUT_FOLDER+subFolders[0]+str(tess_id)}_0{i+1}_info.npy", np.array(info))

        # export
        np.save(f"{OUTPUT_FOLDER+subFolders[2]+str(tess_id)}_0{i+1}_local_cen.npy", local_cen)
        np.save(f"{OUTPUT_FOLDER+subFolders[2]+str(tess_id)}_0{i+1}_global_cen.npy", global_cen)



def export_lightcurve(lc, filename):
    """
    Method to save lightcurve data as CSV and a NumPy array file (.npy) representing flux.

    Inputs: lc = lightcurve to be saved.
            folder = folder in which to save file.
            filename = name of the file.
    """

    if not os.path.isdir(OUTPUT_FOLDER):
        os.mkdir(os.path.join(os.getcwd(), OUTPUT_FOLDER))

    for subfolder in subFolders:
        if not os.path.isdir(subfolder):
            os.makedirs(os.path.join(OUTPUT_FOLDER, subfolder), exist_ok=True)

#   lc.to_csv(f"./data/{filename}.csv", overwrite=True)
    np.save(f"{OUTPUT_FOLDER+subFolders[1]+str(filename)}_flux.npy", np.array(lc['flux']))


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
        global_x = np.array([float(x/u.pix) for x in lc_global['sap_x']])
        global_y = np.array([float(y/u.pix) for y in lc_global['sap_y']])
        local_x = np.array([float(x/u.pix) for x in lc_local['sap_x']])
        local_y = np.array([float(y/u.pix) for y in lc_local['sap_y']])
    else:
        # TO DO: check for centroid_row, centroid_col and do any preprocessing they might require
        logger.info("Error: preprocess_centroid(): No handling for centroid data not stored in sap_x, sap_y")
        return

    # compute r = sqrt(x^2 + y^2) for each centroid location
    local_cen = np.array([get_mag(x,y) for x, y in zip(local_x, local_y)])
    global_cen = np.array([get_mag(x,y) for x, y in zip(global_x, global_y)])

    # normalize by subtracting mean and dividing by standard deviation
    normalize_centroid(local_cen)
    normalize_centroid(global_cen)

    return local_cen, global_cen

if __name__ == "__main__":
    preprocess_tess_data()
