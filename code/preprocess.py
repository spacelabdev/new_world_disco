import logging
import io
import os

import lightkurve as lk
import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

logger.setLevel(logging.INFO)
handler = logging.FileHandler('preprocess.log')
logger.addHandler(handler)

TESS_DATA_URL = 'https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=csv'
LOCAL_DATA_FILE_NAME = 'tess_data.csv'
DEFAULT_TESS_ID = '2016376984' # a working 'v-shaped' lightcurve. Eventually we'll need to run this for all lightcurves from tess
BJD_TO_BCTJD_DIFF = 2457000
OUTPUT_FOLDER = '.'

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

def preprocess_tess_data(tess_id=DEFAULT_TESS_ID):
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

    if os.path.isfile(f'./data/{tess_id}_global_flux.npy'):
        return

    # Download and stitch all lightcurve quarters together.
    id_string = f'TIC {tess_id}'
    
    # print("Loading lightcurves")

    q = lk.search_lightcurve(id_string)
    lcs = q.download_all()

    # print("Stitching lightcurves")

    lc_raw = lcs.stitch()

    # print("Fetching period and duration")

    # Fetch period and duration data from caltech exofop for tess
    data_df = fetch_tess_data_df()

    threshold_crossing_events = data_df[data_df['TIC ID'] == int(tess_id)]

    tce_count = threshold_crossing_events.shape[0]

    for i in range(tce_count):
        
        period, duration = threshold_crossing_events['Period (days)'].iloc[i].item(),  threshold_crossing_events['Duration (hours)'].iloc[i].item()
        t0 = threshold_crossing_events['Epoch (BJD)'].iloc[i].item() - BJD_TO_BCTJD_DIFF

        # info contains: [0]tic, [1]tce, [2]period, [3]epoch, [4]duration, [5]label,
        # [6]Teff, [7]logg, [8]metallicity, [9]mass, [10]radius, [11]density
        info = np.full((12,), np.nan)

        print(threshold_crossing_events[[
            'Epoch (BJD)',
            'TFOPWG Disposition',
            'Stellar Eff Temp (K)',
            'Stellar log(g) (cm/s^2)',
            'Stellar Metallicity',
            'Stellar Mass (M_Sun)',
            'Stellar Radius (R_Sun)']])

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
    

    
        # export
        export_lightcurve(lc_local, f"{tess_id}_local_0{i+1}")
        export_lightcurve(lc_global, f"{tess_id}_global_0{i+1}")
        export_info(info, f"{tess_id}_0{i+1}")

def export_info(info, filename):
    np.save(f"./data/{filename}_info.npy", np.array(info))


def export_lightcurve(lc, filename):
    """
    Method to save lightcurve data as CSV and a NumPy array file (.npy) representing flux.

    Inputs: lc = lightcurve to be saved.
            folder = folder in which to save file.
            filename = name of the file.
    """

    if not os.path.isdir('data'):
        os.mkdir(os.path.join(os.getcwd(), 'data'))

#    lc.to_csv(f"./data/{filename}.csv", overwrite=True)
    np.save(f"./data/{filename}_flux.npy", np.array(lc['flux']))


if __name__ == "__main__":
    preprocess_tess_data()
