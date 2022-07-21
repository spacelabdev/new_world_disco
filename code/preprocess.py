import io
import os

import lightkurve as lk
import numpy as np
import pandas as pd
import requests
import math
from astropy import units as u

TESS_DATA_URL = 'https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=csv'
LOCAL_DATA_FILE_NAME = 'tess_data.csv'
DEFAULT_TESS_ID = '2016376984' # a working 'v-shaped' lightcurve. Eventually we'll need to run this for all lightcurves from tess
BJD_TO_BCTJD_DIFF = 2457000
OUTPUT_FOLDER = 'tess_data/' # modified to save to different output folder

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

    # Download and stitch all lightcurve quarters together.
    id_string = f'TIC {tess_id}'
    q = lk.search_lightcurve(id_string)
    lcs = q.download_all()
    lc_raw = lcs.stitch()

    # Fetch period and duration data from caltech exofop for tess
    data_df = fetch_tess_data_df()

    period, duration = data_df[data_df['TIC ID'] == int(tess_id)]['Period (days)'].item(),  data_df[data_df['TIC ID'] == int(tess_id)]['Duration (hours)'].item()
    t0 = data_df[data_df['TIC ID'] == int(tess_id)]['Epoch (BJD)'].item() - BJD_TO_BCTJD_DIFF

    lc_clean = lc_raw.remove_outliers(sigma=3)

    # Do the hacky masking from here: https://docs.lightkurve.org/tutorials/3-science-examples/exoplanets-machine-learning-preprocessing.html
    temp_fold = lc_clean.fold(period, epoch_time=t0)
    fractional_duration = (duration / 24.0) / period
    phase_mask = np.abs(temp_fold.phase.value) < (fractional_duration * 1.5)
    transit_mask = np.in1d(lc_clean.time.value, temp_fold.time_original.value[phase_mask])

    lc_flat = lc_clean.flatten(mask=transit_mask)
    lc_fold = lc_flat.fold(period, epoch_time=t0)

    lc_global = lc_fold.bin(time_bin_size=period/global_bin_width_factor).normalize() - 1
    #lc_global = (lc_global / np.abs(lc_global.flux.min()) ) * 2.0 + 1
    lc_global = (lc_global / np.abs(np.nanmin(lc_global.flux)) ) * 2.0 + 1

    phase_mask = (lc_fold.phase > -4*fractional_duration) & (lc_fold.phase < 4.0*fractional_duration)
    lc_zoom = lc_fold[phase_mask]

    # we use 8x fractional duration here since we zoomed in on 4x the fractional duration on both sides
    lc_local = lc_zoom.bin(time_bin_size=8*fractional_duration/local_bin_width_factor).normalize() - 1
    lc_local = (lc_local / np.abs(np.nanmin(lc_local.flux)) ) * 2.0 + 1

    # export
    export_lightcurve(lc_local, f"{tess_id}_local")
    export_lightcurve(lc_global, f"{tess_id}_global")

    # add centroid preprocessing
    local_cen, global_cen = preprocess_centroid(lc_local, lc_global)

    # export
    np.save(f"{OUTPUT_FOLDER+tess_id}_local_cen.npy", local_cen)
    np.save(f"{OUTPUT_FOLDER+tess_id}_global_cen.npy", global_cen)


def export_lightcurve(lc, filename):
    """
    Method to save lightcurve data as CSV and a NumPy array file (.npy) representing flux.

    Inputs: lc = lightcurve to be saved.
            folder = folder in which to save file.
            filename = name of the file.
    """

    if not os.path.isdir('data'):
        os.mkdir(os.path.join(os.getcwd(), 'data'))

    lc.to_csv(f"./data/{filename}.csv", overwrite=True)
    np.save(f"./data/{filename}_flux.npy", np.array(lc['flux']))


### Centroid Preprocessing
# section 2.2 of https://arxiv.org/pdf/1810.13434.pdf describes how they normalized
def normalize_centroid(centroid_data):
    # normalize by subtracting median and dividing by standard deviation
    med = np.median(centroid_data)
    std = np.std(centroid_data)
    centroid_data -= med
    centroid_data /= std

def get_mag(x, y):
    # get magnitude as: sqrt(x^2 + y^2)
    return math.sqrt(x*x + y*y)

def preprocess_centroid(lc_local, lc_global):
    """
    Method for preprocessing TESS centroid data

    Input: local and global lightcurve objects (already pre-processed)
    Output: local and global centroid position numpy arrays
    """
    global_x = np.array([float(x/u.pix) for x in lc_global['sap_x']])
    global_y = np.array([float(y/u.pix) for y in lc_global['sap_y']])
    local_x = np.array([float(x/u.pix) for x in lc_local['sap_x']])
    local_y = np.array([float(y/u.pix) for y in lc_local['sap_y']])

    local_cen = np.array([get_mag(x,y) for x, y in zip(local_x, local_y)])
    global_cen = np.array([get_mag(x,y) for x, y in zip(global_x, global_y)])

    normalize_centroid(local_cen)
    normalize_centroid(global_cen)

    return local_cen, global_cen

if __name__ == "__main__":
    preprocess_tess_data()
