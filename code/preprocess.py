import io
import os

import lightkurve as lk
import numpy as np
import pandas as pd
import requests

TESS_DATA_URL = 'https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=csv'
LOCAL_DATA_FILE_NAME = 'tess_data.csv'
DEFAULT_TESS_ID = '2016376984' # a working 'v-shaped' lightcurve. Eventually we'll need to run this for all lightcurves from tess
BJD_TO_BCTJD_DIFF = 2457000
OUTPUT_FOLDER = '.'

# these bin numbers for TESS from Yu et al. (2019) section 2.3: https://iopscience.iop.org/article/10.3847/1538-3881/ab21d6/pdf
global_bin_width_factor = 201
local_bin_width_factor = 61

def fetch_tess_data_df():
    if os.path.isfile(LOCAL_DATA_FILE_NAME):
        return pd.read_csv(LOCAL_DATA_FILE_NAME)
    res = requests.get(TESS_DATA_URL)
    tess_data_raw = res.content
    with open(LOCAL_DATA_FILE_NAME, 'wb+') as f:
        f.write(tess_data_raw)
    return pd.read_csv(io.BytesIO(tess_data_raw))

def preprocess_tess_data(tess_id=DEFAULT_TESS_ID):
    id_string = f'TIC {tess_id}'
    q = lk.search_lightcurve(id_string)
    lcs = q.download_all()
    lc_raw = lcs.stitch()

    # fetch period and duration data from caltech exofop for tess
    data_df = fetch_tess_data_df()
    period, duration = data_df[data_df['TIC ID'] == int(tess_id)]['Period (days)'].item(),  data_df[data_df['TIC ID'] == int(tess_id)]['Duration (hours)'].item()
    t0 = data_df[data_df['TIC ID'] == int(tess_id)]['Epoch (BJD)'].item() - BJD_TO_BCTJD_DIFF

    lc_clean = lc_raw.remove_outliers(sigma=3)

    # do the hacky masking from here: https://docs.lightkurve.org/tutorials/3-science-examples/exoplanets-machine-learning-preprocessing.html
    temp_fold = lc_clean.fold(period, epoch_time=t0)
    fractional_duration = (duration / 24.0) / period
    phase_mask = np.abs(temp_fold.phase.value) < (fractional_duration * 1.5)
    transit_mask = np.in1d(lc_clean.time.value, temp_fold.time_original.value[phase_mask])

    lc_flat = lc_clean.flatten(mask=transit_mask)
    lc_fold = lc_flat.fold(period, epoch_time=t0)

    lc_global = lc_fold.bin(time_bin_size=period/global_bin_width_factor).normalize() - 1
    lc_global = (lc_global / np.abs(lc_global.flux.min()) ) * 2.0 + 1

    phase_mask = (lc_fold.phase > -4*fractional_duration) & (lc_fold.phase < 4.0*fractional_duration)
    lc_zoom = lc_fold[phase_mask]

    # we use 8x fractional duration here since we zoomed in on 4x the fractional duration on both sides
    lc_local = lc_zoom.bin(time_bin_size=8*fractional_duration/local_bin_width_factor).normalize() - 1
    lc_local = (lc_local / np.abs(np.nanmin(lc_local.flux)) ) * 2.0 + 1

    # export
    export_lightcurve(lc_local, f"{tess_id}_local")
    export_lightcurve(lc_global, f"{tess_id}_global")


def export_lightcurve(lc, filename):
    lc.to_csv(f"{filename}.csv",overwrite=True)
    np.save(f"{filename}_flux.npy", np.array(lc['flux']))


if __name__ == "__main__":
    preprocess_tess_data()
