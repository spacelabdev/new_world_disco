import logging
import io
import os
import boto3

import lightkurve as lk
import numpy as np
import pandas as pd
import requests
import math
from astropy import units as u

logging.captureWarnings(True)
logger = logging.getLogger(__name__)

logger.setLevel(logging.DEBUG)
handler = logging.FileHandler('preprocess.log')
logger.addHandler(handler)


TESS_DATA_URL = 'https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=csv'
LOCAL_DATA_FILE_NAME = 'tess_data.csv'
DEFAULT_TESS_ID =  '2016376984' # a working 'v-shaped' lightcurve. Eventually we'll need to run this for all lightcurves from tess
BJD_TO_BCTJD_DIFF = 2457000
S3_BUCKET = 'preprocess-tess-data-bucket'
OUTPUT_FOLDER = 'tess_data/' # modified to save to different output folder
EXPERIMENTAL_FOLDER = 'experimental/' # folder for planets with unknown status
LIGHTKURVE_CACHE_FOLDER = 'lightkurve-cache/'
EARTH_RADIUS = 6378.1

# these bin numbers for TESS from Yu et al. (2019) section 2.3: https://iopscience.iop.org/article/10.3847/1538-3881/ab21d6/pdf
global_bin_width_factor = 2001
local_bin_width_factor = 201

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
    
    # print("Loading lightcurves")

    lcs = download_lightcurves(id_string)

    # print("Stitching lightcurves")

    lc_raw = lcs.stitch()

    # print("Fetching period and duration")

    # Fetch period and duration data from caltech exofop for tess
    data_df = fetch_tess_data_df()

    threshold_crossing_events = data_df[data_df['TIC ID'] == int(tess_id)]

    # some stellar bodies have multiple threshold crossing events
    # we want to generate for each threshold crossing event
    tce_count = threshold_crossing_events.shape[0]

    # print("Processing outliers")

    # sigma set following Ansdell et al https://www.aanda.org/articles/aa/full_html/2020/01/aa35345-19/aa35345-19.html
    lc_clean = lc_raw.remove_outliers(sigma=3.5)

    for i in range(tce_count):
        
        period, duration = threshold_crossing_events['Period (days)'].iloc[i].item(),  threshold_crossing_events['Duration (hours)'].iloc[i].item()
        t0 = threshold_crossing_events['Epoch (BJD)'].iloc[i].item() - BJD_TO_BCTJD_DIFF

        info = extract_stellar_parameters(threshold_crossing_events, tess_id, period, duration, i)

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

        lc_global = lc_fold.bin(time_bin_size=period/global_bin_width_factor, n_bins=global_bin_width_factor).normalize() - 1

        if np.sum(np.isnan(lc_global.flux))/len(lc_global.flux) > 0.25:
            logger.info(f'{tess_id} global view contains > 0.25 NaNs.')
            return

        lc_global = (lc_global / np.abs(np.nanmin(lc_global.flux)) )

        # fill nans and add gaussian noise
        if np.any(np.isnan(lc_global.flux)):
            lc_global.flux = add_gaussian_noise(lc_global.flux)   

        # sometimes we get the wrong number of bins, so we have to abort
        if not (len(lc_global) == global_bin_width_factor):
            logger.info(f'{tess_id} lc_global incorrect dimension: {len(lc_global)}')
            return

        # print("Creating local representation")

        phase_mask = (lc_fold.phase > -4*fractional_duration) & (lc_fold.phase < 4.0*fractional_duration)
        lc_zoom = lc_fold[phase_mask]

        # we use 8x fractional duration here since we zoomed in on 4x the fractional duration on both sides
        lc_local = lc_zoom.bin(time_bin_size=8*fractional_duration/local_bin_width_factor, n_bins=local_bin_width_factor).normalize() - 1

        if np.sum(np.isnan(lc_local.flux))/len(lc_local.flux) > 0.25:
            logger.info(f'{tess_id} local view contains > 0.25 NaNs.')
            return

        lc_local = (lc_local / np.abs(np.nanmin(lc_local.flux)) )

        # fill nans and add gaussian noise
        if np.any(np.isnan(lc_local.flux)):
            lc_local.flux = add_gaussian_noise(lc_local.flux)        
            

        # sometimes we get the wrong number of bins, so we have to abort
        if not (len(lc_local) == local_bin_width_factor):
            logger.info(f'{tess_id} lc_local incorrect dimension: {len(lc_local)}')
            return

        # print(lc_local.dtype.names)

        # add centroid preprocessing
        local_cen, global_cen = preprocess_centroid(lc_local, lc_global)
        
        if info[2] == -1:
            out = f'{OUTPUT_FOLDER}/{EXPERIMENTAL_FOLDER}/'
        else:
            out = OUTPUT_FOLDER

        # export
        export_lightcurve(lc_local, f"{out+str(tess_id)}_0{int(info[1])}_local")
        export_lightcurve(lc_global, f"{out+str(tess_id)}_0{int(info[1])}_global")

        np.save(f"{out+str(tess_id)}_0{int(info[1])}_info.npy", np.array(info))

        # export
        np.save(f"{out+str(tess_id)}_0{int(info[1])}_local_cen.npy", local_cen)
        np.save(f"{out+str(tess_id)}_0{int(info[1])}_global_cen.npy", global_cen)

        # export_data_to_s3(lc_local, out, f'{str(tess_id)}_0{int(info[1])}_local')
        # export_data_to_s3(lc_global, out, f'{str(tess_id)}_0{int(info[1])}_global')
        # export_data_to_s3(info, out, f'{str(tess_id)}_0{int(info[1])}_info')
        # export_data_to_s3(local_cen, out, f'{str(tess_id)}_0{int(info[1])}_local_cen')
        # export_data_to_s3(global_cen, out, f'{str(tess_id)}_0{int(info[1])}_global_cen')

def download_lightcurves(id_string):
    q = lk.search_lightcurve(id_string, author=['SPOC','TESS-SPOC'])
    
    if len(q) == 0:
        q = lk.search_lightcurve(id_string, author='QLP')

    # to increase processing speed we'll remove short cadences
    if len(q) > 20:
        q = q[q.exptime >= u.Quantity(600, u.s)]

    if len(q) == 0:
        return

    return q.download_all(download_dir=LIGHTKURVE_CACHE_FOLDER)

def add_gaussian_noise(lc_flux):
    # Replace nans with median and add guassian noise
    mu = np.nanmedian(lc_flux)
    rms = np.sqrt(np.nanmean(np.square(lc_flux)))
    lc_flux[np.isnan(lc_flux)] = mu 
    lc_flux = np.random.normal(mu, rms, size = len(lc_flux))
    
    return lc_flux

def export_data_to_s3(data, output_folder, filename):
    """
    Method to save lightcurve data as a pickle object.
    Inputs: data = data to be saved.
            output_folder = folder in which to save file.
            filename = name of the file.
    """
    s3 = boto3.client('s3')

    if not 'Contents' in s3.list_objects(Bucket=S3_BUCKET, Prefix=output_folder):
        s3.put_object(Bucket=S3_BUCKET, Key=output_folder)

    # we use an in-memory bytes buffer so we don't have to save locally
    bytes_data = io.BytesIO()
    np.save(bytes_data, data, allow_pickle=True)
    bytes_data.seek(0)
    s3.upload_fileobj(bytes_data, S3_BUCKET, f'{output_folder}{filename}.pkl')

def export_lightcurve(lc, filename):
    """
    Method to save lightcurve data as CSV and a NumPy array file (.npy) representing flux.
    Inputs: lc = lightcurve to be saved.
            folder = folder in which to save file.
            filename = name of the file.
    """

    if not os.path.isdir(OUTPUT_FOLDER):
        os.mkdir(os.path.join(os.getcwd(), OUTPUT_FOLDER))

#   lc.to_csv(f"./data/{filename}.csv", overwrite=True)
    np.save(f"{filename}_flux.npy", np.array(lc['flux']))


### Info Preprocessing
def extract_stellar_parameters(threshold_crossing_events, tess_id, period, duration, i):
    # info contains: [0]tic, [1]tce, [2]label, [3]period, [4]epoch, [5]duration, 
    # [6]Teff, [7]logg, [8]metallicity, [9]mass, [10]stellar_radius, [11]SNR, [12]TESS_Mag, 
    # [13]proper_motion, [14]density, [15]a/Rs, [16]depth, [17]radius_ratio, 
    # [18]impact_parameter_b, [19]logRp_Rearth
    info = np.full((20,), 0, dtype=np.float64)

    info[0] = tess_id
    info[1] = i + 1

    # if label is -1, these are unknowns for the experimental set
    if threshold_crossing_events['TESS Disposition'].iloc[i] in ['KP', 'CP']:
        info[2] = 1
    elif threshold_crossing_events['TESS Disposition'].iloc[i] in ['FA', 'FP']:
        info[2] = 0
    else:
        info[2] = -1

    info[3] = period
    info[4] = threshold_crossing_events['Epoch (BJD)'].iloc[i].item()
    info[5] = duration

    

    Teff = threshold_crossing_events['Stellar Eff Temp (K)'].iloc[i].item()
    logg = threshold_crossing_events['Stellar log(g) (cm/s^2)'].iloc[i].item()
    metallicity = threshold_crossing_events['Stellar Metallicity'].iloc[i].item()
    mass = threshold_crossing_events['Stellar Mass (M_Sun)'].iloc[i].item()
    stellar_radius = threshold_crossing_events['Stellar Radius (R_Sun)'].iloc[i].item()
    planet_snr = threshold_crossing_events['Planet SNR'].iloc[i].item()
    tess_mag = threshold_crossing_events['TESS Mag'].iloc[i].item()
    proper_motion = threshold_crossing_events['PM RA (mas/yr)'].iloc[i].item()

    for i, param in enumerate([Teff, logg, metallicity, mass, stellar_radius, planet_snr, tess_mag, proper_motion]):
        if not np.isnan(param):
            info[5+i+1] = param

    # we need to download the remaining parameters
    stellar_params_link = f'https://exofop.ipac.caltech.edu/tess/download_stellar.php?id={tess_id}&output=csv'

    stellar_params = pd.read_csv(stellar_params_link, sep='|')
    
    densities = stellar_params['Density (g/cm^3)']

    if not np.all(densities.isna()):
        info[14] = densities.dropna().iloc[0].item()

    planet_params_link = f'https://exofop.ipac.caltech.edu/tess/download_planet.php?id={tess_id}&output=csv'

    planet_params = pd.read_csv(planet_params_link, sep='|')

    a_rs = planet_params['a/Rad_s']
    depths = planet_params['Depth (ppm)']
    radius_ratios = planet_params['Rad_p/Rad_s']
    impact_param_b = planet_params['Impact Parameter b']

    for i, param_list in enumerate([a_rs, depths, radius_ratios, impact_param_b]):
        if not np.all(param_list.isna()):
            info[14+i+1] = param_list.dropna().iloc[0].item()
    
    planet_radius = planet_params['Radius (R_Earth)']
    arb_boundary = 13*EARTH_RADIUS
    if not np.all(planet_radius.isna()):
        logRp_Rb = np.log(planet_radius.iloc[0].item())/arb_boundary
        info[19] = logRp_Rb

    return info

### Centroid Preprocessing
# section 2.2 of https://arxiv.org/pdf/1810.13434.pdf describes how they normalized
def normalize_centroid(centroid_data):
    # normalize by subtracting median and dividing by standard deviation
    med = np.nanmedian(centroid_data)
    std = np.nanstd(centroid_data)
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
    elif 'centroid_col' in lc_global.columns and 'centroid_row' in lc_global.columns:
        global_x = np.array([float(x/u.pix) for x in lc_global['centroid_row']])
        global_y = np.array([float(y/u.pix) for y in lc_global['centroid_col']])
        local_x = np.array([float(x/u.pix) for x in lc_local['centroid_row']])
        local_y = np.array([float(y/u.pix) for y in lc_local['centroid_col']])
    else:
        # TO DO: check for centroid_row, centroid_col and do any preprocessing they might require
        logger.info("Error: preprocess_centroid(): No centroid data")
        return

    # compute r = sqrt(x^2 + y^2) for each centroid location
    local_cen = np.array([get_mag(x,y) for x, y in zip(local_x, local_y)])
    global_cen = np.array([get_mag(x,y) for x, y in zip(global_x, global_y)])

    # normalize by subtracting median and dividing by standard deviation
    normalize_centroid(local_cen)
    normalize_centroid(global_cen)

    return local_cen, global_cen

if __name__ == "__main__":
    preprocess_tess_data()