import os

import logging
import datetime
import pandas as pd
import requests
from dateutil.parser import parse as parsedate
import math
import warnings


from joblib import Parallel, delayed

import preprocess

TOI_CATALOG_URL = 'https://archive.stsci.edu/missions/tess/catalogs/toi/tois.csv'
TOI_CATALOG_FILENAME = "./tois.csv"

logger = logging.getLogger(__name__)

logger.setLevel(logging.INFO)
handler = logging.FileHandler('batch_preprocess.log')
logger.addHandler(handler)

def fetch_tic_list(num_tic = -1):
    """
    Method to load list of TESS Input Catalog identifier (TICs). If the TOI
    catalog does not exist locally, this method will load it from the url and
    save a local copy.

    Input: num_tic = number of TOIs requested (if unspecified, returns entirety).
    Returns: List of length num_tois filled with TOI identifiers.
    """

    download_flag = False

    # Validate input
    if not isinstance(num_tic, int):
        raise TypeError(f"Received non-integer input {num_tic}.\nExpected integer > 0.")
    if num_tic < -1 or num_tic == 0:
        raise ValueError("The number of requested TICs is invalid.\nExpected integer > 0.")

    # If local files exist, ensure they're up to date.
    # If local files don't exist, we need to download from the url.
    if os.path.exists(TOI_CATALOG_FILENAME):
        r_head = requests.head(TOI_CATALOG_URL)
        last_update = parsedate(r_head.headers['Last-Modified']).astimezone()
        last_download = datetime.datetime.fromtimestamp(
            os.path.getmtime(TOI_CATALOG_FILENAME)).astimezone()

        if last_update > last_download:
            print("Newer TOI Catalog available.")
            download_flag = True
    else:
        download_flag = True

    if download_flag:
        print("Downloading TOI Catalog...")
        res = requests.get(TOI_CATALOG_URL)
        tic_data_raw = res.content
        with open(TOI_CATALOG_FILENAME, 'wb+') as f:
            f.write(tic_data_raw)
        print("Download Complete!")

    toi_df = pd.read_csv(TOI_CATALOG_FILENAME, header=4)

    tics = toi_df['TIC']

    print(len(tics))

    if num_tic == -1 or num_tic >= len(tics):
        return tics

    return tics.iloc[:num_tic]

def batch_preprocessor(tics):
    """
    Method to preprocess a user-defined number of TESS data objects from the
    TOI catalog.

    TODOs:   1. Try to parallelize. Currently very slow.
             2. Suppress warnings? Many warnings about incompatible columns.
             3. More to do with preprocess.preprocess_tess_data, but saving
                these files in a separate data folder would be ideal.

    Input: tics = list of TICs.
    Returns: number of TOIs successfully processed.
    """


    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for tic in tics:
            try:
                preprocess.preprocess_tess_data(tic)
                
            except Exception as e:
                pass
                #logger.info(f"\nCould not load TIC {tic}. See Error below: \n {e}\n")
            


if __name__ == "__main__":
    
    items = fetch_tic_list()
    n_workers = 12

    # Divide data in batches
    batch_size = math.ceil(len(items) / n_workers)
    batches = [
        items[ix:ix+batch_size]
        for ix in range(0, len(items), batch_size)
    ]

    
    totals = [len(batch) for batch in batches]

    # Parallel process the batches
    result = Parallel(n_jobs=n_workers, verbose=20)(
        delayed(batch_preprocessor)
        (batch)
        for batch in batches
    )
    

    