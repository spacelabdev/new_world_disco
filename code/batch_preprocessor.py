import preprocess

import os
import io

import pandas as pd
import requests

TOI_CATALOG_URL = 'https://archive.stsci.edu/missions/tess/catalogs/toi/tois.csv'
TOI_CATALOG_FILENAME = "tois.csv"

def fetch_tic_list(num_tic = -1):
    """
    Method to load list of TESS Input Catalog identifier (TICs). If the TOI catalog does 
    not exist locally, this method will load it from the url and save a local copy.

    Input: num_tic = number of TOIs requested (if unspecified, returns entire catalog).
    Returns: List of length num_tois filled with TOI identifiers.
    """

    # Validate input
    if type(num_tic) != int:
        raise TypeError(f"Received non-integer input {num_tic}.\nExpected integer > 0.")
    if num_tic < -1 or num_tic == 0:
        raise ValueError("The number of requested TICs is invalid.\nExpected integer > 0.")

    # Check to see if local file exists and load from url and save locally if it doesn't
    if os.path.exists(TOI_CATALOG_FILENAME):
        toi_df = pd.read_csv(TOI_CATALOG_FILENAME)
    else:
        toi_df = pd.read_csv(TOI_CATALOG_URL, header=4)
        toi_df.to_csv(TOI_CATALOG_FILENAME)

    tics = toi_df['TIC']

    if num_tic == -1 or num_tic >= len(tics):
        return tics
    else:
        return tics.iloc[:num_tic]

def batch_preprocessor(num_tic):
    """
    Method to preprocess a user-defined number of TESS data objects from the TOI catalog.

    TODOs:   1. Try to parallelize. Currently very slow.
             2. In some of the CSVs created there are NaNs. Discard or is that expected?
             3. Suppress warnings? Many warnings about incompatible columns.
             4. More to do with preprocess.preprocess_tess_data, but saving these files
                in a separate data folder would be ideal.
                
    Input: num_tic = number of TOIs requested.
    Returns: number of TOIs successfully processed.
    """

    num_processed = 0

    tics = fetch_tic_list(num_tic)

    for tic in tics:
        try:
            preprocess.preprocess_tess_data(tic)
            num_processed += 1
        except Exception as e:
            print(f"\nCould not load TIC {tic}. See Error below: \n {e}\n")

    print(f"\n{num_processed} of {num_tic} TESS Objects of Interest processed.")

    return num_processed

if __name__ == "__main__":
    batch_preprocessor(10)
