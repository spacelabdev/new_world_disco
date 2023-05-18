import requests

TESS_DATA_URL = 'https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=csv'
LOCAL_DATA_FILE_NAME = '/home/ubuntu/spaceLab/tess_data.csv'

def fetch_tess_data_df():
    """
    Method to load TESS data. 

    If data does not exist locally, it will be downloaded from
    TESS_DATA_URL and saved locally.  As more TESS data comes in,
    the existing file will expand to include the newer data.
    """

    res = requests.get(TESS_DATA_URL)
    tess_data_raw = res.content
    with open(LOCAL_DATA_FILE_NAME, 'wb+') as f:
        f.write(tess_data_raw)

if __name__ == "__main__":
    fetch_tess_data_df()