import os

import lightkurve
import pandas as pd
from preprocess import preprocess_tess_data

tess_id = pd.read_csv("testID.csv", skiprows=4)

# print(test_id["TIC"])


# preprocess_tess_data(tic[0])
# print(date[0])
current_dir = os.getcwd()
tess_id_date = str(tess_id["Updated"][0]).replace(":", "-")

path = os.path.join(current_dir, tess_id_date)

print(path)

os.mkdir(path)

print(f"Directory created for current tess id at {path}")