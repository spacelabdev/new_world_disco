import preprocess

import lightkurve as lk
import pandas as pd

df=pd.read_csv("tois.csv",skiprows=4)

def newfunc():
    for tic in df['TIC'][:10]:
        try:
            preprocess.preprocess_tess_data(tic)
        except lightkurve.utils.LightkurveError as e:
            print(e.str())
    
newfunc()