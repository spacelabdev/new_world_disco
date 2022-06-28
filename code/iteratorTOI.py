import preprocess
import lightkurve
import pandas as pd



def TOI_iterator(file, n_TOI):
    df=pd.read_csv(file,header=4)
    if n_TOI==None:
        n_TOI=len(df)
    if 0>n_TOI>len(df):
        return "n_TOI out of range"

    for tic in df["TIC"][:n_TOI]:
        try:
            preprocess.preprocess_tess_data(tic)
        except lightkurve.utils.LightkurveError as e:
            print(str(e))
        



TOI_iterator("tois.csv",10)