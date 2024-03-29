import os
import numpy as np
import pandas as pd

TRAINING_DATASET_PATH = './tess_data'
PREDICTION_DATASET_PATH = './tess_data/experimental'



DEFAULT_COLUMNS = ['TIC', 'TCE', 'label', 'period', 'epoch', 'duration', 'Teff', 'logg', 
                    'metallicity', 'mass', 'stellar_radius', 'SNR', 'TESS_Mag', 'PM',
                    'density', 'a/Rs', 'depth', 'radius_ratio', 'impact_parameter_b', 
                    'logRp_Rearth']
NORMALIZED_COLUMNS = ['period', 'epoch', 'duration', 'Teff', 'logg', 'metallicity', 'mass', 
                        'stellar_radius', 'SNR', 'TESS_Mag', 'PM', 'density', 'a/Rs', 'depth', 
                        'radius_ratio', 'impact_parameter_b', 'logRp_Rearth']

def load_all_stellar_parameters(filepath):
    """
    Loads all *info.npy files from a provided folder.

    filepath = folder in which *info.npy files live
    returns parameters_dict = dictionary where key:value is filename:stellar_parameters_array
    """

    parameters_dict = {}

    for filename in os.listdir(filepath):
        if filename.endswith('info.npy'):
            array_path = os.path.join(filepath, filename)
            parameters_dict[array_path] = np.load(array_path)

    return parameters_dict

def convert_parameters_dict_to_dataframe(parameters_dict):
    
    stellar_params_df = pd.DataFrame()

    for array_path, parameters in parameters_dict.items():
        s = pd.Series(parameters, name=array_path)
        stellar_params_df = pd.concat([stellar_params_df,s], axis=1, copy=False)

    stellar_params_df = stellar_params_df.T
    stellar_params_df.columns = DEFAULT_COLUMNS

    return stellar_params_df

def normalize_stellar_parameters(stellar_parameters_dataframe):
    """
    Normalizes stellar parameters across provided dataset by subtracting the 
    median and dividing by the standard deviation.
    """
    df_subset = stellar_parameters_dataframe[NORMALIZED_COLUMNS]

    if not np.any(np.isnan(df_subset.std())):

        df_subset = (df_subset-df_subset.median())/df_subset.std()

        stellar_parameters_dataframe[NORMALIZED_COLUMNS] = df_subset

def save_normalized_parameters(stellar_paramaters_dataframe):
    stellar_paramaters_dataframe.T.apply(lambda row: np.save(row.name, row.to_numpy()))

if __name__ == '__main__':
    print('Loading data . . .')
    train_dict = load_all_stellar_parameters(TRAINING_DATASET_PATH)
    experimental_dict = load_all_stellar_parameters(PREDICTION_DATASET_PATH)
    
    if len(experimental_dict) > 0:
        parameters_dict = train_dict | experimental_dict
    else:
        parameters_dict = train_dict

    print('Converting to DataFrame . . .')
    stellar_params_df = convert_parameters_dict_to_dataframe(parameters_dict)

    print('Normalizing data . . .')
    normalize_stellar_parameters(stellar_params_df)

    print('Saving normalized data . . .')
    save_normalized_parameters(stellar_params_df)

    print('Done!')