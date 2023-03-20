import boto3
import pandas as pd
import preprocess

# Initializing the s3 bucket name
bucket_name = "preprocessed-data-2023-q1"

# Fetch TESS data from Lightkurve API
tess_data = preprocess.fetch_tess_data_df()


# Loop will go through each of the entries in the
# .csv file (tess_data) and preprocess each of the 
# given light curves
for index, row in tess_data.iterrows():
    tic_id = row['TIC ID']
    #print(tic_id)
    preprocess.preprocess_tess_data(tic_id, tess_data) 
    
    

'''

# Upload preprocess data to S3 bucket
s3 = boto3.resource("s3")
s3.meta.client.upload_file("/path/to/folder", bucket_name, "/path/to/folder")
'''
