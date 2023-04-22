import boto3
import preprocess
from glob import glob
from pathlib import Path

# Initializing the s3 bucket name
bucket_name = "preprocessed-data-2023q1"

# Fetch TESS data from Lightkurve API
tess_data = preprocess.fetch_tess_data_df()


# Loop will go through each of the entries in the
# .csv file (tess_data) and preprocess each of the 
# given light curves
for index, row in tess_data.iterrows():
    tic_id = str(row['TIC ID'])
    preprocess.preprocess_tess_data(tic_id, tess_data)
    

print("Preprocessing finished")

print("Beginning upload to S3 bucket")
#Path to preprocess data directory: "/home/ubuntu/spaceLab/tess_data"

s3 = boto3.client('s3')

#Change to directory where tess_data is located
#/home/ubuntu/spaceLab/tess_data

#Folder and subfolders from preprocess.py file
OUTPUT_FOLDER = 'tess_data/' # modified to save to different output folder
subFolders = ['tic_info/', 'locGlo_flux/', 'locGlo_cent/'] #sub folders within the main output folder

#Loop over each file in each subfolder and add to the respective folder in the bucket
#Subfolders will be created if not already in place
for s in subFolders:
    for file in glob(f"/home/ubuntu/spaceLab/{OUTPUT_FOLDER+s}*"):
        p = Path(file).stem+".npy" #Takes the stem of the file name and appends .npy
        s3.put_object(Bucket=bucket_name, Key=f"{OUTPUT_FOLDER+s+p}")

print("Uploading finished")
