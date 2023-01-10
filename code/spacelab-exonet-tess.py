import os
import io
import boto3
import numpy as np
import glob as glob
import torch
from torch.utils.data import Dataset, Subset
from sklearn.model_selection import train_test_split
import multiprocessing

filepath = '/home/ubuntu/SpaceLab/new_world_disco/code/tess_data'
S3_BUCKET = 'preprocess-tess-data-bucket'
S3_PREFIX = 'tess_data/'

# reference: https://gitlab.com/frontierdevelopmentlab/exoplanets/exonet-pytorch/-/blob/master/exonet.py
class TessDataLoader(Dataset):
    
    '''
    
    PURPOSE: DATA LOADER FOR KERPLER LIGHT CURVES
    INPUT: PATH TO DIRECTOR WITH LIGHT CURVES + INFO FILES
    OUTPUT: LOCAL + GLOBAL VIEWS, LABELS
    
    '''

    def __init__(self):

        self.s3_client = boto3.client('s3')
        ### list of global, local, and info files (assumes certain names of files)
        self.flist_global = self.__load_file_list__('global.pkl')
        self.flist_local = self.__load_file_list__('local.pkl')
        self.flist_info = self.__load_file_list__('info.pkl')
        
        print(len(self.flist_global))
        
        ### list of whitened centroid files
        self.flist_global_cen = self.__load_file_list__('global_cen.pkl')
        self.flist_local_cen = self.__load_file_list__('local_cen.pkl')
        
        ### ids = {TIC}_{TCE}
        self.ids = np.sort([(x.split('/')[-1]).split('_')[1] + '_' + (x.split('/')[-1]).split('_')[2] for x in self.flist_global])

    def __len__(self):

        return self.ids.shape[0]

    def __getitem__(self, idx):


        ### grab local and global views
        data_global = self.__load_contents_from_s3__(self.flist_global[idx])
        data_local = self.__load_contents_from_s3__(self.flist_local[idx])

        ### grab centroid views
        data_global_cen = self.__load_contents_from_s3__(self.flist_global_cen[idx])
        data_local_cen = self.__load_contents_from_s3__(self.flist_local_cen[idx])
        
        ### info file contains: [0]kic, [1]tce, [2]period, [3]epoch, [4]duration, [5]label)
        data_info = self.__load_contents_from_s3__(self.flist_info[idx])

        return (data_local, data_global, data_local_cen, data_global_cen, data_info[3:]), data_info[2]

    def __load_file_list__(self, filename_endswith):
        paginator = self.s3_client.get_paginator('list_objects_v2')
        response_iterator = paginator.paginate(Bucket=S3_BUCKET, Prefix=S3_PREFIX, Delimiter='/')

        file_names = []
        for response in response_iterator:
            for object_data in response['Contents']:
                key = object_data['Key']
                if key.endswith(filename_endswith):
                    file_names.append(key)

        return np.sort(file_names)

    def __load_contents_from_s3__(self, filename):
        bytes_data = io.BytesIO()
        self.s3_client.download_fileobj(S3_BUCKET, filename, bytes_data)
        bytes_data.seek(0)
        contents = np.load(bytes_data, allow_pickle=True)

        return contents


### grab data using data loader
tess_data = TessDataLoader()

# look through some of the data
count = 0
print("len(tessdata)", len(tess_data))
for data in tess_data:
    print("len(data):", len(data))
    print("len(data[0]):", len(data[0]))
    print("shape data[0][0]: (data_local):", data[0][0].shape)
    print("shape data[0][1]: (data_global):", data[0][1].shape)
    print("shape data[0][2]: (data_local_cen):", data[0][2].shape)
    print("shape data[0][3]: (data_global_cen):", data[0][3].shape)
    print("shape data[0][4]: (data_info[6:]):", data[0][4].shape)
    print("data[1]:", data[1])
    print()

    count += 1
    if count == 1:
        break

# unload all of the data from pytorch dataset to numpy and save
x = []
y = []

curr = 0
total = len(tess_data)
print("Loading validation data:")
for x_data, y_data in tess_data:
    print(curr, "/", total)
    if not isinstance(y_data, np.float64):
        print(f'ERROR: {y_data}')
    curr += 1
    
    x.append(x_data)
    y.append(y_data)

def train_test_split_dataset(data, target, train_ratio, validation_ratio, test_ratio):
    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=1-train_ratio, shuffle=True, stratify=target)
    
    x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=test_ratio/(test_ratio + validation_ratio), shuffle=True, stratify=y_test)
    
    return x_train, y_train, x_test, y_test, x_val, y_val

x_train, y_train, x_test, y_test, x_val, y_val = train_test_split_dataset(x, y, 0.8, 0.1, 0.1)

x_val = np.asarray(x_val, dtype=object)
y_val = np.asarray(y_val, dtype=np.float32)
x_test = np.asarray(x_test, dtype=object)
y_test = np.asarray(y_test, dtype=np.float32)
x_train = np.asarray(x_train, dtype=object)
y_train = np.asarray(y_train, dtype=np.float32)



print(f'x_val shape: {x_val.shape}')
print(f'x_val[0][0] shape: {x_val[0][0].shape}')
print(f'x_val[0][1] shape: {x_val[0][1].shape}')
print(f'x_val[0][2] shape: {x_val[0][2].shape}')
print(f'x_val[0][3] shape: {x_val[0][3].shape}')
print(f'x_val[0][4] shape: {x_val[0][4].shape}')
print(f'y_val shape: {y_val.shape}')

print(x_val[0][0])
print(x_val[0][1])
print(x_val[0][2])
print(x_val[0][3])
print(x_val[0][4])

from numpy import save
# save the validation data that was loaded
save("/home/ubuntu/SpaceLab/new_world_disco/outputs/val_x_data.npy", x_val)
save("/home/ubuntu/SpaceLab/new_world_disco/outputs/val_y_data.npy", y_val)
save("/home/ubuntu/SpaceLab/new_world_disco/outputs/test_x_data.npy", x_test)
save("/home/ubuntu/SpaceLab/new_world_disco/outputs/test_y_data.npy", y_test)
save("/home/ubuntu/SpaceLab/new_world_disco/outputs/train_x_data.npy", x_train)
save("/home/ubuntu/SpaceLab/new_world_disco/outputs/train_y_data.npy", y_train)

from numpy import load

x2_val = load("/home/ubuntu/SpaceLab/new_world_disco/outputs/val_x_data.npy", allow_pickle=True)
y2_val = load("/home/ubuntu/SpaceLab/new_world_disco/outputs/val_y_data.npy", allow_pickle=True)
x2_test = load("/home/ubuntu/SpaceLab/new_world_disco/outputs/test_x_data.npy", allow_pickle=True)
y2_test = load("/home/ubuntu/SpaceLab/new_world_disco/outputs/test_y_data.npy", allow_pickle=True)
x2_train = load("/home/ubuntu/SpaceLab/new_world_disco/outputs/train_x_data.npy", allow_pickle=True)
y2_train = load("/home/ubuntu/SpaceLab/new_world_disco/outputs/train_y_data.npy", allow_pickle=True)

# print(x2_val.shape)
# print(y2_val.shape)

# imports for Functional Keras model
import tensorflow
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import MaxPool1D
from tensorflow.keras.layers import GlobalMaxPool1D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Softmax
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import GaussianNoise
from tensorflow.keras.layers import RandomFlip
from tensorflow.keras.optimizers import Adam, Adamax, Nadam, SGD
#from tensorflow.keras.utils import plot_model


print("Input shapes to the Extranet model:")
X_LOCAL_SHAPE = data[0][0].shape
print("X_LOCAL_SHAPE:", X_LOCAL_SHAPE)
X_GLOBAL_SHAPE = data[0][1].shape
print("X_GLOBAL_SHAPE:", X_GLOBAL_SHAPE)
X_LOCAL_CEN_SHAPE = data[0][2].shape
print("X_LOCAL_CEN_SHAPE:", X_LOCAL_CEN_SHAPE)
X_GLOBAL_CEN_SHAPE = data[0][3].shape
print("X_GLOBAL_CEN_SHAPE:", X_GLOBAL_CEN_SHAPE)
X_STARPARS_SHAPE = data[0][4].shape
print("X_STARPARS_SHAPE:", X_STARPARS_SHAPE)
print("Label:", data[1])

FC_LOCAL_OUT_SHAPE = None
FC_GLOBAL_OUT_SHAPE = None
R_LEARN = 2e-4
# print("Adam optimizer learning rate:", R_LEARN)

# fully connected global network used for ExtranetXS
def create_fc_global_xs():
    in_layer = Input(shape=(X_GLOBAL_SHAPE[0], 2))
    # unclear what the input shape should be
    # extranet has an input of 2 and concatenate the 2 inputs along dim 1

    in_layer = RandomFlip('vertical')(in_layer)
    in_layer = GaussianNoise(0.3)(in_layer)

    fc = Conv1D(16, kernel_size=5, strides=1, padding='same')(in_layer)
    #fc = BatchNormalization()(fc)
    fc = Activation('relu')(fc)
    fc = Conv1D(16, kernel_size=5, strides=1, padding='same')(fc)
    fc = Activation('relu')(fc)
    fc = MaxPool1D(pool_size=2, strides=2)(fc)
    fc = Dropout(.25)(fc)

    fc = Conv1D(32, kernel_size=5, strides=1, padding='same')(fc)
    fc = Activation('relu')(fc)
    fc = Conv1D(32, kernel_size=5, strides=1, padding='same')(fc)
    fc = Activation('relu')(fc)
    fc = MaxPool1D(pool_size=2, strides=2)(fc)
    fc = Dropout(.25)(fc)

    fc = Conv1D(64, kernel_size=5, strides=1, padding='same')(fc)
    fc = Activation('relu')(fc)
    fc = Conv1D(64, kernel_size=5, strides=1, padding='same')(fc)
    #fc = Activation('relu')(fc)
    #fc = MaxPool1D(pool_size=2, strides=2)(fc)
    #fc = Dropout(.25)(fc)

    #fc = Conv1D(128, kernel_size=5, strides=1, padding='same')(fc)
    #fc = Activation('relu')(fc)
    #fc = Conv1D(128, kernel_size=5, strides=1, padding='same')(fc)

    out_layer = Activation('relu')(fc)
    
    # save the shape
    FC_GLOBAL_OUT_SHAPE = out_layer.shape

    model = Model(inputs=in_layer, outputs=out_layer, name='fully_connected_global_xs')
    return model, FC_GLOBAL_OUT_SHAPE

fc_global_model, FC_GLOBAL_OUT_SHAPE = create_fc_global_xs()
fc_global_model.summary()
print()
print("Output shape:", FC_GLOBAL_OUT_SHAPE)

# plot_model(fc_global_model, to_file='fc_global_model.jpg', show_shapes=True, show_layer_names=True)

# fully connected global network used for Extranet
def create_fc_local():
    in_layer = Input(shape=(X_LOCAL_SHAPE[0], 2))
    # unclear what the input shape should be
    # extranet has an input of 2 and concatenate the 2 inputs along dim 1

    in_layer = RandomFlip('vertical')(in_layer)
    in_layer = GaussianNoise(0.25)(in_layer)

    fc = Conv1D(16, kernel_size=5, strides=1, padding='same')(in_layer)
    fc = Activation('relu')(fc)
    fc = Conv1D(16, kernel_size=5, strides=1, padding='same')(fc)
    fc = Activation('relu')(fc)
    fc = MaxPool1D(pool_size=7, strides=2)(fc)
    fc = Dropout(.25)(fc)

    fc = Conv1D(32, kernel_size=5, strides=1, padding='same')(fc)
    fc = Activation('relu')(fc)
    fc = Conv1D(32, kernel_size=5, strides=1, padding='same')(fc)
    fc = Activation('relu')(fc)
    out_layer = MaxPool1D(pool_size=7, strides=2)(fc)

    # save the shape
    FC_LOCAL_OUT_SHAPE = out_layer.shape

    model = Model(inputs=in_layer, outputs=out_layer, name='fully_connected_local')
    return model, FC_LOCAL_OUT_SHAPE

fc_local_model, FC_LOCAL_OUT_SHAPE = create_fc_local()
fc_local_model.summary()
# print("\nOutput shape:", FC_LOCAL_OUT_SHAPE)

# plot_model(fc_local_model, to_file='fc_local_model.jpg', show_shapes=True, show_layer_names=True)

def create_final_layer_xs(in_shape=(177,)):
    # fully connected layers that combine local + global and does binary classification

    # input shape is flattened fc_local + fc_global + extra star parameters length
    in_layer = Input(shape=in_shape)
    #in_layer = GaussianNoise(0.3)(in_layer)
    fc = Dense(177, activation='relu')(in_layer)
    fc = Dropout(.25)(fc)
    fc = Dense(177, activation='relu')(fc)
    fc = Dropout(.25)(fc)
    fc = Dense(177, activation='relu')(fc)
    fc = Dropout(.25)(fc)
    #fc = Dense(128, activation='relu')(in_layer)
    #fc = Dropout(.25)(fc)
    #fc = Dense(128, activation='relu')(fc)
    #fc = Dropout(.25)(fc)
    #fc = Dense(64, activation='relu')(fc)
    #fc = Dropout(.25)(fc)
    #fc = Dense(64, activation='relu')(fc)
    #fc = Dropout(.25)(fc)
    #fc = Dense(32, activation='relu')(fc)
    #fc = Dropout(.25)(fc)
    #fc = Dense(32, activation='relu')(fc)
    #fc = Dropout(.25)(fc)
    #fc = Dense(16, activation='relu')(fc)
    #fc = Dropout(.25)(fc)
    #fc = Dense(16, activation='relu')(fc)
    #fc = Dropout(.25)(fc)
    out_layer = Dense(1, activation='sigmoid')(fc)

    model = Model(in_layer, out_layer, name='final_layer_classifier_xs')
    return model

final_layer_model = create_final_layer()
final_layer_model.summary()

METRICS = [
      tensorflow.keras.metrics.TruePositives(name='tp'),
      tensorflow.keras.metrics.FalsePositives(name='fp'),
      tensorflow.keras.metrics.TrueNegatives(name='tn'),
      tensorflow.keras.metrics.FalseNegatives(name='fn'), 
      tensorflow.keras.metrics.BinaryAccuracy(name='accuracy'),
      tensorflow.keras.metrics.Precision(name='precision'),
      tensorflow.keras.metrics.Recall(name='recall'),
      tensorflow.keras.metrics.AUC(name='auc'),
      tensorflow.keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
]

# plot_model(final_layer_model, to_file='final_layer.jpg', show_shapes=True, show_layer_names=True)

def ExtranetXSModelCopy(metrics):
    '''
    Extranet Model
    INPUT: 
        x_local
        x_global
        x_local_cen
        x_global_cen
        x_star
    OUTPUT:
        model used for binary classification
    '''
    print("Creating Extranet model")
    # read inputs to the model with given shapes
    x_local = Input(shape=X_LOCAL_SHAPE)
    x_local_cen = Input(shape=X_LOCAL_CEN_SHAPE)

    x_global = Input(shape=X_GLOBAL_SHAPE)
    x_global_cen = Input(shape=X_GLOBAL_CEN_SHAPE)

    x_star = Input(shape=X_STARPARS_SHAPE)

    # concatenate inputs respectively
    x_local_all = Concatenate(axis=1)([x_local, x_local_cen]) # these have to be concatenated to shape (X, 2)
    x_global_all = Concatenate(axis=1)([x_global, x_global_cen])
    
    #checking the shape after concat
    print("x_local_all.shape", x_local_all.shape)
    print("x_global_all.shape", x_global_all.shape)

    # reshape the concatenated inputs - **unsure if this reshapes correctly along axis**
    x_local_all = Reshape(target_shape=(x_local_all.shape[1]//2, 2))(x_local_all)
    x_global_all = Reshape(target_shape=(x_global_all.shape[1]//2, 2))(x_global_all)

    #checking the shape after concat
    print("\nShape after reshape")
    print("x_local_all.shape", x_local_all.shape)
    print("x_global_all.shape", x_global_all.shape)

    # call corresponding models
    fc_global, _ = create_fc_global_xs()
    fc_local, _ = create_fc_local()
    
    # get outputs from feeing inputs to the models
    out_global = fc_global(x_global_all)
    out_local = fc_local(x_local_all)

    print("\nShape after model outputs")
    print("out_global.shape", out_global.shape)
    print("out_local.shape", out_local.shape)
    
    # do global pooling
    out_global = GlobalMaxPool1D() (out_global)
    out_local = GlobalMaxPool1D()(out_local)
    print("out global shape after global pooling:", out_global.shape)
    print("out local shape after global pooling:", out_local.shape)
    # skipping global pooling bc the dimensionality reduction doesnt make sense to me now

    # flatten the outputs
    out_global = Flatten()(out_global)
    out_local = Flatten()(out_local)

    print("\nShape after flatten outputs")
    print("out_global.shape", out_global.shape)
    print("out_local.shape", out_local.shape)

    # concatenate local, global and stellar params
    out = Concatenate()([out_global, out_local, x_star])

    print("\nConcatenated out shape:", out.shape)
    print("(Should be", out_global.shape[1], "global +", out_local.shape[1], "local +", x_star.shape[1], 'stellar params)')
    # pass the flattened length to the input shape
    final_layer = create_final_layer_xs(in_shape=(out.shape[1],)) # should be 16586

    out = final_layer(out)

    print("\nShape of output after final layer:", out.shape)

    model = Model([x_local, x_global, x_local_cen, x_global_cen, x_star], out, name='ExtranetXS_model')
    opt = SGD(learning_rate=R_LEARN)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=metrics)
    return model


def make_extranetxs_model(metrics=METRICS, output_bias=None):
    if output_bias is not None:
        output_bias = tensorflow.keras.initializers.Constant(output_bias)

    model = ExtranetXSModelCopy(metrics)

    return model

# plot_model(extranet, to_file='extranet.jpg', show_shapes=True, show_layer_names=True)


# plot_model(extranetxs, to_file='extranetxs.jpg', show_shapes=True, show_layer_names=True)

NUM_EPOCHS = 1000
BATCH_SIZE = 32

early_stopping = tensorflow.keras.callbacks.EarlyStopping(monitor='val_loss', verbose=1, patience=50, mode='auto', restore_best_weights=True)

#training_batch = x2_val[:32]
#extranetxs.train_on_batch(x=[[x[0] for x in training_batch], [x[1] for x in training_batch], [x[2] for x in training_batch], [x[3] for x in training_batch], [x[4] for x in training_batch]] , y=y2_val[:32])

scaler = StandardScaler()
x_features = [scaler.fit_transform(np.array([x[0] for x in x2_train])), 
              scaler.fit_transform(np.array([x[1] for x in x2_train])), 
              scaler.fit_transform(np.array([x[2] for x in x2_train])), 
              scaler.fit_transform(np.array([x[3] for x in x2_train])), 
              scaler.fit_transform(np.array([x[4] for x in x2_train]))]

print(x_features.shape)
val_features = [
    np.array(scaler.fit_transform([x[0] for x in x2_val])), 
    np.array(scaler.fit_transform([x[1] for x in x2_val])), 
    np.array(scaler.fit_transform([x[2] for x in x2_val])), 
    np.array(scaler.fit_transform([x[3] for x in x2_val])), 
    np.array(scaler.fit_transform([x[4] for x in x2_val]))
]
test_features = [scaler.fit_transform(np.array([x[0] for x in x2_test])), 
              scaler.fit_transform(np.array([x[1] for x in x2_test])), 
              scaler.fit_transform(np.array([x[2] for x in x2_test])), 
              scaler.fit_transform(np.array([x[3] for x in x2_test])), 
              scaler.fit_transform(np.array([x[4] for x in x2_test]))]


neg, pos = np.bincount(y)

total = pos+neg
weight_for_0 = (1/neg)*(total/2.0)
weight_for_1 = (1/pos)*(total/2.0)


class_weight = {0:weight_for_0, 1:weight_for_1}

weighted_model = make_extranetxs_model()

weighted_history = weighted_model.fit(
    x_features,
    y2_train,
    batch_size=BATCH_SIZE,
    epochs=NUM_EPOCHS,
    validation_data=(val_features, y2_val),
    callbacks=[early_stopping],
    # The class weights go here
    class_weight=class_weight)

import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import RocCurveDisplay

########################################
####### CALCULATE STATISTICS ###########
########################################

### setup screen output
print("\nCALCULATING METRICS...\n")

### calculate average precision & precision-recall curves
AP = average_precision_score(test_gt, test_output, average=None)
# print("   average precision = {0:0.4f}\n".format(AP))

### calculate precision-recall curve
P, R, _ = precision_recall_curve(test_gt, test_output)

tpr, fpr, _ = roc_curve(test_gt, test_output)

### calculate confusion matrix based on different thresholds
thresh = [0.5, 0.6, 0.7, 0.8, 0.9]
prec_thresh, recall_thresh = np.zeros(len(thresh)), np.zeros(len(thresh))

for n, nval in enumerate(thresh):
    pred_byte = np.zeros(len(test_output))
    for i, val in enumerate(test_output):
        if val > nval:
            pred_byte[i] = 1.0
        else:
            pred_byte[i] = 0.0
    prec_thresh[n] = precision_score(test_gt, pred_byte, zero_division=1)
    recall_thresh[n] = recall_score(test_gt, pred_byte, zero_division=1)
    print("   thresh = {0:0.2f}, precision = {1:0.2f}, recall = {2:0.2f}".format(thresh[n], prec_thresh[n], recall_thresh[n]))
    tn, fp, fn, tp = confusion_matrix(test_gt, pred_byte).ravel()
    print("      TN = {0:0}, FP = {1:0}, FN = {2:0}, TP = {3:0}".format(tn, fp, fn, tp))


import matplotlib.pyplot as plt
PrecisionRecallDisplay.from_predictions(test_gt, test_output)

plt.savefig('/home/ubuntu/SpaceLab/new_world_disco/code/outputs/precision_recall_curve.png', dpi=300, bbox_inches='tight')

RocCurveDisplay.from_predictions(test_gt, test_output)
plt.savefig('/home/ubuntu/SpaceLab/new_world_disco/code/outputs/roc_curve.png', dpi=300, bbox_inches='tight')
