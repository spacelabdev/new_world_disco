import os
import numpy as np
import glob as glob
import torch
from torch.utils.data import Dataset
#import pydot
#import graphviz

filepath = './data'

# try the default kepler data loader in exonet.py
# reference: https://gitlab.com/frontierdevelopmentlab/exoplanets/exonet-pytorch/-/blob/master/exonet.py
class KeplerDataLoader(Dataset):
    
    '''
    
    PURPOSE: DATA LOADER FOR KERPLER LIGHT CURVES
    INPUT: PATH TO DIRECTOR WITH LIGHT CURVES + INFO FILES
    OUTPUT: LOCAL + GLOBAL VIEWS, LABELS
    
    '''

    def __init__(self, filepath):

        ### list of global, local, and info files (assumes certain names of files)
        self.flist_global = np.sort(glob.glob(os.path.join(filepath, '*global.npy')))
        self.flist_local = np.sort(glob.glob(os.path.join(filepath, '*local.npy')))
        self.flist_info = np.sort(glob.glob(os.path.join(filepath, '*info.npy')))
        
        ### list of whitened centroid files
        self.flist_global_cen = np.sort(glob.glob(os.path.join(filepath, '*global_cen_w.npy')))
        self.flist_local_cen = np.sort(glob.glob(os.path.join(filepath, '*local_cen_w.npy')))
        
        ### ids = {TIC}_{TCE}
        self.ids = np.sort([(x.split('/')[-1]).split('_')[1] + '_' + (x.split('/')[-1]).split('_')[2] for x in self.flist_global])

    def __len__(self):

        return self.ids.shape[0]

    def __getitem__(self, idx):

        ### grab local and global views
        data_global = np.load(self.flist_global[idx])
        data_local = np.load(self.flist_local[idx])

        ### grab centroid views
        data_global_cen = np.load(self.flist_global_cen[idx])
        data_local_cen = np.load(self.flist_local_cen[idx])
        
        ### info file contains: [0]kic, [1]tce, [2]period, [3]epoch, [4]duration, [5]label)
        data_info = np.load(self.flist_info[idx])
        
        return (data_local, data_global, data_local_cen, data_global_cen, data_info[6:]), data_info[5]


### grab data using data loader
kepler_train_data = KeplerDataLoader(filepath=os.path.join(filepath, 'train'))
kepler_val_data = KeplerDataLoader(filepath=os.path.join(filepath, 'test'))

# look through some of the data
count = 0
print("len(kepler_val_data)", len(kepler_val_data))
for data in kepler_val_data:
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
x_val = []
y_val = []

curr = 0
total = len(kepler_val_data)
print("Loading validation data:")
for x_data, y_data in kepler_val_data:
    print(curr, "/", total)
    if not isinstance(y_data, np.float64):
        print(f'ERROR: {y_data}')
    curr += 1
    x_val.append(x_data)
    y_val.append(y_data)

x_train = []
y_train = []
print("Loading training data:")
curr = 0
total = len(kepler_train_data)
for x_data, y_data in kepler_train_data:
    print(curr, "/", total)
    if not isinstance(y_data, np.float64):
        print(f'ERROR: {y_data}')
    curr += 1
    x_train.append(x_data)
    y_train.append(y_data)


x_val = np.asarray(x_val, dtype=object)
y_val = np.asarray(y_val, dtype=np.float32)
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
save("./outputs/val_x_data.npy", x_val)
save("./outputs/val_y_data.npy", y_val)
save("./outputs/train_x_data.npy", x_train)
save("./outputs/train_y_data.npy", y_train)

from numpy import load

x2_val = load("./outputs/val_x_data.npy", allow_pickle=True)
y2_val = load("./outputs/val_y_data.npy", allow_pickle=True)
x2_train = load("./outputs/train_x_data.npy", allow_pickle=True)
y2_train = load("./outputs/train_y_data.npy", allow_pickle=True)


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
from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.utils import plot_model

# Define static input shapes
data = kepler_val_data[0]

# print("Input shapes to the Extranet model:")
X_LOCAL_SHAPE = data[0][0].shape
# print("X_LOCAL_SHAPE:", X_LOCAL_SHAPE)
X_GLOBAL_SHAPE = data[0][1].shape
# print("X_GLOBAL_SHAPE:", X_GLOBAL_SHAPE)
X_LOCAL_CEN_SHAPE = data[0][2].shape
# print("X_LOCAL_CEN_SHAPE:", X_LOCAL_CEN_SHAPE)
X_GLOBAL_CEN_SHAPE = data[0][3].shape
# print("X_GLOBAL_CEN_SHAPE:", X_GLOBAL_CEN_SHAPE)
X_STARPARS_SHAPE = data[0][4].shape
# print("X_STARPARS_SHAPE:", X_STARPARS_SHAPE)
# print("Label:", data[1])

FC_LOCAL_OUT_SHAPE = None
FC_GLOBAL_OUT_SHAPE = None
R_LEARN = 1e-5
# print("Adam optimizer learning rate:", R_LEARN)

# fully connected global network used for Extranet
def create_fc_global():
    in_layer = Input(shape=(X_GLOBAL_SHAPE[0], 2))
    # unclear what the input shape should be
    # extranet has an input of 2 and concatenate the 2 inputs along dim 1

    fc = Conv1D(16, kernel_size=5, strides=1, padding='same')(in_layer)
    fc = Activation('relu')(fc)
    fc = Conv1D(16, kernel_size=5, strides=1, padding='same')(fc)
    fc = Activation('relu')(fc)
    fc = MaxPool1D(pool_size=5, strides=2)(fc)

    fc = Conv1D(32, kernel_size=5, strides=1, padding='same')(fc)
    fc = Activation('relu')(fc)
    fc = Conv1D(32, kernel_size=5, strides=1, padding='same')(fc)
    fc = Activation('relu')(fc)
    fc = MaxPool1D(pool_size=5, strides=2)(fc)

    fc = Conv1D(64, kernel_size=5, strides=1, padding='same')(fc)
    fc = Activation('relu')(fc)
    fc = Conv1D(64, kernel_size=5, strides=1, padding='same')(fc)
    fc = Activation('relu')(fc)
    fc = MaxPool1D(pool_size=5, strides=2)(fc)

    fc = Conv1D(128, kernel_size=5, strides=1, padding='same')(fc)
    fc = Activation('relu')(fc)
    fc = Conv1D(128, kernel_size=5, strides=1, padding='same')(fc)
    fc = Activation('relu')(fc)
    fc = MaxPool1D(pool_size=5, strides=2)(fc)

    fc = Conv1D(256, kernel_size=5, strides=1, padding='same')(fc)
    fc = Activation('relu')(fc)
    fc = Conv1D(256, kernel_size=5, strides=1, padding='same')(fc)
    fc = Activation('relu')(fc)
    out_layer = MaxPool1D(pool_size=5, strides=2)(fc)
    
    # save the shape
    FC_GLOBAL_OUT_SHAPE = out_layer.shape

    model = Model(inputs=in_layer, outputs=out_layer, name='fully_connected_global')
    return model, FC_GLOBAL_OUT_SHAPE

fc_global_model, FC_GLOBAL_OUT_SHAPE = create_fc_global()
fc_global_model.summary()
print()
print("Output shape:", FC_GLOBAL_OUT_SHAPE)

# plot_model(fc_global_model, to_file='fc_global_model.jpg', show_shapes=True, show_layer_names=True)

# fully connected global network used for Extranet
def create_fc_local():
    in_layer = Input(shape=(X_LOCAL_SHAPE[0], 2))
    # unclear what the input shape should be
    # extranet has an input of 2 and concatenate the 2 inputs along dim 1

    fc = Conv1D(16, kernel_size=5, strides=1, padding='same')(in_layer)
    fc = Activation('relu')(fc)
    fc = Conv1D(16, kernel_size=5, strides=1, padding='same')(fc)
    fc = Activation('relu')(fc)
    fc = MaxPool1D(pool_size=7, strides=2)(fc)

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

def create_final_layer(in_shape=(16586,)):
    # fully connected layers that combine local + global and does binary classification

    # input shape is flattened fc_local + fc_global + extra star parameters length
    '''
    input_length =  FC_LOCAL_OUT_SHAPE[1]  * FC_LOCAL_OUT_SHAPE[2]
    input_length += FC_GLOBAL_OUT_SHAPE[1] * FC_GLOBAL_OUT_SHAPE[2]
    input_length += X_STARPARS_SHAPE[0]
    print("Input length:", input_length)
    in_layer = Input(shape=(input_length,))
    '''
    in_layer = Input(shape=in_shape)
    fc = Dense(512, activation='relu')(in_layer)
    fc = Dense(512, activation='relu')(fc)
    fc = Dense(512, activation='relu')(fc)
    out_layer = Dense(1, activation='sigmoid')(fc)

    model = Model(in_layer, out_layer, name='final_layer_classifier')
    return model

final_layer_model = create_final_layer()
final_layer_model.summary()

# plot_model(final_layer_model, to_file='final_layer.jpg', show_shapes=True, show_layer_names=True)

def ExtranetModelCopy():
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
#    print("x_local_all.shape", x_local_all.shape)
#    print("x_global_all.shape", x_global_all.shape)

    # reshape the concatenated inputs - **unsure if this reshapes correctly along axis**
    x_local_all = Reshape(target_shape=(x_local_all.shape[1]//2, 2))(x_local_all)
    x_global_all = Reshape(target_shape=(x_global_all.shape[1]//2, 2))(x_global_all)

    #checking the shape after concat
#    print("\nShape after reshape")
#    print("x_local_all.shape", x_local_all.shape)
#    print("x_global_all.shape", x_global_all.shape)

    # call corresponding models
    fc_global, _ = create_fc_global()
    fc_local, _ = create_fc_local()
    
    # get outputs from feeing inputs to the models
    out_global = fc_global(x_global_all)
    out_local = fc_local(x_local_all)

#    print("\nShape after model outputs")
#    print("out_global.shape", out_global.shape)
#    print("out_local.shape", out_local.shape)
    
    # do global pooling
    '''
    out_global = GlobalMaxPool1D() (out_global)
    out_local = GlobalMaxPool1D()(out_local)
    print("local shape after global pooling:", out_global.shape)
    '''
    # skipping global pooling bc the dimensionality reduction doesnt make sense to me now

    # flatten the outputs
    out_global = Flatten()(out_global)
    out_local = Flatten()(out_local)

#    print("\nShape after flatten outputs")
#    print("out_global.shape", out_global.shape)
#    print("out_local.shape", out_local.shape)

    # concatenate local, global and stellar params
    out = Concatenate()([out_global, out_local, x_star])

#    print("\nConcatenated out shape:", out.shape)
#    print("(Should be", out_global.shape[1], "global +", out_local.shape[1], "local +", x_star.shape[1], 'stellar params)')
    # pass the flattened length to the input shape
    final_layer = create_final_layer(in_shape=(out.shape[1],)) # should be 16586

    out = final_layer(out)

#    print("\nShape of output after final layer:", out.shape)

    model = Model([x_local, x_global, x_local_cen, x_global_cen, x_star], out, name='Extranet_model')
    opt = Adam(learning_rate=R_LEARN)
    model.compile(loss='binary_crossentropy', optimizer=opt)
    return model

extranet = ExtranetModelCopy()

extranet.summary()

# plot_model(extranet, to_file='extranet.jpg', show_shapes=True, show_layer_names=True)

# fully connected global network used for ExtranetXS
def create_fc_global():
    in_layer = Input(shape=(X_GLOBAL_SHAPE[0], 2))
    # unclear what the input shape should be
    # extranet has an input of 2 and concatenate the 2 inputs along dim 1

    fc = Conv1D(16, kernel_size=5, strides=1, padding='same')(in_layer)
    fc = Activation('relu')(fc)
    fc = MaxPool1D(pool_size=2, strides=2)(fc)

    fc = Conv1D(16, kernel_size=5, strides=1, padding='same')(fc)
    fc = Activation('relu')(fc)
    fc = MaxPool1D(pool_size=2, strides=2)(fc)

    fc = Conv1D(32, kernel_size=5, strides=1, padding='same')(fc)
    out_layer = Activation('relu')(fc)
    
    # save the shape
    FC_GLOBAL_OUT_SHAPE = out_layer.shape

    model = Model(inputs=in_layer, outputs=out_layer, name='fully_connected_global_xs')
    return model, FC_GLOBAL_OUT_SHAPE


def create_final_layer(in_shape=(58,)):
    # fully connected layers that combine local + global and does binary classification

    # input shape is flattened fc_local + fc_global + extra star parameters length
    in_layer = Input(shape=in_shape)
    fc = Dense(58, activation='relu')(in_layer)
    out_layer = Dense(1, activation='sigmoid')(fc)

    model = Model(in_layer, out_layer, name='final_layer_classifier_xs')
    return model

def ExtranetXSModelCopy():
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
#    print("x_local_all.shape", x_local_all.shape)
#    print("x_global_all.shape", x_global_all.shape)

    # reshape the concatenated inputs - **unsure if this reshapes correctly along axis**
    x_local_all = Reshape(target_shape=(x_local_all.shape[1]//2, 2))(x_local_all)
    x_global_all = Reshape(target_shape=(x_global_all.shape[1]//2, 2))(x_global_all)

    #checking the shape after concat
#    print("\nShape after reshape")
#    print("x_local_all.shape", x_local_all.shape)
#    print("x_global_all.shape", x_global_all.shape)

    # call corresponding models
    fc_global, _ = create_fc_global()
    fc_local, _ = create_fc_local()
    
    # get outputs from feeing inputs to the models
    out_global = fc_global(x_global_all)
    out_local = fc_local(x_local_all)

#    print("\nShape after model outputs")
#    print("out_global.shape", out_global.shape)
#    print("out_local.shape", out_local.shape)
    
    # do global pooling
    out_global = GlobalMaxPool1D() (out_global)
    out_local = GlobalMaxPool1D()(out_local)
#    print("out global shape after global pooling:", out_global.shape)
#    print("out local shape after global pooling:", out_local.shape)
    # skipping global pooling bc the dimensionality reduction doesnt make sense to me now

    # flatten the outputs
    out_global = Flatten()(out_global)
    out_local = Flatten()(out_local)

#    print("\nShape after flatten outputs")
#    print("out_global.shape", out_global.shape)
#    print("out_local.shape", out_local.shape)

    # concatenate local, global and stellar params
    out = Concatenate()([out_global, out_local, x_star])

#    print("\nConcatenated out shape:", out.shape)
#    print("(Should be", out_global.shape[1], "global +", out_local.shape[1], "local +", x_star.shape[1], 'stellar params)')
    # pass the flattened length to the input shape
    final_layer = create_final_layer(in_shape=(out.shape[1],)) # should be 16586

    out = final_layer(out)

#    print("\nShape of output after final layer:", out.shape)

    model = Model([x_local, x_global, x_local_cen, x_global_cen, x_star], out, name='ExtranetXS_model')
    opt = Adam(learning_rate=R_LEARN)
    model.compile(loss='binary_crossentropy', optimizer=opt)
    return model

extranetxs = ExtranetXSModelCopy()
extranetxs.summary()

# plot_model(extranetxs, to_file='extranetxs.jpg', show_shapes=True, show_layer_names=True)

NUM_EPOCHS = 100
BATCH_SIZE = 32

#training_batch = x2_val[:32]
#extranetxs.train_on_batch(x=[[x[0] for x in training_batch], [x[1] for x in training_batch], [x[2] for x in training_batch], [x[3] for x in training_batch], [x[4] for x in training_batch]] , y=y2_val[:32])

x_features = [np.array([x[0] for x in x2_train]), 
              np.array([x[1] for x in x2_train]), 
              np.array([x[2] for x in x2_train]), 
              np.array([x[3] for x in x2_train]), 
              np.array([x[4] for x in x2_train])]
# print(x_features.shape)
history = extranetxs.fit(x=x_features, y=y2_train, batch_size=BATCH_SIZE, epochs=100)

#print(history)
val_features = [np.array([x[0] for x in x2_val]), 
                np.array([x[1] for x in x2_val]), 
                np.array([x[2] for x in x2_val]), 
                np.array([x[3] for x in x2_val]), 
                np.array([x[4] for x in x2_val])]
val_output = extranetxs.predict(val_features)

val_gt = y2_val.astype(None).ravel()
val_output = val_output.astype(None).ravel()
save("./outputs/train_y_output.npy", val_output)

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

########################################
####### CALCULATE STATISTICS ###########
########################################

### setup screen output
print("\nCALCULATING METRICS...\n")

### calculate average precision & precision-recall curves
AP = average_precision_score(val_gt, val_output, average=None)
# print("   average precision = {0:0.4f}\n".format(AP))

### calculate precision-recall curve
P, R, _ = precision_recall_curve(val_gt, val_output)

### calculate confusion matrix based on different thresholds
thresh = [0.5, 0.6, 0.7, 0.8, 0.9]
prec_thresh, recall_thresh = np.zeros(len(thresh)), np.zeros(len(thresh))
for n, nval in enumerate(thresh):
    pred_byte = np.zeros(len(val_output))
    for i, val in enumerate(val_output):
        if val > nval:
            pred_byte[i] = 1.0
        else:
            pred_byte[i] = 0.0
    prec_thresh[n] = precision_score(val_gt, pred_byte)
    recall_thresh[n] = recall_score(val_gt, pred_byte)
    print("   thresh = {0:0.2f}, precision = {1:0.2f}, recall = {2:0.2f}".format(thresh[n], prec_thresh[n], recall_thresh[n]))
    tn, fp, fn, tp = confusion_matrix(val_gt, pred_byte).ravel()
    print("      TN = {0:0}, FP = {1:0}, FN = {2:0}, TP = {3:0}".format(tn, fp, fn, tp))

