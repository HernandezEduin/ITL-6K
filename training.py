import numpy as np
import csv
import cv2
import glob
import os
# import pandas as pd
# from scipy.io import savemat
# from keras import Model
import tensorflow as tf
# from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint, TerminateOnNaN
from tensorflow.keras.layers import Reshape
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import LearningRateScheduler
from sklearn.ensemble import VotingRegressor
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import multiprocessing #parallel loading
import concurrent.futures
import random
import time
import preprocess
import pandas as pd
from tensorflow.keras.models import load_model
import tikzplotlib
from sklearn.metrics import precision_score, recall_score,f1_score, accuracy_score, confusion_matrix
import seaborn as sns
from tensorflow.keras.callbacks import Callback
# import dask.dataframe as dd
from preprocess import *
import seaborn as sns
#------------------------------------------------------------------------------
'Deep Learning Models'
relu = lambda x: tf.math.maximum(x,0.0)
def scheduler(epoch, lr):
    if epoch%500 == 0:
        return lr*tf.math.exp(-0.1)
    else:
        return lr
    
class SchedulerandTrackerCallback(Callback):
    def __init__(self,scheduler):
        self.scheduler = scheduler
        self.epoch_lr = []
        self.epoch_loss = []
        
    def on_epoch_begin(self,epoch,logs = None):
        current_lr = self.model.optimizer.learning_rate.numpy()
        new_lr = self.scheduler(epoch,current_lr)
        tf.keras.backend.set_value(self.model.optimizer.learning_rate, new_lr)
    def on_epoch_end(self,epoch,logs = None):
        current_lr = self.model.optimizer.learning_rate.numpy()
        loss = logs.get('loss')
        self.epoch_lr.append(current_lr)
        self.epoch_loss.append(loss)
        


def model2(input_shape, output_shape):
    model = Sequential()
    
    model.add(layers.Conv2D(16,kernel_size=3, activation = relu,input_shape = input_shape,))
    model.add(layers.MaxPooling2D(pool_size =2))
    model.add(layers.Dense(32, activation = 'relu'))
    model.add(layers.Dropout(0.25))
    model.add(layers.Reshape((16,-1)))
              
    model.add(layers.Bidirectional(layers.LSTM(16,input_shape = input_shape, return_sequences=True)))
    model.add(layers.Dense(64, activation = 'sigmoid'))
    model.add(layers.Dropout(0.25))
 
    model.add(layers.Dense(256, activation = 'sigmoid'))
    model.add(layers.Dropout(0.25))
    model.add(layers.Dense(1024, activation = 'sigmoid'))
    model.add(layers.Dropout(0.25))

    
    model.add(layers.Flatten())
    model.add(layers.Dense(output_shape[0]*output_shape[1], activation = 'sigmoid'))
    model.add(layers.Reshape(output_shape))
    return model


model_dict = {'2': model2}

#------------------------------------------------------------------------------
start_time = time.time()
'Reading Functions'
'Offline learning / batch training'
'Prev version without chunk'
def read_voltage(filepath='./Image EIT/voltage.csv'): #  Reading voltage .csv file 16*2048(raw data from FPGA)
    assert os.path.isfile(filepath), "Error!" + filepath + ' does not exist!'
    
    data = []
    inputdata = []
    a = 1
    counter = 1 #will be the number of data
    with open(filepath,"r") as f:
        reader = csv.reader(f)
        row_count = sum(1 for row in reader)
        
        
    with open(filepath,"r") as f:
        reader = csv.reader(f)
        
        # reader1 = csv.reader(f)
        for i, line in enumerate(reader):
            # print(i)
            if i >= a and i <= (a+15): # Only append 2nd row to 17th row for every datapoint
                data.append(line)
            elif i!= 0:
                data = np.array(data)
                'you can exchange this line for other purposes'
                # savemat('matrix' + str(counter) + '.mat',mdict = {'arr':data}) 
                inputdata.append(data)
                data = []
                a = a + 17
                counter += 1
                # print(counter)
            if i == (row_count-1): #to save the last entry of the mat file
                # savemat('matrix' + str( counter) + '.mat',mdict = {'arr':data})
                inputdata.append(data)
    return inputdata

'Prev version without chunk'
def load_image(path,length):
    assert os.path.isdir(path), "Error!" + path + ' does not exist!'
    
    images = []
    for i in length:
        img = cv2.imread(path +'label'+str(i)+'.jpg',0)
        images.append(img)
    return images

'Use random indices'
def read_voltage(index, chunk, filepath = './Image EIT/voltage.csv',chunk_size = 100):
    assert os.path.isfile(filepath), "Error!" + filepath + ' does not exist!'
    # print(filepath)
    data = []
    inputdata = []
    random_index = 0
    # a = 1
    counter = 0
    a = counter*17+1
    'Chunk of data goes from 0~99, 100~199, and so on.'
    # counter = (chunk-1)*100 #will be the number of data
    
    with open(filepath,"r") as f:
        reader = csv.reader(f)
        row_count = sum(1 for row in reader)
        # print(row_count)
        
    with open(filepath,"r") as f:
        reader = csv.reader(f)
        print(f'Reading csv files of chunk {chunk} of {filepath}.')
        
        for i, line in enumerate(reader):
            # print(i,counter,chunk)
            a = counter*17+1
            # print(i,random_index)
            if counter in index:
                # print(counter)
                if i >= a and i <= (a+15): # Only append 2nd row to 17th row for every datapoint
                    # print('in1')
                    data.append(line)
                elif i!= 0 and i%17 == 0: #i != 0
                    # print('in2')
                    data = np.array(data)
                    # 'PCA method'
                    # data = np.array(data).astype(float).astype(int) #Insert sus preprocessing
                    # data = pca_transform(data)  #Insert sus preprocessing
                    # print(type(data))
                    'you can exchange this line for other purposes'
                    inputdata.append(data)
                    data = []
                    a = a + 17
                    counter += 1
                    random_index+=1
                    
                if i == (row_count-1): #to save the last entry of the mat file
                    # 'PCA method'
                    # data = np.array(data).astype(float).astype(int) #Insert sus preprocessing
                    # data = pca_transform(data)  #Insert sus preprocessing
                    inputdata.append(data)
                    
                # if(i == (chunk*17*chunk_size)):
                #     break;
                    
            else:
                if i!=0 and i%17 == 0:
                    counter +=1
                    
                    continue
                
            # if random_index == len(index):
            #     break
    print(counter, random_index)        
    return inputdata,index

def load_image(path,length,index):
    assert os.path.isdir(path), "Error!" + path + ' does not exist!'
    print('in')
    images = []
    print('Reading images of ' + path)
    for i in index:
        # print(i)
        img = cv2.imread(path +'label'+str(i)+'.jpg',0)
        images.append(img)
    return images

class MetricsCallback(Callback):
    def __init__(self,test_data,epochs_to_save):
        super(MetricsCallback,self).__init__()
        self.test_data = test_data
        self.epochs_to_save = epochs_to_save
        self.metrics_history = []
        
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) in self.epochs_to_save:
            X_test, y_test = self.test_data
            y_pred = (self.model.predict(X_test) >= 0.5).astype(np.int32)

            # Flatten the images if needed
            y_test_flat = y_test.flatten()
            y_pred_flat = y_pred.flatten()

            # Compute metrics
            accuracy = accuracy_score(y_test_flat, y_pred_flat)
            precision = precision_score(y_test_flat, y_pred_flat)
            recall = recall_score(y_test_flat, y_pred_flat)
            f1 = f1_score(y_test_flat, y_pred_flat)
            cm = confusion_matrix(y_test_flat, y_pred_flat)

            # Save metrics
            metrics = {
                'epoch': epoch + 1,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1# Convert numpy array to list for JSON serialization
            }
            
            self.metrics_history.append(metrics)
            self.model.save('model'+str(epoch+1)+'epochs.h')
            
#------------------------------------plt.imshow--------------------------------
# folder = "C:\\Users\lab716a\Documents\School\EIT\CameraTrain\EIT image label\Labelled\Vector"
model_name = '2' #Use 2
epochs = 1000  #Use 5000
learning_rate = 0.001 #Use 0.001
batch_size = 64 #Use 64
csv_folder = "./Oct 2023/csv/"
csv_folder1 = "./March 31 2024/csv/"
folder = "./Oct 2023/"
folder1 = "./March 31 2024/"
chunk = 1
load = True

#------------------------------------------------------------------------------
'Load voltage data from OCT 2023(csv inputs)'

input_filename = [csv_folder +'voltage first.csv',csv_folder+'voltage second.csv',csv_folder+
                  'voltage third.csv',csv_folder+'voltage fourth.csv',csv_folder+'voltage fifth.csv',csv_folder+
                  'voltage sixth.csv',csv_folder+'voltage seventh.csv',csv_folder+'voltage eighth.csv']
                   # ,csv_folder1+ '1 circle.csv' , csv_folder1 +'2 circles.csv' ,csv_folder1 +'3 circles.csv' , csv_folder1 +'4 circles.csv']
image_filename = [folder + 'Labelled 1022 first recording', folder +'Labelled 1022 second recording', folder +'Labelled 1022 third recording', 
               folder +'Labelled 1022 fourth recording',folder + 'Labelled 1024 fifth recording',folder + 'Labelled 1024 sixth recording', 
               folder +'Labelled 1024 seventh recording',folder +'Labelled 1025 eighth recording']
               # ,folder1 + 'Labelled march 31 1 circle', folder1 + 'Labelled march 31 2 circles' , folder1+'labelled march 31 3 circles',folder1 +'Labelled march 31 4 circles']



inputdata = []
# validation_dataset = tf.data.Dataset.from_tensors((tf.zeros((0,)), tf.zeros((0,))))
validation_dataset = None
x_testappend = []
y_testappend = []
loss = []
validdata = []

'chunk ranges from 1 to 2'
for chunk in range(1,2):
    'Initialize inputdata and images variable for every chunk'
    testdata = []
    inputdata = []
    shape = (0,128,128)
    images = np.empty(shape)
    valid_images = np.empty(shape)
    random.seed(42)
    index = random.sample(range(0,723),723)
    index = sorted(index)
    
    # index = np.arange(723)
    test_indices = []
    
    for i in range(723):
        if i not in index:
            test_indices.append(i)

    'This is the original'
    for i,filename in enumerate(input_filename):
        print('Chunk number ' + str(chunk))
        'Voltage input'
        data,index = read_voltage(index, chunk, filename)
        # data = np.array(data)[:,::2,:]  #skipping 1 pin/ 3 pins
        inputdata += data;
        '====================================================================='
        'Remove this if un-needed'
        # valid_data, valid_index = read_voltage(test_indices,chunk,filename)
        # validdata += valid_data
        '====================================================================='
        
        'Image'
        numberofdata = len(glob.glob('Labelled 1022 first recording/Vector/'+'*jpg'))
        imagesofonefile = np.array(load_image(image_filename[i] +'/Vector/',numberofdata,index)).astype(int)
        images = np.vstack((images,imagesofonefile))
        


    'This is the original'
    'Pre-processing inputdata after every chunk'
    inputdata = np.array(inputdata).astype(float).astype(int)
    
    'Skipping one pin/3 pins' # slice 2 for 1, slice 4 for 3
    # inputdata = inputdata[:,::4,:]
    
    inputdata = inputdata[...,np.newaxis] #adding new axis at the last dimension, for conv2d
    
    # validdata = np.array(validdata).astype(float).astype(int)
    # validdata = validdata[...,np.newaxis]

    'Normalization'
    mean = inputdata.mean(axis=(0,2),keepdims = True)
    var = inputdata.std(axis=(0,2),keepdims = True)

    inputdata = (inputdata-mean)/var
    'PREPROCESSING FILTERS'
    'Method 1'
    # inputdata = bandpass_filter(inputdata) # Filters outliers (saturated data)
    'Method 2'
    # inputdata = pca_transform(inputdata) # Reduce number of features
    'Method 3'
    # inputdata = savitzky(inputdata) # Smoothing
    'Method 4'
    # inputdata = wavelet(inputdata) # Denoising
    'Method 5'
    # inputdata = wavelet(inputdata)
    # inputdata = pca_transform(inputdata)
    'Method 6'
    # inputdata = savitzky(inputdata)
    # inputdata = pca_transform(inputdata)
    
    'Preprocessing image after every chunk'
    images = 1*(images > 100)

    'This is the original'
    x_train ,x_test, y_train, y_test = train_test_split(inputdata, images,test_size = 0.2,random_state=42) # add Test size

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)
    

    
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
    if load == False: # Training model, not loading model
        'Intialize model'
        if chunk == 1:
            'Setting learningrate scheduler and record loss and its learning rate'
            # callback = LearningRateScheduler(scheduler)
            callback = SchedulerandTrackerCallback(scheduler)
            model = model_dict[model_name](inputdata.shape[1:], output_shape=images.shape[1:])
            opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
            model.build(inputdata.shape)
            '------------------------------------------------------'
            'Try out differrent loses'
            # model.compile(loss = "mse", optimizer = opt)
            model.compile(loss='binary_crossentropy', optimizer = opt,metrics = ['accuracy']) #If it is binary maybe this will work better
           
        'With callback epochs'
        # epochs_to_save = [200, 400, 600, 800, 1000]
        # metrics_callback = MetricsCallback(test_data=(x_test, y_test), epochs_to_save=epochs_to_save)
        # callbacks = [metrics_callback,callback]
        # history = model.fit(train_dataset, epochs=epochs, validation_data= test_dataset,callbacks = callbacks,shuffle = True)
           
        
        history = model.fit(train_dataset, epochs=epochs, validation_data= test_dataset,callbacks = callback,shuffle = True)
        loss.append (history.history['loss'])
        'save trained model here'
        # model.save('model500data.h')
        # model.save('modelwaveletpca.h5')
        # model.save('modeltrain_on_4circles.h')
    else:   # Loading model
        # model = load_model('modelwavelet.h') # CHANGE THIS IF YOU ARE LOADING MODEL TO APPROPRIATE DATASET
        model = load_model('modeloriginal.h')
        # model = load_model('modelsavitzky.h')
        # model = load_model('modelsavitzkypca.h')
        # model = load_model('modelwaveletpcapca.h')
        
'Heatmap of loss with each epoch and learning rate'

plt.figure()
plt.plot(history.history['loss'], label = 'Training loss')
plt.plot(history.history['val_loss'], label = 'Validation loss')
plt.legend()  # Add legend elements
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')

end_time = time.time()
execution_time = end_time - start_time
print("Execution time:",execution_time)

'============================================================================='
'Plotting heatmap of learning rate and losses'
# Get learning rates and losses from the callback
learning_rates = callback.epoch_lr
losses = callback.epoch_loss

# Convert lists to 2D arrays (single row)
lr_data = np.array([learning_rates])  # Learning rates
loss_data = np.array([losses])       # Losses

# Plot Learning Rate Heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(lr_data, annot=True, fmt=".4f", cmap="viridis", cbar=True)
plt.title("Learning Rate Heatmap")
plt.xlabel("Iteration")
plt.ylabel("Metric")
plt.show()

# Plot Loss Heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(loss_data, annot=True, fmt=".4f", cmap="coolwarm", cbar=True)
plt.title("Loss Heatmap")
plt.xlabel("Iteration")
plt.ylabel("Metric")
plt.show()

# Combine learning rates and losses into a single array
combined_data = np.array([learning_rates, losses])

# Plot combined heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(combined_data.T, annot=True, fmt=".4f", cmap="coolwarm", cbar=True)
plt.title("Learning Rate and Loss Relationship Heatmap")
plt.xlabel("Metric")
plt.ylabel("Iteration")
plt.show()
'============================================================================='


numberofdata = len(glob.glob(folder + 'Labelled 1022 first recording/Vector/'+'*jpg'))


savitzky_data = savitzky(inputdata)
plt.figure()
plt.plot(inputdata[1,0,::5,0])
# plt.plot(savitzky_data[1,1,::3,0])
plt.title('Normalized Voltage of Pin 1')

plt.figure()

plt.title('Normalized Voltage with Savitzky Filter of Pin 1')

plt.figure()
plt.plot(inputdata[1,1,:,0])
plt.title('Normalized Voltage of Pin 2')
plt.figure()
plt.plot(inputdata[1,2,:,0])
plt.title('Normalized Voltage of Pin 3')
plt.figure()
plt.plot(inputdata[1,3,:,0])
plt.title('Normalized Voltage of Pin 4')
plt.figure()
plt.plot(inputdata[1,4,:,0])
plt.title('Normalized Voltage of Pin 5')
plt.figure()
plt.plot(inputdata[1,5,:,0])
plt.title('Normalized Voltage of Pin 6')
plt.figure()
plt.plot(inputdata[1,6,:,0])
plt.title('Normalized Voltage of Pin 7')
plt.figure()
plt.plot(inputdata[1,7,:,0])
plt.title('Normalized Voltage of Pin 8')
plt.figure()
plt.plot(inputdata[1,8,:,0])
plt.title('Normalized Voltage of Pin 9')
plt.figure()
plt.plot(inputdata[1,9,:,0])
plt.title('Normalized Voltage of Pin 10')
plt.figure()
plt.plot(inputdata[1,10,:,0])
plt.title('Normalized Voltage of Pin 11')
plt.figure()
plt.plot(inputdata[1,11,:,0])
plt.title('Normalized Voltage of Pin 12')
plt.figure()
plt.plot(inputdata[1,12,:,0])
plt.title('Normalized Voltage of Pin 13')
plt.figure()
plt.plot(inputdata[1,13,:,0])
plt.title('Normalized Voltage of Pin 14')
plt.figure()
plt.plot(inputdata[1,14,:,0])
plt.title('Normalized Voltage of Pin 15')
plt.figure()
plt.plot(inputdata[1,15,:,0])
plt.title('Normalized Voltage of Pin 16')

plt.figure()
plt.plot(inputdata[1,2,:,0])
plt.axvspan(len(inputdata[1,1,:,0])-128*5-1,len(inputdata[1,1,:,0])-128*2-1, color='red', alpha=0.5)
plt.title('Pin 2')

plt.figure()
plt.plot(inputdata[1,1,:,0])
plt.axvspan(len(inputdata[1,1,:,0])-128*3-1,len(inputdata[1,1,:,0])-128*0-1, color='red', alpha=0.5)
plt.title('Pin 1')

plt.figure()
plt.plot(inputdata[1,0,:,0])
plt.axvspan(len(inputdata[1,1,:,0])-128*1-1,len(inputdata[1,1,:,0])-128*0-1, color='red', alpha=0.5)
plt.axvspan(0,128*2, color='red', alpha=0.5)

#------------------------------------------------------------------------------
'FINETUNING MODEL'
# input_filename = [csv_folder +'voltage first.csv',csv_folder+'voltage second.csv',
#                   csv_folder+'voltage third.csv',csv_folder+'voltage fourth.csv',
#                   csv_folder+'voltage fifth.csv',csv_folder+'voltage sixth.csv',
#                   csv_folder+'voltage seventh.csv',csv_folder+'voltage eighth.csv']
                  
# image_filename = [folder + 'Labelled 1022 first recording', folder +'Labelled 1022 second recording',
#                   folder +'Labelled 1022 third recording', folder +'Labelled 1022 fourth recording',
#                   folder + 'Labelled 1024 fifth recording',folder + 'Labelled 1024 sixth recording', 
#                   folder +'Labelled 1024 seventh recording',folder +'Labelled 1025 eighth recording']
os.chdir(r'C:\Users\lab716a\Documents\School\EIT\CameraTrain\EIT image label')
model = load_model('modeltrain_on_4circles.h')
for layer in model.layers[:6]:  # Freezing first 6 layers (Conv2D, MaxPooling2D, Dense, Dropout, Reshape, BiRNN)
    layer.trainable = False 
# test_filename = [csv_folder +'voltage first.csv',csv_folder+'voltage second.csv']

# testimage_filename = [folder + 'Labelled 1022 first recording', folder +'Labelled 1022 second recording']

input_filename = [csv_folder+'voltage fifth.csv',csv_folder+'voltage sixth.csv']
image_filename = [folder + 'Labelled 1024 fifth recording',folder + 'Labelled 1024 sixth recording']

test_filename = input_filename
testimage_filename = image_filename
inputdata = []
testdata = []
shape = (0,128,128)
index = np.arange(0,723)

images = np.empty(shape)
for i,filename in enumerate(input_filename):
    print('Chunk number ' + str(chunk))
    'Voltage input'
    data,index = read_voltage(index, chunk, filename)
    # data = np.array(data)[:,::2,:]  #skipping 1 pin/ 3 pins
    inputdata += data;
    numberofdata = len(glob.glob('Labelled 1022 first recording/Vector/'+'*jpg'))
    imagesofonefile = np.array(load_image(image_filename[i] +'/Vector/',numberofdata,index)).astype(int)
    images = np.vstack((images,imagesofonefile))
    
'Test data for different dataset'
testimages = np.empty(shape)        
for i,filename in enumerate(test_filename):
    data,index = read_voltage(index,chunk,filename)
    testdata += data;
        
    imagesofonefile = np.array(load_image(testimage_filename[i] +'/Vector/',numberofdata,index)).astype(int)
    testimages = np.vstack((testimages,imagesofonefile))

inputdata = np.array(inputdata).astype(float).astype(int)
    
inputdata = inputdata[...,np.newaxis] #adding new axis at the last dimension, for conv2d
    

'Normalization'
mean = inputdata.mean(axis=(0,2),keepdims = True)
var = inputdata.std(axis=(0,2),keepdims = True)
inputdata = (inputdata-mean)/var

images = 1*(images > 100)

'Experiment for testing model trained on one dataset of circles, test data on different circle dataset'
'Pre-processing test data'
testdata = np.array(testdata).astype(float).astype(int)
testdata = testdata[...,np.newaxis]
'NOrmalization'
mean = testdata.mean(axis=(0,2),keepdims = True)
var = testdata.std(axis =(0,2), keepdims = True)
    
testdata = (testdata-mean)/var
testimages = 1*(testimages>100)
inputdata,images = shuffle(inputdata,images,random_state = 42)
'Take a portion of dataset for finitetuning'
inputdata = inputdata[:int(inputdata.shape[0]/2)]
images = images[:int(images.shape[0]/2)]

testdata,testimages = shuffle(testdata,testimages,random_state = 42)

'Reduce batch size for finetuning'
train_dataset = tf.data.Dataset.from_tensor_slices((inputdata, images)).batch(int(batch_size/2))
test_dataset = tf.data.Dataset.from_tensor_slices((testdata, testimages)).batch(int(batch_size/2))

'Reduce learning rate for finetuning'
opt = tf.keras.optimizers.Adam(learning_rate=learning_rate/10)
callback = SchedulerandTrackerCallback(scheduler)
model.compile(loss='binary_crossentropy', optimizer = opt,metrics = ['accuracy'])
'Reduce number of epochs for finetuning'
history = model.fit(train_dataset, epochs=int(epochs/8), validation_data= test_dataset,callbacks = callback,shuffle = True)

# #------------------------------------------------------------------------------
'Plot 1 frame of 1 electrodes'
# for i in range(16):
#     plt.figure(i)
#     plt.plot(inputdata[0][i])
#     print(np.mean(inputdata[0][i]))
# inputdata = inputdata / np.linalg.norm(inputdata)

#------------------------------------------------------------------------------
'Train Model'
#GPU (Checking GPU status)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
        
model = model_dict[model_name](inputdata.shape[1:], output_shape=images.shape[1:])


history = history.history
'Plot loss'
plt.figure()
plt.plot(history['loss'])
#------------------------------------------------------------------------------

plt.figure()
plt.plot(history['val_accuracy'])    
'Plotting'

res = model.predict(testdata) #x_test
threshold = 0.5
binary_reconstruction = np.where(res >= threshold, 1, 0)
# accu = np.mean(np.abs(binary_reconstruction - y_test))
y_test_flat = testimages.reshape(-1) #y_test
binary_reconstruction_flat = binary_reconstruction.reshape(-1)
accu = np.mean(np.equal(testimages,binary_reconstruction))
accu1 = precision_score(y_test_flat,binary_reconstruction_flat)
accu2 = recall_score(y_test_flat,binary_reconstruction_flat)
f1 = f1_score(y_test_flat,binary_reconstruction_flat)