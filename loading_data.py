import os
import numpy as np
import csv

def read_voltage(index, chunk, filepath = './Image EIT/voltage.csv',chunk_size = 1):
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