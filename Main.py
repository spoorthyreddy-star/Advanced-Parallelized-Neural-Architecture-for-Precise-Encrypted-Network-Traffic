
from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
import matplotlib.pyplot as plt
import numpy as np
from tkinter import simpledialog
from tkinter import filedialog
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from keras.callbacks import ModelCheckpoint 
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras.models import Sequential, load_model
from keras.utils.np_utils import to_categorical
from keras.layers import Conv2D, MaxPool2D, InputLayer, BatchNormalization, GlobalAveragePooling2D
from keras.optimizers import Adam

main = tkinter.Tk()
main.title("Encrypted Network Traffic Classification Using Deep and Parallel Network-In-Network Models") #designing main screen
main.geometry("1300x1200")

global filename, dataset
global X, Y
global X_train, X_test, y_train, y_test
global accuracy, precision, recall, fscore, labels, nin_model
global scaler, labels, label_encoder

def uploadDataset():
    global filename, dataset, labels
    filename = filedialog.askopenfilename(initialdir="Dataset")
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n\n")
    dataset = pd.read_csv(filename)
    text.insert(END,str(dataset))
    labels, label_count = np.unique(dataset['Label'], return_counts=True)
    label = dataset.groupby('Label').size()
    label.plot(kind="bar")
    plt.xlabel("Network Category Type")
    plt.ylabel("Count")
    plt.title("Network Category Graph")
    plt.show()

def DatasetPreprocessing():
    text.delete('1.0', END)
    global X, Y, dataset, label_encoder
    global X_train, X_test, y_train, y_test, scaler

    #dataset contains non-numeric values but ML algorithms accept only numeric values so by applying Lable
    #encoding class converting all non-numeric data into numeric data
    dataset.fillna(0, inplace = True)
    dataset.drop(['Traffic_Type'], axis = 1,inplace=True)
    label_encoder = []
    columns = dataset.columns
    types = dataset.dtypes.values
    for i in range(len(types)):
        name = types[i]
        if name == 'object': #finding column with object type
            le = LabelEncoder()
            dataset[columns[i]] = pd.Series(le.fit_transform(dataset[columns[i]].astype(str)))#encode all str columns to numeric 
            label_encoder.append(le)    
    text.insert(END,"Dataset Normalization & Preprocessing Task Completed\n\n")
    text.insert(END,str(dataset)+"\n\n")
    #dataset preprocessing such as replacing missing values, normalization and splitting dataset into train and test
    data = dataset.values
    X = data[:,0:data.shape[1]-1] #extracting X and Y Features from the dataset
    Y = data[:,data.shape[1]-1]
    print(X.shape)
    print(np.unique(Y))
    print(Y)
    Y = Y.astype(int)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices) #shuffling the dataset
    X = X[indices]
    Y = Y[indices]
    #normalizing or scaling values
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    #reshape dataset as 3 dimenssion
    X = np.reshape(X, (X.shape[0], X.shape[1], 1, 1))
    Y = to_categorical(Y)
    #splitting dataset into train and test where application using 80% dataset for training and 20% for testing
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2) #split dataset into train and test
    text.insert(END,"Dataset Train & Test Splits\n")
    text.insert(END,"Total images found in dataset : "+str(X.shape[0])+"\n")
    text.insert(END,"80% dataset used for training  : "+str(X_train.shape[0])+"\n")
    text.insert(END,"20% dataset user for testing   : "+str(X_test.shape[0])+"\n")


def calculateMetrics(algorithm, testY, predict):
    global labels
    p = precision_score(testY, predict,average='macro') * 100
    r = recall_score(testY, predict,average='macro') * 100
    f = f1_score(testY, predict,average='macro') * 100
    a = accuracy_score(testY,predict)*100
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    text.insert(END,algorithm+" Accuracy  : "+str(a)+"\n")
    text.insert(END,algorithm+" Precision : "+str(p)+"\n")
    text.insert(END,algorithm+" Recall    : "+str(r)+"\n")
    text.insert(END,algorithm+" FSCORE    : "+str(f)+"\n\n")
    conf_matrix = confusion_matrix(testY, predict)
    ax = sns.heatmap(conf_matrix, xticklabels = labels, yticklabels = labels, annot = True, cmap="viridis" ,fmt ="g");
    ax.set_ylim([0,len(labels)])
    plt.title(algorithm+" Confusion matrix") 
    plt.ylabel('True class') 
    plt.xlabel('Predicted class') 
    plt.show() 

#now train existing standard CNN algorithm    
def runStandardCNN():
    text.delete('1.0', END)
    global accuracy, precision, recall, fscore
    global X_train, y_train, X_test, y_test
    accuracy = []
    precision = []
    recall = [] 
    fscore = []
    #training standard CNN without multilayer perceptron and global average pooling
    standard_cnn = Sequential()
    standard_cnn.add(Convolution2D(32, (1, 1), input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3]), activation = 'relu'))
    standard_cnn.add(MaxPooling2D(pool_size = (1, 1)))
    standard_cnn.add(Convolution2D(32, (1, 1), activation = 'relu'))
    standard_cnn.add(MaxPooling2D(pool_size = (1, 1)))
    standard_cnn.add(Flatten())
    standard_cnn.add(Dense(units = 256, activation = 'relu'))
    standard_cnn.add(Dense(units = y_train.shape[1], activation = 'softmax'))
    standard_cnn.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    #train and load the model
    if os.path.exists("model/standard_cnn_weights.hdf5") == False:
        model_check_point = ModelCheckpoint(filepath='model/standard_cnn_weights.hdf5', verbose = 1, save_best_only = True)
        hist = standard_cnn.fit(X_train, y_train, batch_size = 256, epochs = 10, validation_data=(X_test, y_test), callbacks=[model_check_point], verbose=1)
    else:
        standard_cnn = load_model("model/standard_cnn_weights.hdf5")
    predict = standard_cnn.predict(X_test)
    predict = np.argmax(predict, axis=1)
    testY = np.argmax(y_test, axis=1)
    calculateMetrics("Existing Standard CNN", testY, predict)

#run propose NIN model
def runNINCNN():
    global nin_model
    global X_train, y_train, X_test, y_test
    #now creating multi layer perceptron with global average pooling layer where first 3 layers are used to processed packet header and remaining layer
    #will process packet body
    nin_model = Sequential()
    nin_model.add(InputLayer(input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3])))
    #creating first hidden layer with 25 neurons to filter data 25 times for packet header 
    nin_model.add(Conv2D(25, (5, 5), activation='relu', strides=(1, 1), padding='same'))
    #defining another layer
    nin_model.add(MaxPool2D(pool_size=(2, 2), padding='same'))
    nin_model.add(Conv2D(50, (5, 5), activation='relu', strides=(2, 2), padding='same'))
    nin_model.add(MaxPool2D(pool_size=(2, 2), padding='same'))
    nin_model.add(BatchNormalization())
    #now creating second filtration layer to filter packet body
    nin_model.add(Conv2D(70, (3, 3), activation='relu', strides=(2, 2), padding='same'))
    nin_model.add(MaxPool2D(pool_size=(1, 1), padding='valid'))
    nin_model.add(BatchNormalization())
    nin_model.add(Dense(units=100, activation='relu'))
    nin_model.add(Dense(units=100, activation='relu'))
    nin_model.add(Dropout(0.25))
    #now adding global average pooling layer
    nin_model.add(GlobalAveragePooling2D())
    nin_model.add(Dense(units=y_train.shape[1], activation='softmax'))
    #compiling the model
    nin_model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
    #now train and load the model
    if os.path.exists("model/nin_weights.hdf5") == False:
        model_check_point = ModelCheckpoint(filepath='model/nin_weights.hdf5', verbose = 1, save_best_only = True)
        hist = nin_model.fit(X_train, y_train, batch_size = 256, epochs = 10, validation_data=(X_test, y_test), callbacks=[model_check_point], verbose=1)     
    else:
        nin_model.load_weights("model/nin_weights.hdf5")
    #performing prediction on test data and calculate accuracy and other metrics    
    predict = nin_model.predict(X_test)
    predict = np.argmax(predict, axis=1)
    testY = np.argmax(y_test, axis=1)
    calculateMetrics("Propose Parallel Deep NIN (Network in Network) Algorithm", testY, predict)

def graph():
    df = pd.DataFrame([['Standard CNN','Accuracy',accuracy[0]],['Standard CNN','Precision',precision[0]],['Standard CNN','Recall',recall[0]],['Standard CNN','FSCORE',fscore[0]],
                       ['Propose Parallel Deep NIN Algorithm','Accuracy',accuracy[1]],['Propose Parallel Deep NIN Algorithm','Precision',precision[1]],['Propose Parallel Deep NIN Algorithm','Recall',recall[1]],['Propose Parallel Deep NIN Algorithm','FSCORE',fscore[1]],
                      ],columns=['Algorithms','Accuracy','Value'])
    df.pivot("Algorithms", "Accuracy", "Value").plot(kind='bar')
    plt.title("All Algorithm Comparison Graph")
    plt.show()    

def predict():
    global nin_model, scaler, label_encoder, labels
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir="Dataset")#upload test data
    dataset = pd.read_csv(filename)#read data from uploaded file
    dataset.fillna(0, inplace = True)#removing missing values
    index = 0
    columns = dataset.columns
    types = dataset.dtypes.values
    for i in range(len(types)): #label encoding to convert non-numeric data to numeric data
        name = types[i]
        if name == 'object': #finding column with object type
            dataset[columns[i]] = pd.Series(label_encoder[index].fit_transform(dataset[columns[i]].astype(str)))
            index = index + 1
    dataset = dataset.values
    X = scaler.transform(dataset)#normalizing values
    X = np.reshape(X, (X.shape[0], X.shape[1], 1, 1))
    traffic_type_predict = nin_model.predict(X)#performing prediction on test data
    for i in range(len(X)):
        text.insert(END,"Traffic Test Data : "+str(dataset[i]))
        text.insert(END,"Network Traffic Classified As ===> "+labels[int(np.argmax(traffic_type_predict[i]))])
        text.insert(END,"\n")
    
    


font = ('times', 16, 'bold')
title = Label(main, text='Encrypted Network Traffic Classification Using Deep and Parallel Network-In-Network Models')
title.config(bg='LightGoldenrod1', fg='medium orchid')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 12, 'bold')
text=Text(main,height=22,width=140)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=200)
text.config(font=font1)


font1 = ('times', 12, 'bold')
uploadButton = Button(main, text="Upload ISCX VPN-nonVPN Dataset", command=uploadDataset)
uploadButton.place(x=50,y=100)
uploadButton.config(font=font1)  

preButton = Button(main, text="Dataset Preprocessing", command=DatasetPreprocessing)
preButton.place(x=370,y=100)
preButton.config(font=font1) 

nbButton = Button(main, text="Run Standard CNN Algorithm", command=runStandardCNN)
nbButton.place(x=610,y=100)
nbButton.config(font=font1) 

rfButton = Button(main, text="Run Deep Parallel NIN Algorithm", command=runNINCNN)
rfButton.place(x=860,y=100)
rfButton.config(font=font1) 

graphButton = Button(main, text="Comparison Graph", command=graph)
graphButton.place(x=50,y=150)
graphButton.config(font=font1)

predictButton = Button(main, text="Traffic Classification using Encrypted Test Data", command=predict)
predictButton.place(x=370,y=150)
predictButton.config(font=font1)  

#main.config(bg='OliveDrab2')
main.mainloop()
