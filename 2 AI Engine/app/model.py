import os
import pip
import numpy as np
import librosa

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from keras.utils import np_utils
from keras.layers import Dense, Dropout, Flatten, Conv1D, Input, MaxPooling1D
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
from keras.models import load_model	
	
def load_data(root):

	global classes, x_train, x_valid, y_train, y_valid

	train_audio_path = f'app/{root}'
	
	labels = os.listdir(train_audio_path)
	print(labels)
	
	all_wave = []
	all_label = []
	for label in labels:
		waves = [f for f in os.listdir(train_audio_path + '/'+ label) if f.endswith('.wav')]
		print(f'{label},{len(waves)}')
		r = 0
		for wav in waves:
			try:
				sample, sample_rate = librosa.load(train_audio_path + '/' + label + '/' + wav, sr = 16000)
				sample = librosa.resample(sample, sample_rate, 8000)
				if(len(sample) == 8000):
					r += 1
					all_wave.append(sample)
					all_label.append(label)
			except Exception as e:
				print(e)
		print(f'{label}:{r}')
	
	le = LabelEncoder()
	y = le.fit_transform(all_label)
	classes = list(le.classes_)
	y = np_utils.to_categorical(y, num_classes=len(labels))
	all_wave = np.array(all_wave).reshape(-1, 8000, 1)
	
	x_train, x_valid, y_train, y_valid = train_test_split(np.array(all_wave),np.array(y),stratify=y,test_size = 0.2,random_state=777,shuffle=True)
	
def create_model():
    inputs = Input(shape=(8000,1))
    
    #First Conv1D layer
    conv = Conv1D(8,13, padding='valid', activation='relu', strides=1)(inputs)
    conv = MaxPooling1D(3)(conv)
    conv = Dropout(0.3)(conv)
    
    #Second Conv1D layer
    conv = Conv1D(16, 11, padding='valid', activation='relu', strides=1)(conv)
    conv = MaxPooling1D(3)(conv)
    conv = Dropout(0.3)(conv)
    
    #Third Conv1D layer
    conv = Conv1D(32, 9, padding='valid', activation='relu', strides=1)(conv)
    conv = MaxPooling1D(3)(conv)
    conv = Dropout(0.3)(conv)
    
    #Fourth Conv1D layer
    conv = Conv1D(64, 7, padding='valid', activation='relu', strides=1)(conv)
    conv = MaxPooling1D(3)(conv)
    conv = Dropout(0.3)(conv)
    
    #Flatten layer
    conv = Flatten()(conv)
    
    #Dense Layer 1
    conv = Dense(256, activation='relu')(conv)
    conv = Dropout(0.3)(conv)
    
    #Dense Layer 2
    conv = Dense(128, activation='relu')(conv)
    conv = Dropout(0.3)(conv)
    
    outputs = Dense(len(labels), activation='softmax')(conv)
    
    model = Model(inputs, outputs)
    return model
	
def train_model(from_scratch = False):
    if(from_scratch or not os.path.exists('best_model.hdf5')):
        model = create_model()
        print('Model created')
    else:
        model = load_model('best_model.hdf5')
        print('Model loaded')
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam',metrics = ['accuracy'])
    es = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 1, patience = 10, min_delta = 0.0001)
    mc = ModelCheckpoint('best_model.hdf5', monitor = 'val_accuracy', verbose = 1, save_best_only = True, mode = 'max')
    model.fit(x_train, y_train , epochs = 100, callbacks = [es, mc], batch_size = 32, validation_data = (x_valid, y_valid))
    return model
	
def load_trainded_model():
    model = load_model('best_model.hdf5')
    return model

def initialize_model(already_trained = True):
	global model
	model =  load_trainded_model() if already_trained else train_model()
	summary = model.summary()
	return summary
	
def predict_audio(audio, sample_rate):
	audio = np.array(audio, dtype=np.float32)
	audio = librosa.resample(audio, sample_rate, 8000)
	prob = model.predict(audio.reshape(1,8000,1))
	index = np.argmax(prob[0])
	prediction = classes[index]
	return prediction