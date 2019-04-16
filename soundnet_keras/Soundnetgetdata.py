from keras.layers import BatchNormalization, Activation, Conv1D, MaxPooling1D, ZeroPadding1D, InputLayer
from keras.models import Sequential
import numpy as np
import matplotlib.pyplot as plt
import librosa #audio library

def build_model():
    #model_weights = np.load(r"C:\Users\djaym7\Desktop\Github\EmotionRecognition\soundnet_keras\models\sound8.npy",encoding = 'latin1').item()
    model_weights = np.load(r"/drive/My Drive/AudioWav/sound8.npy",encoding = 'latin1').item()
    model = Sequential()
    #Input layer: audio raw waveform (1,length_audio,1)
    model.add(InputLayer(batch_input_shape=(1, None, 1)))
    filter_parameters = [{'name': 'conv1', 'num_filters': 16, 'padding': 32,
                          'kernel_size': 64, 'conv_strides': 2,
                          'pool_size': 8, 'pool_strides': 8}, #pool1

                         {'name': 'conv2', 'num_filters': 32, 'padding': 16,
                          'kernel_size': 32, 'conv_strides': 2,
                          'pool_size': 8, 'pool_strides': 8}, #pool2

                         {'name': 'conv3', 'num_filters': 64, 'padding': 8,
                          'kernel_size': 16, 'conv_strides': 2},

                         {'name': 'conv4', 'num_filters': 128, 'padding': 4,
                          'kernel_size': 8, 'conv_strides': 2},

                         {'name': 'conv5', 'num_filters': 256, 'padding': 2,
                          'kernel_size': 4, 'conv_strides': 2,
                          'pool_size': 4, 'pool_strides': 4}, #pool5

                         {'name': 'conv6', 'num_filters': 512, 'padding': 2,
                          'kernel_size': 4, 'conv_strides': 2},

                         {'name': 'conv7', 'num_filters': 1024, 'padding': 2,
                          'kernel_size': 4, 'conv_strides': 2},

                         {'name': 'conv8_2', 'num_filters': 401, 'padding': 0,
                          'kernel_size': 8, 'conv_strides': 2},#output: VGG 401 classes
                         ]

    for x in filter_parameters:
        #for each [zero_padding - conv - batchNormalization - relu]
        model.add(ZeroPadding1D(padding=x['padding']))
        model.add(Conv1D(x['num_filters'],
                         kernel_size=x['kernel_size'],
                         strides=x['conv_strides'],
                         padding='valid'))
        weights = model_weights[x['name']]['weights'].reshape(model.layers[-1].get_weights()[0].shape)
        biases = model_weights[x['name']]['biases']

        model.layers[-1].set_weights([weights, biases])  #set weights in convolutional layer

        if 'conv8' not in x['name']:
            gamma = model_weights[x['name']]['gamma']
            beta = model_weights[x['name']]['beta']
            mean = model_weights[x['name']]['mean']
            var = model_weights[x['name']]['var']

            
            model.add(BatchNormalization())
            model.layers[-1].set_weights([gamma, beta, mean, var]) #set weights in batchNormalization
            model.add(Activation('relu'))
            
        if 'pool_size' in x:
            #add 3 pooling layers
            model.add(MaxPooling1D(pool_size=x['pool_size'],
                                   strides=x['pool_strides'],
                                   padding='valid'))

    return model



def overlap(X, window_size, window_step):
    if window_size % 2 != 0:
        raise ValueError("Window size must be even!")
    # Make sure there are an even number of windows before stridetricks
    append = np.zeros((window_size - len(X) % window_size))
    X = np.hstack((X, append))

    ws = window_size
    ss = window_step
    a = X

    valid = len(a) - ws
    nw = (valid) // ss
    out = np.ndarray((nw,ws),dtype = a.dtype)

    for i in range(nw):
        # "slide" the window along the samples
        start = i * ss
        stop = start + ws
        out[i] = a[start : stop]

    return out



def preprocess(audio):
    audio *= 256.0  # SoundNet requires an input range between -256 and 256
    # reshaping the audio data, in this way it fits into the graph (batch_size, num_samples, num_filter_channels)
    audio = np.reshape(audio, (1, -1, 1))
    return audio



from keras import backend as K
model = build_model()

# return the list of activations as tensors for an specific layer in the model 
def getActivations(data,number_layer):
    intermediate_tensor = []
    #get Hidden Representation function
    get_layer_output = K.function([model.layers[0].input],
                                  [model.layers[number_layer].output])
    
    for audio in data:
        #get Hidden Representation       
        layer_output = get_layer_output([audio])[0] # multidimensional vector
        tensor = layer_output.reshape(1,-1) # change vector shape to 1 (tensor)
        intermediate_tensor.append(tensor[0]) # list of tensor activations for each object in Esc10
    return intermediate_tensor



def load_audio(audio_file):
    sample_rate = 22050  # SoundNet works on monophonic-audio files with sample rate of 22050.
    audio, sr = librosa.load(audio_file, dtype='float32', sr=sample_rate, mono=True) #load audio
    max_size = 110361
    audio=np.pad(audio,pad_width=(0,max_size-len(audio)),mode='constant')
    
    #overlapping 
    #audio = overlap(audio,20,10)
    #output = []
    #for i in audio:
    #    i = preprocess(i)
    #    i = np.asarray(getActivations([i],31))
    #    output.append(i)
    
    
    
    
    audio = preprocess(audio)
    audio = audio.reshape(1,110361)
    return audio





#audio = load_audio(r"C:\Users\djaym7\Desktop\Github\EmotionRecognition\soundnet_keras\kk.wav")

