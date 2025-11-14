Initially we set up all the important libraries required to us , like tensorflow , librose ,sklearn.(preprocess,ensemble) etc... 
After that we need to mount drive and then set path of audio samples files and create a pandas DataFrame to store the data and labels.

Now the real work starts---->
# Feature Extraction and Database Building
MFCC - Mel frequency cepstral coefficients. Extraction technique widely used in speech and audio processing for tasks like speech recognition, speaker identification, and music analysis.

MFCCs role -

1.Signal analysis - It breaks down complex signals into simpler forms repr rate and characterisits of sound waves

2.Frequency transform - humans do not understand freq on a linear scale . Hnece mere mel freq. is used which rounds off to human auditory systems

3.Cepstral repr - mel freq --- > cepstrum where audios signal pitch is seperated form slow variations (timbre) carrying the most imp info regarding the audio ( ie . most important part of audio is seperated )

features related to MFCC

1.Windowing ----> stops spectral leakages , or sudden jumps in amplitudes of audio .

2.FFT ---->converts time domained signal of each framed signal into freq domain.

3.Mel-filterblank ---- > Frequency Band seperation - Divided FFT output into these bands capturing E level of each band

4.Log mel-spectrum - ie logartithmic compression -- By taking the logarithm of the output from the Mel-filterbank, the dynamic range of the signal is compressed.

# MODEL SELECTION

There are many models on which we can train data on . ANN , CNN etc.

ANN - Artificial Neural Network -- a computer system modeled after the human brain that learns from data to perform tasks like classification and pattern recognition.

CNN - Convultional Neural Network --  a type of machine learning model primarily used for image recognition that uses special layers to automatically learn and detect patterns in images, such as edges, textures, and shapes

CNN > ANN in this case.

In short , use of the CNN features used here ---
Dense - fully connected layers , used in classification / regression

Conv1d - 1D layer used for time series ,audio signals ,freq

conv2d - 2D layer , used for imgs /spectrograms

MaxPooling1D / MaxPool2D - Pooling layers

Reduce the spatial/temporal dimension after convolution to extract key features.

Dropout - Helps preventing overfitting by randomly dropping neurons.

flatten - convert all layers into dense layers

### Implement data augmentation

Apply data augmentation techniques to the training data to increase its size and variability, which can help the model generalize better and reduce overfitting.

useful for smaller datasets and is a common technique in fields like CV, where transformations like flipping, rotating, and cropping are applied to existing images to create new, synthetic data.

IT INCREASES ACCURACY BY ----

1.INCREASING DATA DIVERSITY WHICH HELPS IN GENERALIZATION

2.REDUCES OVERFITTING AS IT BECOMES TOUGH FOR MODEL TO MEMORIZE THE DATA

3.IMPORVES ROBUSTNESS AS THERE ARE VARIATIONS IN DATA WHICH HELPS IN TRAINING

# **Structure of this Augmentation of data**
We mixed audio signals , ie created a new feature mixing it by taking a ratio , ie signal A - > .6 , signal B - > .4 , creating more variations.

We set the parameters by defining augment_audio -
  noise_factor : Factor to scale the added noise.

  shift_factor : Factor to determine the maximum shift in time.

  pitch_factor : Factor to determine the maximum pitch shift.

 i.   noise = np.random.randn(len(augmented_signal))
    augmented_signal = augmented_signal + noise_factor * noise            
 ---> to add some background noise so that more variations of sound can be identified and training can become more rigorous

 ii.   shift = int(sr * shift_factor * random.uniform(-1, 1))
    augmented_signal = np.roll(augmented_signal, shift)
--->shifts the audio samples along an axis (np.roll())in random amounts at any time
understanding roll by an example --->
signal = [2,3,4,5]

np.roll(signal,2) = [4,5,2,3]  but in producing audio we dont want the last part of audio to come in the beginning , hence we have to mark em as zero .

corrected ---> np.roll(signal ,2) =[0,0,2,3]

iii. also we are trying to vary the speech a little up and down , to create more variations.

iv. Now we need to take one max length of audio and keeping it as a reference , trim or fix all the other audios of the same length

v.Now we applied augmentation 2+ the original sample (1+2=3)

after padding , storing data -

vi.We need to extract the features . as raw audiofomrms tougher to understand by the CNN ,it is first converted to MFCC .

    librosa.feature.mfcc   
  ---> extracts MFCCs , each clip 128 D vectors


# ADDING BATCH NORMALIZATION LAYER --- > INC IN LAYERS CAN POTENTIALLY INCREASE ACCURACY AND LION OPTIMIZER

It increases accuracy as it normalizises the effects , helping in increasing LR , leading to mroe stable and faster training process

Lion optimizers ----> better than adam .
It is a SGD method which uses sign oprator to control the magnitude .lion is more efficient than Adam as it only keeps memory of Adam.

Lion optimizers need more time to learn , hence 10^-6 order to LR is better as comapred to normal LR of adam as 10^-3 orders as it will give high value errors.
Here but Adam is giving more accuracy.

AdamW optimizer also used , but still accuracy cannot be increased .

    layers.Conv1D(32, 3, activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling1D(2),

conv1d ---> applies 32 feature detectors for each of length 3 samples

ReLu --->converts negative outputs to 0 and prevents vanishing of gradients

Batch Normalization --->stab. training mean =0 var =1 per batch

maxpooling --->removes temporal res by half stride =2 , Keeps the strongest activation (most important feature) in each 2-sample window.

lets understand it by an example .
1D map --> [1,2,3,4,5,6]  , layers.MaxPooling1D(2),

here pools of 2 are created . [1,2] ,[3,4] ,[5,6] . Max vals of pools selected . 2,4,6 . Hence here we are keeping the strongest activation(or the maximum values) and the most prominent feature.

    layers.GlobalAveragePooling1D(),

---> Converts each feature map (from 256 filters) into one single value (mean across time).

    layers.Dense(128, activation='relu'),

---> learns non-linear comb of extracted features

    layers.Dropout(0.4),

---> dropping 40 % of neurons

    layers.Dense(5, activation='softmax')

---> 5 target classes . softmax converts raw scores into prob summing to 1 and model predicts most likely class


# ResNet-9
is a compact version of the original ResNet architecture.
It uses residual connections (skip connections) to help the network learn deeper patterns without vanishing gradients.

In this method of RESNET , we need to solve problem of exploding gradient . Hence to fix this , residual blocks were introduced which uses skip connections which connects the layers of activation by skipping connections .The skip connection connects layers again by skipping some layers to form residual block.

Pros of ResNet --->
---> If any layer gets hurt in performance then by regularization it will be skipped resulting in a very deep NN

This method was first tested on a CIFAR 10 dataset ( collection of images that are commonly used to train machine learning and computer vision algorithms) over 100s of layers

Tried adding it but it did not work properly as accuracy became lower .


# TESTING

We have trained the data , now we have to test randomly

If max_len is already defined it remains .
16000*3 = 48K samples.

    if len(signal) > max_len:
        return signal[:max_len]
    elif len(signal) < max_len:
        pad_width = max_len - len(signal)
        return np.pad(signal, (0, pad_width), mode='constant')
    else:
        return signal
---> ensuring every input waveform is of max_len samples , if longer then truncates , if shorted then pads w/ 0

    scaled_feature = np.mean(mfccs.T, axis=0)
Transposes MFCCs to shape (time_frames, n_mfcc) then averages across time → one 1D vector of length n_mfcc


THEN TEST EVALUATED DATA AND PLOTTED ON GRAPH
