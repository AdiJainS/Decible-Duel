# Decible-Duel
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

