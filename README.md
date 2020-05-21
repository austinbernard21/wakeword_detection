# wakeword_detection
custom wakeword detection on minimal size dataset

audio dataset used for negative samples was the google speech dataset with single words spoken in one second audio wav files.
My custom wakeword was recorded about 100 times with 50 additional test recordings. The model is trained on a very low dataset since my wake word has very low amount of data
The model performance performs very well in a enclosed environment, but the real time detection is still in progress. The resampling of the audio and 
decimation of the audio samples in real time are creating a slight difference in the validation data  distribution and the actual test data distribution. This can be fixed with more data and possibly a more complex model. 
This was just to train a simple wake word detection system with a minimal amount of custom data.

