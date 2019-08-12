# IJCAI2019-DRL4SER
The python implementation for paper "Towards Discriminative Representation Learning for Speech Emotion Recognition" in IJCAI-2019

Only three steps are required to explore this projection.
1, Data preparation
2, Model Training
3, Inferring Emotion from given speech
4, And you can also export the model into .pb format, to provide a personal service.

Just enjoying.
# For data preparation

"python feature_extractor/extract_audio_features.py"

You may need the installation of OpenSMILE to achieve the best performance.

# For training
"python train.py"

You can adjust the hyper-parameters to better match up the task and model

# For inferring
"python inferring.py --checkpoint=<THE_PATH_OF_MODEL> "

A .pb is suggestion in constructing service, however, the inferring.py is provided to assess the trained model quickly and easily.



