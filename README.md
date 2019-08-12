# IJCAI2019-DRL4SER
The python implementation for paper "Towards Discriminative Representation Learning for Speech Emotion Recognition" in IJCAI-2019

Speech emotion recognition system.
For training
python train.py
# For more adjustment, plz modify the hparamters
For inferring
python inferring.py --checkpoint='THE_PATH_OF_MODEL'
## AS a component, plz import the class in "emotion_inferring.predictor" directly
BE CARE: The system can use different types of input for emotion inferring. FOR TTS, plz use mel-spec('raw' in hparamters) for better coordination.
