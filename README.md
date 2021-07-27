# How Speech is Recognized to Be Emotional - A Study Based on Information Decomposition

This code is a pytorch version for speechflow model in "How Speech is Recognized to Be Emotional - A Study Based on Information Decomposition", which is modified for our investigations from the original [Speechflow](https://github.com/auspicious3000/SpeechSplit).

For ACRNN in this paper, its code can be found at [3-D ACRNN](https://github.com/xuanjihe/speech-emotion-recognition).


## Dependencies

This project is built with python 3.6, for other packages, you can install them by ```pip install -r requirements.txt```.


## To Prepare Training Data(take VCTK as an example here, for other dataset, you should modify some settings in below files)

1. Prepare your wavefiles

2. ```cd tools```

3. make training data and valid data, for better training, you can make an entire training wave file for each speaker by ```sh make_cat.sh```, and make separate validation wave files for each speaker by ```sh make_valid.sh```, here validation speakers can not appear in training data.

4. Extract spectrogram and f0: ```python make_spect_f0_VCTK.py```
    - you should provide d-vectors by a pre-trained model. D-vectors for speakers in VCTK calculated by our pre-trained model are provided in ```./data/VCTK_dvec/dvector_VCTK.npz```
    - a mapping from speakers to IDs and another from speakers to corresponding genders are needeed
    - for validation data, you should change some settings

5. Generate training metadata: ```make_metasplit_VCTK.py```

6. Generate validation metadata: ```make_demodata_VCTK.py```


## To Train

1. change setting at ```hparams.py``` and ```run.py```

2. Run the training scripts: ```python run.py```


## To implement information decomposition

An example is provide in ```infer_batch.py```, in which you should define the input pickle file.


## Final Words

This speechflow model is the most important tool for us to analyse impacts between speech components and performance of mordern emotion recognition systems. This code is modified for our task from the original [Speechflow](https://github.com/auspicious3000/SpeechSplit). Thanks for Kaizhi Qian providing the original code, which is much helpful for us.
