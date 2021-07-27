import os
import sys
sys.path.append("..")
import pickle
import numpy as np
import soundfile as sf
from scipy import signal
from librosa.filters import mel
from numpy.random import RandomState
from pysptk import sptk
from utils import butter_highpass
from utils import speaker_normalization
from utils import pySTFT


# Modify as needed
spknum = 200
rootDir = '../data/VCTK_cat_train'
tarDir = '../data/training_cat'

targetDir_f0 = tarDir + '/raptf0'
targetDir_dvec = tarDir + '/dvec'
targetDir = tarDir + '/spmel'
spkDir = '../data/VCTK_wav/spk2spkid' # defining a mapping from a speaker to its id
dvecDir = '../data/VCTK_dvec/dvector_VCTK.npz' # defining a mapping from a speaker to its d-vector
spk2gen = pickle.load(open('../data/VCTK_wav/spk2gen.pkl', "rb")) # defining a mapping from a speaker to its gender

mel_basis = mel(16000, 1024, fmin=90, fmax=7600, n_mels=80).T
min_level = np.exp(-100 / 20 * np.log(10))
b, a = butter_highpass(30, 16000, order=5)

# load speaker ids
spk2spkid = {}
spk_file = open(spkDir)
while True:
    line = spk_file.readline()
    if not line:
        break
    info = line.split( )
    spk2spkid[info[0]] = int(info[1])

# load and save d-vectors
spk_dvec = np.load(dvecDir)['spkers']
dvecs = np.load(dvecDir)['feats']
for i in range(spk_dvec.shape[0]):
    spker = spk_dvec[i]
    ###
    if spk2spkid[spker] >= spknum:
        continue
    dvec = dvecs[i]
    print('dvec:', spker, dvec.shape)
    if not os.path.exists(os.path.join(targetDir_dvec, spker)):
        os.makedirs(os.path.join(targetDir_dvec, spker))
    np.save(os.path.join(targetDir_dvec, spker, spker),
                dvec, allow_pickle=False)
 
dirName, subdirList, _ = next(os.walk(rootDir))
print('Found directory: %s' % dirName)

for subdir in sorted(subdirList):
    ###
    if spk2spkid[subdir] >= spknum:
        continue
    print('mel&f0:', subdir)
    
    if not os.path.exists(os.path.join(targetDir, subdir)):
        os.makedirs(os.path.join(targetDir, subdir))
    if not os.path.exists(os.path.join(targetDir_f0, subdir)):
        os.makedirs(os.path.join(targetDir_f0, subdir))    
    _,_, fileList = next(os.walk(os.path.join(dirName,subdir)))
    
    if spk2gen[subdir] == 'M':
        lo, hi = 50, 250
    elif spk2gen[subdir] == 'F':
        lo, hi = 100, 600
    else:
        raise ValueError
        
    prng = RandomState(int(subdir[1:])) 
    for fileName in sorted(fileList):
        # read audio file
        x, fs = sf.read(os.path.join(dirName,subdir,fileName))
        assert fs == 16000
        if x.shape[0] % 256 == 0:
            x = np.concatenate((x, np.array([1e-06])), axis=0)
        y = signal.filtfilt(b, a, x)
        wav = y * 0.96 + (prng.rand(y.shape[0])-0.5)*1e-06
        if wav.shape[0] < 10000:
            continue
        
        # compute spectrogram
        D = pySTFT(wav).T
        D_mel = np.dot(D, mel_basis)
        D_db = 20 * np.log10(np.maximum(min_level, D_mel)) - 16
        S = (D_db + 100) / 100        
        
        # extract f0
        f0_rapt = sptk.rapt(wav.astype(np.float32)*32768, fs, 256, min=lo, max=hi, otype=2)
        index_nonzero = (f0_rapt != -1e10)
        mean_f0, std_f0 = np.mean(f0_rapt[index_nonzero]), np.std(f0_rapt[index_nonzero])
        f0_norm = speaker_normalization(f0_rapt, index_nonzero, mean_f0, std_f0)
        print(f0_norm.mean())
        print(f0_norm)
        assert len(S) == len(f0_rapt)
            
        np.save(os.path.join(targetDir, subdir, fileName[:-4]),
                S.astype(np.float32), allow_pickle=False)    
        np.save(os.path.join(targetDir_f0, subdir, fileName[:-4]),
                f0_norm.astype(np.float32), allow_pickle=False)
    print(' ')
