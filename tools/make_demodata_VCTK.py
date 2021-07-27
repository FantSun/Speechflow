import os
import pickle
import numpy as np

rootDir = '../data/valid'
tarDir = '../data/training_cat'
tarFile = 'demo.pkl'
len_pad = 192
dirName, subdirList, _ = next(os.walk(os.path.join(rootDir, 'spmel')))
print('Found directory: %s' % dirName)

dvecDir = '../data/VCTK_dvec/dvector_VCTK.npz'
spk2dvec = {}
spk_dvec = np.load(dvecDir)['spkers']
dvecs = np.load(dvecDir)['feats']
for i in range(spk_dvec.shape[0]):
    spker = spk_dvec[i]
    dvec = dvecs[i]
    print('dvec:', spker, dvec.shape)
    spk2dvec[spker] = dvec

speakers = []
for speaker in sorted(subdirList):
    print('Processing speaker: %s' % speaker)
    utterances = []
    utterances.append(speaker)
    _, _, fileList = next(os.walk(os.path.join(dirName,speaker)))
    
    # use hardcoded onehot embeddings in order to be cosistent with the test speakers
    # modify as needed
    spkid = spk2dvec[speaker].reshape(1, -1)
    utterances.append(spkid)
    
    # create file list
    for fileName in sorted(fileList):
        info = []
        info.append(np.load(os.path.join(os.path.join(rootDir, 'spmel'), speaker, fileName))[:len_pad])
        info.append(np.load(os.path.join(os.path.join(rootDir, 'raptf0'), speaker, fileName))[:len_pad])
        info.append(np.load(os.path.join(os.path.join(rootDir, 'raptf0'), speaker, fileName))[:len_pad].shape[0])
        info.append('valid')
        utterances.append(info)
    speakers.append(utterances)
    
with open(os.path.join(tarDir, tarFile), 'wb') as handle:
    pickle.dump(speakers, handle)    
