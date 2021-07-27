import os
import pickle
import numpy as np

rootDir = '../data/training_cat/spmel'
dirName, subdirList, _ = next(os.walk(rootDir))
print('Found directory: %s' % dirName)

speakers = []
for speaker in sorted(subdirList):
    print('Processing speaker: %s' % speaker)

    _, _, fileList = next(os.walk(os.path.join(dirName,speaker)))
    # create file list
    for fileName in sorted(fileList):
        utterances = []
        utterances.append(speaker)
        utterances.append(int(speaker[-1]))
        utterances.append(os.path.join(speaker, fileName))
        utterances.append(os.path.join(speaker, speaker + '.npy'))
        speakers.append(utterances)
    
with open(os.path.join(rootDir, 'train.pkl'), 'wb') as handle:
    pickle.dump(speakers, handle)
