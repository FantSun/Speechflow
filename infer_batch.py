# demo converting original spectrograms to constructed spectrograms with alternative components.
# here pickle file is defined with 3-D arrays of log-Mels with deltas and delta-deltas mentioned in ACRNN.
import os
import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
from hparams import hparams
from utils import pad_seq_to_2
from utils import quantize_f0_numpy
from model import Generator_3 as Generator
from model import Generator_6 as F0_Converter
import python_speech_features as ps

def load_data(in_dir):
    f = open(in_dir,'rb')
    train_data,train_f0,train_emotion,train_spker,test_data,test_f0,test_emotion,Testseg_emotion,test_spker,valid_data,valid_f0,valid_emotion,Validseg_emotion,valid_spker,pernums_test,pernums_valid = pickle.load(f, encoding='bytes')
    # you should define the pickle file by youself, in which, "train_data,train_f0,train_spker,test_data,test_f0,test_spker,valid_data,valid_f0,valid_spker" are necessary, 'xxxx_data's are 3-D arrays of log-Mels with deltas and delta-deltas to be converted, 'xxxx_f0's are corresponding f0s, 'xxxx_spker's are corresponding speaker labels. Other values could be ignored.
    return train_data,train_f0,train_spker,test_data,test_f0,test_spker,valid_data,valid_f0,valid_spker

device = 'cuda:0' # 'cuda:i' for gpu i or 'cpu' for cpu
condition = 'REC'
# select condition from "REC, NR, NF, NC, NRF, NRC, NFC, NRFC"
# REC: reconstruction
# NR: no rhythm
# NF: no pitch
# NC: no content
# NRF: only content
# NRC: only pitch
# NFC: only thythm
# NRFC: no rhythm, pitch and content

epoch = 200000 # training epoch of model
length = 192 # length of 

G = Generator(hparams).eval().to(device)
g_checkpoint = torch.load('run/models/' + str(epoch) + '-G.ckpt', map_location=lambda storage, loc: storage)
G.load_state_dict(g_checkpoint['model'])

# define a dictionary between speaker labels and d-vectors
# here you should provide a 32-dim d-vector for each speaker as its timbre embedding by a pre-trained d-vector model
dvecDir = 'data/test_dvec/dvector_test.npz'
spk2dvec = {}
spk_dvec = np.load(dvecDir)['spkers']
dvecs = np.load(dvecDir)['feats']
for i in range(spk_dvec.shape[0]):
    spk2dvec[spk_dvec[i]] = dvecs[i]

path_result = "./results/" + condition + '_' + str(epoch)
train_data,train_f0,train_spker,test_data,test_f0,test_spker,valid_data,valid_f0,valid_spker = load_data('./IEMOCAP192.pkl') # load data

if not os.path.exists(path_result):
    os.makedirs(path_result)

for sets in ['train', 'test', 'valid']:

    if sets == 'train':
        sets_data0 = train_data
        sets_f0 = train_f0
        sets_spker = train_spker[:, 0]
    elif sets == 'test':
        sets_data0 = test_data
        sets_f0 = test_f0
        sets_spker = test_spker[:, 0]
    elif sets == 'valid':
        sets_data0 = valid_data
        sets_f0 = valid_f0
        sets_spker = valid_spker[:, 0]

    file_result = os.path.join(path_result, sets + '.npy')

    sets_num = len(sets_data0)
    sets_data = np.empty((sets_num,length,80,3),dtype = np.float32)
    specs = sets_data0[:, :, :, 0] # select log-Mels in datasets besides their deltas and delta-deltas

    for i in range(sets_num):
        x_org = specs[i]
        f0_org = sets_f0[i]
        print(i, x_org.shape, f0_org.shape, sets_spker[i])
        len_org = x_org.shape[0]
        emb_org = torch.from_numpy(spk2dvec[sets_spker[i]]).to(device).reshape(1, 32)
        uttr_org_pad, len_org_pad = pad_seq_to_2(x_org[np.newaxis,:,:], length)
        uttr_org_pad = torch.from_numpy(uttr_org_pad).to(device)
        f0_org_pad = np.pad(f0_org, (0, length-len_org), 'constant', constant_values=(0, 0))
        f0_org_quantized = quantize_f0_numpy(f0_org_pad)[0]
        f0_org_onehot = f0_org_quantized[np.newaxis, :, :]
        f0_org_onehot = torch.from_numpy(f0_org_onehot).to(device)
        uttr_f0_org = torch.cat((uttr_org_pad, f0_org_onehot), dim=-1)
        
        spect_vc = []
        with torch.no_grad():
            if condition == 'REC':
                x_identic_val = G(uttr_f0_org, uttr_org_pad, emb_org)
            if condition == 'NR':
                x_identic_val = G(uttr_f0_org, uttr_org_pad * 0.0, emb_org)
            if condition == 'NF':
                x_identic_val = G(torch.cat((uttr_org_pad, f0_org_onehot * 0.0), dim=-1), uttr_org_pad, emb_org)
            if condition == 'NU':
                x_identic_val = G(uttr_f0_org, uttr_org_pad, emb_org * 0.0)
            if condition == 'NRF':
                x_identic_val = G(torch.cat((uttr_org_pad, f0_org_onehot * 0.0), dim=-1), uttr_org_pad * 0.0, emb_org)
            if condition == 'NRC':
                x_identic_val = G(torch.cat((uttr_org_pad * 0.0, f0_org_onehot), dim=-1), uttr_org_pad * 0.0, emb_org)
            if condition == 'NFC':
                x_identic_val = G(torch.cat((uttr_org_pad * 0.0, f0_org_onehot * 0.0), dim=-1), uttr_org_pad, emb_org)
            if condition == 'NRFC':
                x_identic_val = G(torch.cat((uttr_org_pad * 0.0, f0_org_onehot * 0.0), dim=-1), uttr_org_pad * 0.0, emb_org)
                
            uttr_trg = x_identic_val[0, :len_org, :].cpu().numpy()
    
            print(uttr_trg.shape)
            sets_data[i,:,:,0] = uttr_trg # log-Mels
            delta1 = ps.delta(uttr_trg, 2) # deltas
            sets_data[i,:,:,1] = delta1
            delta2 = ps.delta(delta1, 2) # delta-deltas
            sets_data[i,:,:,2] = delta2

    np.save(file_result, sets_data)
