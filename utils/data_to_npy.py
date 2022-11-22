import numpy as np
import os
import wfdb

from filters import filter_bandpass


DATA_PATH = os.path.join('data', 'ptb-xl', 'records100')

def main():
    os.makedirs(os.path.join('data', 'ptb-xl', 'records100_filt'))
    for dir in os.listdir(DATA_PATH):
        print(dir)
        os.makedirs(os.path.join('data', 'ptb-xl', 'records100_filt', dir))
        for file in os.listdir(os.path.join(DATA_PATH, dir)):
            if file.endswith('.dat'):
                x = wfdb.rdsamp(os.path.join(DATA_PATH, dir, file[:-4]))
                y = filter_bandpass(x[0])
                # save to npy
                np.save(os.path.join('data', 'ptb-xl', 'records100_filt', dir, file.replace('dat', 'npy')), y)
                # save to csv
                # pd.DataFrame(y).to_csv(os.path.join('data', 'ptb-xl', 'records100_filt', dir, file.replace('dat', 'csv')), index = False, header=False)
                # save to dat / hea - does not work
                # wfdb.wrsamp(os.path.join('data', 'ptb-xl', 'records100_filt', '00000000'), p_signal=y, fs=100, fmt=['16']*12, units=['mV']*12, sig_name=['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'])
                # z = wfdb.rdsamp(os.path.join('data', 'ptb-xl', 'records100_filt', '00000000'))

if __name__ == '__main__':
    main()