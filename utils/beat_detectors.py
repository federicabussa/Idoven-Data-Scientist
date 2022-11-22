from ecgdetectors import Detectors
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering

'''
ecg = x
record_name = rec

'records500/16000/16489_hr'
'''

CHANNELS = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
METHODS = ['Elgendi et al (Two average)', 'Kalidas & Tamil (Wavelet transform)', 'Engzee', 'Christov', 'Hamilton', 'Pan Tompkins', 'WQRS']

def detect_Rpeaks(
        ecg, 
        record_name,
        leads = 'all', 
        methods = 'all'
    ):
    sample_rate = int(record_name.split('/')[0][-3:])
    detectors = Detectors(sample_rate)
    res = {}
    if type(leads) == str and leads != 'all':
        leads = [leads]
    if type(methods) == str and methods != 'all':
        methods = [methods]
    for lead in CHANNELS:
        if lead in leads or leads == 'all':
            res[lead] = {}
            for det_name, det_method in detectors.get_detector_list():
                if det_name in methods or methods == 'all':
                    try:
                        res[lead][det_name] = det_method(ecg[:,CHANNELS.index(lead)])
                    except:
                        continue
    return res


def plot_annotations(ecg, res, record_name):
    sns.set_theme()
    for ch in res.keys():    
        plt.figure(figsize = (15, 8)), plt.plot(ecg[:,CHANNELS.index(ch)]), 
        plt.title(record_name+' - lead '+ch)
        for det_name, det_res in res[ch].items():
            plt.plot(det_res, ecg[det_res, CHANNELS.index(ch)], 'o', label = det_name)
        plt.xticks([0, 1000, 2000, 3000, 4000, 5000], ['0', '2', '4', '6', '8', '10'])
        plt.xlabel('Time [sec]')
        plt.ylabel('Voltage [mV]')
        plt.grid('on')
        plt.legend(bbox_to_anchor=(1.01, 1.0), loc='upper left')
        plt.tight_layout()
        plt.show()


def _plot_cluster_annot(ecg, series, clustering):
    dict_colors = {
        0: '#1f77b4',
        1: '#ff7f0e',
        2: '#2ca02c',
        3: '#9467bd',
        4: '#e377c2',
        5: '#17becf',
        6: 'k',
        7: 'r',
        8: 'g',
        9: 'b',
        10:'c',
        11: 'm',
        12: 'plum',
        13: '#1f27b4',
        14: '#ff1f0e',
        15:'#2ca02c',
        16:'#d69728',
        17:'#9497bd',
        18:'#8c564b',
        19:'#e377c2',
        20:'#7f7f5f',
        21:'#bcbd72',
        22:'#17becf',
        23:'#1a55FF',
        24: 'plum',
        25: 'mediumseagreen',
        26: 'indianred',
        27: 'lightgoldenrodyellow'
    }
    plt.figure(figsize = (17,4)), 
    plt.plot(ecg, alpha=0.6),
    for label in range(np.amax(clustering.labels_)+1):
        cm_beat_samples = np.array([x[0] for i, x in enumerate(series) if clustering.labels_[i]==label])
        cm_beat_values = np.asarray([ecg[sample] for sample in cm_beat_samples])
        plt.plot(cm_beat_samples, cm_beat_values, dict_colors[label%len(dict_colors)], linestyle='None', marker = 'o')
    plt.grid('on')
    plt.title('Clusterized Annotations')
    plt.show()


def merge_multiple_detections_on_single_channel(ecg, res, record_name, channel):

    signal = ecg[:, CHANNELS.index(channel)]
    res_detectors = res[channel]

    sample_rate = int(record_name.split('/')[0][-3:])

    series = []
    
    for i, (_, v) in enumerate(res_detectors.items()):
        series+=[(ind, i) for ind in v] 

    th_r = np.median([max(abs(signal[n:n+sample_rate*2])) for n in range(0, len(signal), sample_rate*2)])
    th_range = np.median([max(signal[n:n+sample_rate*2])-min(signal[n:n+sample_rate*2]) for n in range(0, len(signal), sample_rate*2)])

    clustering = AgglomerativeClustering(n_clusters = None, affinity='manhattan', distance_threshold=sample_rate/4, linkage='average').fit(np.array(series)) # 300 is chosen empirically ...

    _plot_cluster_annot(signal, series, clustering)

    r_peaks = []
    th = int(sample_rate/5) # before:100
    not_validated = []

    for label in range(np.amax(clustering.labels_)+1):
        beats = [x for i, x in enumerate(series) if clustering.labels_[i]==label]
        if True: # 'simplified' post-processing
            if len(beats)<4:
                continue
            min_cluster_sample = min(beats)[0]
            max_cluster_sample = max(beats)[0]
            # if segment between min and max sample is greater than 100 ms (stamdard qrs duration is 120 ms) 
            # & in some portions, the signal amplitude reaches the max amplitude 
            if max_cluster_sample - min_cluster_sample > sample_rate / 5 and any(signal[min_cluster_sample:max_cluster_sample]>th_r*0.7):
                r_peaks.append(np.argmax(signal[min_cluster_sample:max_cluster_sample])+min_cluster_sample)
            elif max_cluster_sample - min_cluster_sample <= sample_rate / 5:
                start = min_cluster_sample-th if min_cluster_sample-th>0 else 0
                end = max_cluster_sample+th if max_cluster_sample+th<=len(signal) else len(signal)
                if any(signal[start:end]>th_r*0.7):
                    r_peaks.append(np.argmax(signal[start:end])+start)
        r_peaks.sort()

    plt.figure(figsize=(14, 8)), plt.plot(signal), plt.plot(r_peaks, signal[r_peaks], 'o'), plt.title(record_name+' - final detection after post-processing'), plt.show()

    return
        