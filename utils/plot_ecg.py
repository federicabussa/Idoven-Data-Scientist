'''


Adapted from: https://github.com/dy1901/ecg_plot
Standard of presentation: https://www.ahajournals.org/doi/10.1161/circulationaha.106.180200

'''

from math import ceil
from matplotlib import pyplot as plt 
from matplotlib.ticker import AutoMinorLocator
import numpy as np


def _set_clean_axis():
    plt.figure()
    plt.rcParams['xtick.bottom'] = False
    plt.rcParams['xtick.labelbottom'] = False
    plt.rcParams['ytick.left'] = False
    plt.rcParams['ytick.labelleft'] = False
    plt.close()

def plot_ecg(
        ecg, 
        record_name,
        # sample_rate    = 100, 
        # title          =  '12-leads ECG',
        lead_index     = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'], 
        lead_order     = None,
        columns        = 4,
        row_height     = 6,
        display_factor = 1,
        rhythm_strip = 'II'
    ):

    if ecg.shape[0] != 12:
        ecg = ecg.T
    sample_rate = int(record_name.split('/')[0][-3:])
    title = '12-leads ECG - PTB-XL record: '+record_name.split('/')[-1]

    lead_order = list(range(0,len(ecg)))
    secs  = ecg.shape[1]/sample_rate
    if secs > 10:
        ecg = ecg[:, :10*sample_rate]
    leads = len(lead_order)
    rows  = int(ceil(leads/columns))
    line_width = 0.5

    fig, ax = plt.subplots(figsize=(secs * display_factor, (rows) * row_height / 5 * display_factor))
    # display_factor = display_factor ** 0.5
    fig.subplots_adjust(
        hspace = 0, 
        wspace = 0,
        left   = 0,  # the left side of the subplots of the figure
        right  = 1,  # the right side of the subplots of the figure
        bottom = 0,  # the bottom of the subplots of the figure
        top    = 1
        )

    fig.suptitle(title, fontname = 'monospace', fontsize = 11 * display_factor)
    
    x_min = 0
    x_max = secs
    y_min = row_height/4 - (rows/2)*row_height -3 
    y_max = row_height/4

    color_major = (1,0,0)
    color_minor = (1, 0.8, 0.8)
    color_line  = (0,0,0.8)

    ax.set_xticks(np.arange(x_min,x_max,0.2))   
    ax.set_yticks(np.arange(y_min,y_max+0.5,0.5)) 

    ax.minorticks_on()
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    
    ax.grid(which='minor', linestyle='-', linewidth=0.5 * display_factor, color=color_minor, alpha = 0.8)
    ax.grid(which='major', linestyle='-', linewidth=0.5 * display_factor, color=color_major, alpha = 0.8)

    ax.set_ylim(y_min ,y_max + 1) 
    ax.set_xlim(x_min,x_max)

    for c in range(0, columns):
        for r in range(0, rows):
            if c * rows + r < leads:
                y_offset = -(row_height/2) * ceil(r%rows)
                x_offset = 0
                if c:
                    x_offset = secs/columns * c
                    # show separate line
                    ax.plot([x_offset, x_offset], [ecg[t_lead][0] + y_offset - 0.3, ecg[t_lead][0] + y_offset + 0.3], linewidth=line_width * display_factor, color=color_line)
                t_lead = lead_order[c * rows + r]
                step = 1.0/sample_rate
                # show lead name
                ax.text(x_offset + 0.07, y_offset - 0.5, lead_index[t_lead], fontname = 'monospace', fontsize=9 * display_factor)
                ax.plot(
                    np.arange(0, len(ecg[t_lead, :int(sample_rate*2.5)])*step, step) + x_offset, 
                    ecg[t_lead, int(2.5*sample_rate)*c:int(2.5*sample_rate)*(c+1)] + y_offset,
                    linewidth=line_width * display_factor * 1.5, 
                    color=color_line
                    )
    # ax.set_xticklabels(x_labels)
    # ax.set_yticklabels(y_labels)
    # ax.set_xlabel('Time [s]')
    # ax.set_ylabel('Voltage [mV]')
    x_offset = 0 
    y_offset = -(row_height/2) * ceil((r+1)%(rows+1)) # -9.5
    ax.text(x_offset + 0.07, y_offset + 1.3, 'Rhythm Strip: '+rhythm_strip, fontname = 'monospace', fontsize=9 * display_factor)
    rhythm_strip = lead_index.index(rhythm_strip)
    ax.plot(
        np.arange(0, len(ecg[rhythm_strip, :])/sample_rate, step), 
        ecg[rhythm_strip, :] + y_offset,
        linewidth=line_width * display_factor * 1.5, 
        color=color_line
        )

    ax.text(0.07, y_offset-1, '10mm/mV, 25mm/s', fontname = 'monospace', fontsize=9 * display_factor)

    plt.rcParams['xtick.bottom'] = False
    plt.rcParams['xtick.labelbottom'] = False
    plt.rcParams['ytick.left'] = False
    plt.rcParams['ytick.labelleft'] = False

    plt.show()
    
def plot_ecg_complete(
        ecg, 
        record_name,
        lead_index     = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'], 
        lead_order     = None,
        columns        = 2,
        row_height     = 4,
        display_factor = 1,
    ):

    if ecg.shape[0] != 12:
        ecg = ecg.T
    sample_rate = int(record_name.split('/')[0][-3:])
    title = '12-leads ECG - PTB-XL record: '+record_name.split('/')[-1]

    lead_order = list(range(0,len(ecg)))
    secs  = ecg.shape[1]/sample_rate
    if secs > 10:
        ecg = ecg[:, :10*sample_rate]
    leads = len(lead_order)
    rows  = int(ceil(leads/columns))
    line_width = 0.5

    fig, ax = plt.subplots(figsize=(secs * columns * display_factor, (rows) * row_height / 5 * display_factor))

    fig.subplots_adjust(
        hspace = 0, 
        wspace = 0,
        left   = 0,  # the left side of the subplots of the figure
        right  = 1,  # the right side of the subplots of the figure
        bottom = 0,  # the bottom of the subplots of the figure
        top    = 1
        )

    fig.suptitle(title, fontname = 'monospace')
    
    x_min = 0
    x_max = columns*secs
    y_min = row_height/4 - (rows/2)*row_height -1
    y_max = row_height/4

    color_major = (1,0,0)
    color_minor = (1, 0.8, 0.8)
    color_line  = (0,0,0.8)

    ax.set_xticks(np.arange(x_min,x_max,0.2))   
    ax.set_yticks(np.arange(y_min,y_max+0.5,0.5)) 

    ax.minorticks_on()
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    
    ax.grid(which='minor', linestyle='-', linewidth=0.5 * display_factor, color=color_minor, alpha = 0.8)
    ax.grid(which='major', linestyle='-', linewidth=0.5 * display_factor, color=color_major, alpha = 0.8)

    ax.set_ylim(y_min ,y_max + 1) 
    ax.set_xlim(x_min,x_max)

    for c in range(0, columns):
        for r in range(0, rows):
            if c * rows + r < leads:
                y_offset = -(row_height/2) * ceil(r%rows)
                x_offset = 0
                if c:
                    x_offset = secs * c
                    # show separate line
                    ax.plot([x_offset, x_offset], [ecg[t_lead][0] + y_offset - 0.3, ecg[t_lead][0] + y_offset + 0.3], linewidth=line_width * display_factor, color=color_line)
                t_lead = lead_order[c * rows + r]
                step = 1.0/sample_rate
                # show lead name
                ax.text(x_offset + 0.07, y_offset - 0.5, lead_index[t_lead], fontname = 'monospace', fontsize=9 * display_factor)
                ax.plot(
                    np.arange(0, len(ecg[t_lead, :])*step, step) + x_offset, 
                    ecg[t_lead, :] + y_offset,
                    linewidth=line_width * display_factor * 1.5, 
                    color=color_line
                    )

    ax.text(0.07, y_offset-1.5, '10mm/mV, 25mm/s', fontname = 'monospace', fontsize=9 * display_factor)

    plt.rcParams['xtick.bottom'] = False
    plt.rcParams['xtick.labelbottom'] = False
    plt.rcParams['ytick.left'] = False
    plt.rcParams['ytick.labelleft'] = False

    plt.show()
    
