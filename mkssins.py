#Imports
import numpy as np
import matplotlib.pylab as plt
import warnings
import time
import pickle
import sys
import katcali
import katcali.io as kio
import katcali.label_dump as kl
import katcali.diode as kd
from pathlib import Path
def good_ant(fname):

    """
    Fuction to retrieve list of good antennas.
    """

    data=kio.load_data(fname)
    bad_ants=kio.check_ants(fname)

    ants_good = []
    for i in np.array(kio.ant_list(data)):
        if i not in bad_ants:
            ants_good.append(i)
    else:
        print (str(i) + ' is bad')

    return ants_good
"""
Define a function that will access the data block and return visibility data, flags, noise diodes vector. 
"""
def visData(fname, ant, pol):
    """
    Parameters:
    ----------
    Fname: Path to observation block.

    Returns:
    -------
    vis, flags (SARAO prior flags), noise_diodes vector (nd)

    Note : Current function looks at one pol and one dish.
    """

    data = kio.load_data(fname)
    target, c0, band_ants, flux_model = kio.check_ants(fname)
    ants_good = good_ant(fname)
    data.select(ants=ant, pol=pol)
    recv = ant + pol
    print(recv)
    corr_id = kio.cal_corr_id(data, recv)

    assert(recv == data.corr_products[corr_id][0])
    assert(recv == data.corr_products[corr_id][1])

    print("Correlation ID:", corr_id, "Receiver:", recv)

    # Load visibilities and flags
    vis, flags = kio.call_vis(fname, recv)
    print("Shape of vis:", vis.copy().shape)
    vis_backup = vis.copy()

    ra, dec, az, el = kio.load_coordinates(data)
    ang_deg = kio.load_ang_deg(ra, dec, c0)
    ch_ref = 800
    timestamps, freqs = kio.load_tf(data)
    dp_tt, dp_ss, dp_f, dp_w, dp_t, dp_s, dp_slew, dp_stop = kl.cal_dp_label(data, flags, ant, pol, ch_ref, ang_deg)


    nd_on_time, nd_cycle, nd_set = kd.cal_nd_basic_para(fname)
    nd_on_edge, nd_off_edge = kd.cal_nd_edges(timestamps, nd_set, nd_cycle, nd_on_time)
    nd_ratio, nd_0, nd_1x = kd.cal_nd_ratio(timestamps, nd_on_time, nd_on_edge, data.dump_period)


    nd_t0, nd_t1x, nd_s0, nd_s1x, nd_t0_ca, nd_t0_cb, nd_t1x_ca, nd_t1x_cb = kl.cal_label_intersec(dp_tt, dp_ss, nd_0, nd_1x)

    return vis, nd_s0


def MaskedArrayVisibilityFlags(vis, flags_mask, nd_s0):
    """
    Reason of the masked array: Apply masks to noise diodes and bright RFI flags, so that they are not time differenced in the TOD array. Ensures that we are performing correct neighbouring time channel subtractions
    
    Parameters:
    ----------
    visibility, flags and nd_s0 from the visData Fuction.

    Returns:
    --------
    Visibility-Flags Masked Array
    """
    #vis, flags, nd_s0 = visData(fname)
    data0 = vis.copy


    nd_flags = np.ones_like(vis, dtype=bool)          # Empty mask, where all values are set to true. True is flagged data
    nd_flags[nd_s0, :] = False                        # Set the data with noise diodes removed to False so that this data is not flagged as bad data.  This is the scan only data.
    #other_flags = np.logical_or(flags, flags_mask)   # All other flags from the visData function. Boolean value is True.
    other_flags = flags_mask 

    allflags =  np.logical_or(nd_flags, other_flags)  # Apply logical operator or to combine the noise diode flags, and visilibity data at a specific stage flags. 

    data_masked = np.ma.masked_array(vis, mask=allflags, fill_value = np.nan)

    return data_masked

def SkySubtraction(data_masked):

    """
    Function Returns differencing of the  visibility masked array.
    """

    vis_ss =data_masked[1:,:] - data_masked[0:-1,:]
    visSS=vis_ss.filled()

    return visSS



def abba(array: np.ndarray):
    """Calculate ABBA dithering (interpolation) from a 1D array (time series).
    Performs 4 channel differencing of time.
    
    Parameters
    -------
    array: np.ndarray
        Numpy array of visibility as a function of time dump.
        Can be masked array (np.nan will be propagated)

    Returns
    -------
    out: ndarray after ABBA (shape N-3)
    """
    array = array.filled()
    vis_abba = (array[1:-2]+array[2:-1])/2 - (array[0:-3]+array[3:])/2
    return vis_abba




def plot_bandpass(x : np.ndarray, y : np.ndarray, label=None, Title =None, ylim : tuple = None, figsize=(8, 3), ax=None):
    """Plot bandpass (visibility vs frequency channel)"""
    if ax is None:  # Create a new figure and axes if not being passed in as a parameter
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.plot(x, y, label=label)
    if ylim is not None:
        ax.set_ylim(*ylim)
    return ax


def plot(x : np.ndarray, label=None, Title =None, ylim : tuple = None, figsize=(20, 6), ax=None):
    """Plot bandpass (visibility vs frequency channel)"""
    if ax is None:  # Create a new figure and axes if not being passed in as a parameter
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.plot(x, label=label)
    ax.legend()
    if ylim is not None:
        ax.set_ylim(*ylim)
        
    return ax



def plot_waterfall(x : np.ndarray,label=None,  Title =None, ylim : tuple = None, figsize=None, ax=None, vmax=None, interpolation= None):
    """Plot bandpass (visibility vs frequency channel)"""
    if ax is None:  # Create a new figure and axes if not being passed in as a parameter
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.imshow((x), label=label, vmax=vmax, interpolation=interpolation, cmap='viridis', aspect='auto')
    ax.set_title(Title)
    
    if ylim is not None:
        ax.set_ylim(*ylim)
    return ax


def plot_hist(x : np.ndarray, label=None, Title =None, ylim : tuple = None, figsize=None, ax=None, bins=None, alpha=None, density=True, color='r',edgecolor='black'):
    """Plot bandpass (visibility vs frequency channel)"""
    if ax is None:  # Create a new figure and axes if not being passed in as a parameter
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.hist(x, label=label,bins=bins, alpha=alpha, density=density, color=color, edgecolor=edgecolor)
    ax.legend()
    if ylim is not None:
        ax.set_ylim(*ylim)
        
    return ax



def ants_checked_L1(fname, path):
    obsfolder = Path(path)
    
    ants =[]
    obsblock_ant_pol = []
    for f in sorted(obsfolder.glob(fname+'_m*h*')):
        filename =  f.name.split('_')[0]
        antpol = f.name.split('1630519596_')[1]
        ant= antpol[0:4].split('h')
        ants.append(ant)
        pol = antpol[4:5]
        
       
    return(ants)
