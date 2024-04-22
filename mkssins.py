#Imports
import pickle
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib as mpl
import katcali.io as kio
import katcali.label_dump as kl
import katcali.diode as kd
from pathlib import Path

def good_ant(fname):

    """ This fuction retrieves list of good antennas from the observation.
    Parameters:
    ----------
    Fname : Path to observation block.

    Returns:
    --------
    ants_good : List of antennas.
    
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


def visData(fname, ant, pol, verbos=False):
    
    """ This function  will access the data block and return visibility data, flags, noise diodes vector. 
    
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
    if verbos:  # i.e. verbos=True
        print(recv)
    corr_id = kio.cal_corr_id(data, recv)

    assert(recv == data.corr_products[corr_id][0])
    assert(recv == data.corr_products[corr_id][1])

    if verbos:
        print("Correlation ID:", corr_id, "Receiver:", recv)

    # Load visibilities and flags
    vis, flags = kio.call_vis(fname, recv)
    if verbos:
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


def MaskedArrayVisibilityFlags(vis, flags, nd_s0):
    
    """This function applies masks to noise diodes and bright RFI flags, so that they are not time differenced in the TOD array. Ensures that we are performing correct neighbouring time channel subtractions
    
    Parameters:
    ----------
    visibility, flags and nd_s0 from the visData Fuction.

    Returns:
    --------
    Visibility-Flags Masked Array
    """
    #vis, flags, nd_s0 = visData(fname)
    data0 = vis.copy


    nd_flags = np.ones_like(vis, dtype=bool)      
    # Empty mask, where all values are set to true. True is flagged data
    nd_flags[nd_s0, :] = False                        # Set the data with noise diodes removed to False so that this data is not flagged as bad data.  This is the scan only data.
    #other_flags = np.logical_or(flags_L0, flags_L1)   # All other flags from the visData function. Boolean value is True.
    other_flags = flags 

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

def plot_hist(x : np.ndarray, label=None, Title =None, xlim : tuple = None, figsize=None, ax=None, bins=None, alpha=None, density=None, color=None ,edgecolor=None, histtype=None, xlabel=None, ylabel=None):
   
    if ax is None:  # Create a new figure and axes if not being passed in as a parameter
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.hist(x, label=label,bins=bins, alpha=alpha, density=density, color=color, edgecolor=edgecolor, histtype=histtype)
    ax.legend()
    ax.set_title(Title)
    ax.set_xlabel(xlabel=xlabel)
    ax.set_ylabel(ylabel=ylabel)
    if xlim is not None:
        ax.set_xlim(*xlim)
        
    return ax


def plot(x : np.ndarray, label=None, Title =None, ylim : tuple = None, figsize=(20, 6), ax=None, marker=None, linestyle='-'):
    """Plot bandpass (visibility vs frequency channel)"""
    if ax is None:  # Create a new figure and axes if not being, passed in as a parameter
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.plot(x, label=label,marker = '', linestyle='-')
    ax.set_title(Title)
    ax.legend()

    
    if ylim is not None:
        ax.set_ylim(*ylim)
        
    return ax



def plot_waterfall(x,label=None,  Title =None, ylim : tuple = None, figsize=None, ax=None, vmax=None,vmin=None, interpolation= None, norm=None, cmap=None, xlabel=None, ylabel=None, clabel=None):

    if ax is None:  # Create a new figure and axes if not being passed in as a parameter
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    im = ax.imshow(x, label=label, vmax=vmax, vmin=vmin, interpolation=interpolation, cmap='viridis', aspect='auto', norm=norm)
    ax.set_title(Title)
    ax.set_xlabel(xlabel=xlabel)
    ax.set_ylabel(ylabel=ylabel)
    #fig.colorbar(im, ax=ax) 
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),ax=ax, label=clabel) 
    
    if ylim is not None:
        ax.set_ylim(*ylim)
    return ax



def ants_checked_L1(fname, path):
    ants =[]
    obsblock_ant_pol = []
    obsfolder = Path(path)
    for f in sorted(obsfolder.glob(fname+'_m*h*')):
        filename =  f.name.split('_')[0]
        #print(filename)
        antpol = f.name.split(fname+'_')[1]
        #print(antpol)
        ant= antpol[0:4].split('h')
        ants.append(ant)
        pol = antpol[4:5]
    
       
    return(ants)


def cal_zscore(SS_all_spectrums):

    """ This fuction calculates the z-scores.
    Parameters:
    ----------
    SS_all_spectrums: Array of all antenna sky subtraction spectrums, has the shape (no.Dishes, Time, Frequency)
    
    Returns:
    --------

    z_score: 2D array of the normalised Incoherent Spectrum.
    
    """
    
    IncoherentSpectrum = np.nanmean(np.abs(SS_all_spectrums), axis=0)
    no_dishes = SS_all_spectrums.shape[0]
    c_fold = np.pi / 2 - 1                            #Auto C_fold Ratio
    meanEst = np.nanmean(IncoherentSpectrum, axis=0) #Time axis averaged MeanEstimate
    std_sq = c_fold*meanEst**2 
   
    z_score = ((IncoherentSpectrum-meanEst))/np.sqrt(std_sq/no_dishes)
   
    
    return z_score                                   #returns 2D array of the z scores
    
