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

from astropy.coordinates import SkyCoord
from astropy import units as u

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
    p_radec=np.loadtxt('radio_source2021.txt')
    
    dp_sb=dp_ss[0]
    dp_se=dp_ss[-1]
    
    p = SkyCoord(data.ra*u.deg,  data.dec*u.deg, frame='icrs')
    ang_lim=.5
    
    dp_ptr_list=kl.cal_ptr_mask(p,p_radec,nd_s0, dp_sb,dp_se,ang_lim)


    return vis, nd_s0, dp_ptr_list, flags

def MaskedArrayVisibilityFlags(vis, nd_s0, pipeline_flags =  None, sarao_flags = None, pointsource_flags=None):
    
    """This function applies masks to noise diodes and bright RFI flags, so that they are not time differenced in the TOD array. Ensures that we are performing correct neighbouring time channel subtractions
    
    Parameters:
    ----------
    visibility, flags and nd_s0 from the visData Fuction.

    Returns:
    --------
    Visibility-Flags Masked Array
    """
    #vis, flags, nd_s0 = visData(fname)

    data0 = vis.copy()

    # Step 1: Create noise diode flags
    nd_flags = np.ones_like(vis, dtype=bool)
    nd_flags[nd_s0, :] = False  # Noise diode off regions are not flagged (False)

    # Step 2: Combine SARAO flags if provided
    if sarao_flags is not None:
        all_flags = np.logical_or(nd_flags, sarao_flags)
    else:
        all_flags = nd_flags

    # Step 3: Combine pipeline flags if provided
    if pipeline_flags is not None:
        all_flags = np.logical_or(all_flags, pipeline_flags)

    # Step 4: Combine point source flags if provided
    if pointsource_flags is not None:
        point_source_mask = np.ones_like(vis, dtype=bool)
        point_source_mask[pointsource_flags, :] = False
        all_flags = np.logical_or(~point_source_mask, all_flags)

    # Step 5: Create masked array
    data_masked = np.ma.masked_array(vis, mask=all_flags, fill_value=np.nan)

    return data_masked

def SkySubtraction(data_masked):

    """
    Function Returns differencing of the  visibility masked array.
    """

    vis_ss =data_masked[1:,:] - data_masked[0:-1,:]
    visSS=vis_ss.filled()

    return visSS



def abba(array):
    """Calculate ABBA dithering (interpolation) from a 2D array (time, freq).
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
    ax.set_ylim()
    if xlim is not None:
        ax.set_xlim(*xlim)
        
    return ax

def plot(x : np.ndarray, label=None, Title =None, ylim : tuple = None, figsize=(20, 6), ax=None, marker=None, linestyle='-', xlabel=None, ylabel=None, xlim=None):
    """Plot bandpass (visibility vs frequency channel)"""
    if ax is None:  # Create a new figure and axes if not being, passed in as a parameter
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.plot(x,label=label,marker = '', linestyle='-')
    ax.set_title(Title)
    ax.set_xlabel(xlabel=xlabel)
    ax.set_ylabel(ylabel=ylabel)
    ax.legend()
    ax.grid(color='grey', which='both', lw=0.1)
    
def plot_x_y(x : np.ndarray,y:np.ndarray, label=None, Title =None, ylim : tuple = None, figsize=(20, 6), ax=None, marker=None, linestyle='-', xlabel=None, ylabel=None, xlim=None):
    """Plot bandpass (visibility vs frequency channel)"""
    if ax is None:  # Create a new figure and axes if not being, passed in as a parameter
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.plot(x,y,label=label,marker = '', linestyle='-')
    ax.set_title(Title)
    ax.set_xlabel(xlabel=xlabel)
    ax.set_ylabel(ylabel=ylabel)
    ax.legend()
    ax.grid(color='grey', which='both', lw=0.1)


    
    if ylim is not None:
        ax.set_ylim(*ylim)

    if xlim is not None:
        ax.set_xlim(*xlim)
        
    return ax


def plot_rfi_stats(x : np.ndarray,y:np.ndarray, label=None, Title =None, ylim : tuple = None, figsize=(20, 6), ax=None, marker=None, linestyle=None, xlabel=None, ylabel=None, xlim=None, linewidth=None, color=None):
    if ax is None:  # Create a new figure and axes if not being, passed in as a parameter
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    plt.step(x,y,label=label, linestyle=linestyle, linewidth=linewidth, color=color)
    ax.set_title(Title)
    ax.set_xlabel(xlabel=xlabel)
    ax.set_ylabel(ylabel=ylabel)
    ax.legend()
    ax.grid(color='grey', which='both', lw=0.1)


    
    if ylim is not None:
        ax.set_ylim(*ylim)

    if xlim is not None:
        ax.set_xlim(*xlim)
        
    return ax
def plot_waterfall(x,label=None,  Title =None, ylim : tuple = None, figsize=None, ax=None, vmax=None,vmin=None, interpolation= None, norm=None, cmap=None, xlabel=None, ylabel=None, clabel=None):

    if ax is None:  # Create a new figure and axes if not being passed in as a parameter
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    im = ax.imshow(x, label=label, vmax=vmax, vmin=vmin, interpolation=interpolation, cmap='viridis', aspect='auto', norm=norm)
    ax.set_title(Title)
    ax.set_xlabel(xlabel=xlabel)
    ax.set_ylabel(ylabel=ylabel)
    #fig.colorbar(im, ax=ax) 
    plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),ax=ax, label=clabel) 
    
    if ylim is not None:
        ax.set_ylim(*ylim)
    return ax


def ants_checked_L1(fname, path, pol):
    ants =[]
    obsblock_ant_pol = []
    obsfolder = Path(path)
    for f in sorted(obsfolder.glob(fname+f'_m*{pol}*')):
        filename =  f.name.split('_')[0]
        #print(filename)
        antpol = f.name.split(fname+'_')[1]
        #print(antpol)
        ant= antpol[0:4].split(f'{pol}')
        #print(ant)
        ants.append(ant[0])
        #print(ants)
        pol = antpol[4:5]
    return ants
    
def cal_zscore(SS_all_spectrums, ssins_flags = None):

    """ This fuction calculates the z-scores.
    Parameters:
    ----------
    SS_all_spectrums: Array of all antenna sky subtraction spectrums, has the shape (no.Dishes, Time, Frequency)
    
    Returns:
    --------

    z_score: 2D array of the normalised Incoherent Spectrum.
    
    """
    if ssins_flags is None:
        IncoherentSpectrum = np.nanmean(np.abs(SS_all_spectrums), axis=0)
    
    else:
        new_ins = np.ma.array(data =  np.nanmean(np.abs(SS_all_spectrums), axis=0), mask = ssins_flags, fill_value=np.nan)
        IncoherentSpectrum = new_ins.filled()
        
    no_dishes = SS_all_spectrums.shape[0]
    c_fold = np.pi / 2 - 1                            #Auto C_fold Ratio
    meanEst = np.nanmean(IncoherentSpectrum, axis=0) #Time axis averaged MeanEstimate
    std_sq = c_fold*meanEst**2 
   
    z_score = ((IncoherentSpectrum-meanEst))/np.sqrt(std_sq/no_dishes)
   
    
    return z_score     

def mask_to_flags(nd_s0, zscore_mask, ants, pipeline_flags=None):

    """This function will return the flags for the raw, non-time differenced data. This function propagates the masks of the outliers found in the z-scores to flags in the Time-Ordered Data (non-time differenced)
    
    Parameters:
    -----------
    zscore_mask: 2D (t, f), boolean array of the outliers for a specific thresholding. (True - Flag Data , False -  Unflagged Data)
    
    Returns:
    -------
    zscore_flags_dict: Returns the flags as a dictionary, can be combined with the older pipline flags (dictionary) 


    """
    shape = list(zscore_mask.shape)
    flags_new = np.zeros([shape[0] + 3] + shape[1:], dtype=bool)  #(t, f) ----> (time, frequency)  # expanded the dims
    flags_new[:-3] = zscore_mask
    flags_new[3:] = np.logical_or(flags_new[3:], flags_new[:-3])
    #nd_flags= stacked_flags(nd_flags, ants)
    nd_flags = np.ones_like(flags_new, dtype=bool)
    nd_flags[nd_s0, :] = False 
    if pipeline_flags is not None:
        pipeline_flags = stacked_flags(pipeline_flags, ants)
        flags = np.logical_or(nd_flags, pipeline_flags)
        new_flags = np.logical_or(flags, flags_new)
    else:
        new_flags = np.logical_or(flags_new, nd_flags)
                
    return new_flags


def stacked_flags(pipeline_flags, ants):
    """This function create a combined mask by summing the flags accross recievers and taking a relevant score
    Parameters:
    -----------
    score == 59
    pipeline_flags: dict of the pipelines flags for each receiver in the observation block
    
    Return:
    -------
    stacked_flag: 2D nd.array (t,f)
    
    """
    
    stacked_flags = np.stack(list(pipeline_flags.values()), axis=0)
    stacked_int_flags = stacked_flags.astype(int)
    stacked_score= np.sum(stacked_int_flags, axis=0)
    stacked_flag = ((stacked_score.astype(float) >= len(ants)-1)) 
    return stacked_flag

def pipeline_flags(nd_s0, ants, nd_flags, pipeline):
    
    pipeline_flags= stacked_flags(pipeline, ants)
    nd_flags= stacked_flags(nd_flags, ants)
    nd_flags = np.ones_like(pipeline_flags, dtype=bool)
    nd_flags[nd_s0, :] = False 

    
    pipeline_flags =  np.logical_or(nd_flags, pipeline_flags)
    return pipeline_flags

    
def mask_all_fchan_tchan(z_flags, c_t, c_f):
    z_flags_all = z_flags.copy()
    

    for i in range(z_flags_all.shape[1]):      
        num_flagged = np.sum(z_flags[:, i]==True)
    
        c = num_flagged / z_flags_all.shape[0]
        
        if c > c_f:
            z_flags_all[:, i] = True  
    
    for i in range(z_flags_all.shape[0]):
        num_flagged = np.sum(z_flags[i, :]==True)

        c = num_flagged / z_flags_all.shape[1]
    
        if (c > c_t):
            z_flags_all[i, :] = True
    return z_flags_all




def dict_to_array(dictionary):
    array = []
    for chan, flag in dictionary.items():
        array.append(flag)
    array = np.array(array)
    return array



def count_flags(flag, flag_extra=None, nreceiver=None): 

    """This function calculates the flag percentage accross freqeuncy and time channels.
    int(True) = 1
    int(False) = 0
    This fuction takes the sum for each access using these Boolean 
    Parameters:
    flag: Boolean Array, with shape (nrec, Time, Frequency)
    nreceiver: int
    """
    flag_counts = {}
    if flag_extra is None and flag.ndim == 3:
        
        flags_fchan = 100*(np.sum(flag, axis=(0,1))/(flag.shape[1]*flag.shape[0])) # nrec and frequency channel by summing over the time dimension, output (3647,)
        flags_tchan = 100*(np.sum(flag, axis=(0,2))/(flag.shape[2]*flag.shape[0])) # nrec and time channel by summing over the frequency dimension, output (f,)
        flag_counts = {'flags_fchan': flags_fchan, 'flags_tchan': flags_tchan}
        
    elif flag_extra is not None and nreceiver is not None:
        if flag_extra.ndim == 2:
            flag_3D = np.tile(flag_extra[np.newaxis, :, :], (nreceiver, 1, 1)) # Expand flag to 3D with nreceiver
            flag_combined = np.logical_or(flag_3D, flag)
            flags_fchan = 100*(np.sum(flag_combined, axis=(0,1))/(flag_combined.shape[1]*flag_combined.shape[0]))
            flags_tchan = 100*(np.sum(flag_combined, axis=(0,2))/(flag_combined.shape[2]*flag_combined.shape[0]))  
            flag_counts = {'flags_fchan': flags_fchan, 'flags_tchan': flags_tchan}
        
    elif flag.ndim == 2 and nreceiver is not None:
        flag_3D = np.tile(flag[np.newaxis, :, :], (nreceiver, 1, 1)) # Expand flag to 3D with nreceiver
        flag_combined = np.logical_or(flag_3D, flag)
        flags_fchan = 100*(np.sum(flag_combined, axis=(0,1))/(flag_combined.shape[1]*flag_combined.shape[0]))
        flags_tchan = 100*(np.sum(flag_combined, axis=(0,2))/(flag_combined.shape[2]*flag_combined.shape[0]))  
        flag_counts = {'flags_fchan': flags_fchan, 'flags_tchan': flags_tchan}
        
        
    else:
        raise Exception('nreceiver is required') # raise error that nreceiver is required


    return flag_counts


def l1_flags_dict_to_array(fname, pol, path):
    
    ants = ants_checked_L1(fname, path, pol)
    l1_flags_dict = {}
    mask_dir = Path('/idia/projects/hi_im/raw_vis/MeerKLASS2021/level1/mask/checked/')
    
    for dish in ants:
        
        try:
            with open(mask_dir / f'{fname}_{dish}_mask2', 'rb') as f:
                d3 = pickle.load(f)
                print (f'mask2 loaded for dish {dish}')
        except(Exception):
            with open(mask_dir / f'{fname}_{dish}_mask', 'rb')as f:
                d3 = pickle.load(f)
                print (f'mask loaded for dish {dish}')
        mask_flags=d3['mask']
        l1_flags_dict[dish]  = mask_flags
        l1_flags = dict_to_array(l1_flags_dict)
    return l1_flags, len(ants)



    

def l4_flags_dict_to_array(fname, pol, path):
    ants = ants_checked_L4(fname, path, pol)
    l4_flags_dict = {}
    mask_dir = Path('/idia/projects/hi_im/raw_vis/MeerKLASS2021/level4/mask/')
    for dish in ants:
        try:
            with open(mask_dir / f'{fname}_{dish}_level4_mask2', 'rb') as f:
                d3 = pickle.load(f)
                print(d3.keys())
        except(Exception):
            try:
                with open(mask_dir / f'{fname}_{dish}_level4_mask', 'rb') as f:
                    d3 = pickle.load(f)
                    print(f'mask loaded for dish {dish}')
        
            except FileNotFoundError:
                print(f'No mask file found for dish {dish}')
    
                  
        try:
            mask_flags = d3['Inten_mask']
        except KeyError:
            print("no'Inten_mask' found in d3")
            mask_flags = d3['mask']
            
        l4_flags_dict[dish]  = mask_flags
        l4_flags = dict_to_array(l4_flags_dict)
    return l4_flags

def freqs_array(fname):
    data=kio.load_data(fname)
    freqs_ = data.channel_freqs
    chans = data.channels
    return freqs_, chans


def ants_checked_L4(fname, path, pol):
    ants =[]
    obsblock_ant_pol = []
    obsfolder = Path(path)
    for f in sorted(obsfolder.glob(fname+f'_m*')):
        filename =  f.name.split('_')[0]
        #print(filename)
        antpol = f.name.split(fname+'_')[1]
        #print(antpol)
        ant= antpol[0:4].split(f'{pol}')
        #print(ant)
        ants.append(ant[0])
        #print(ants)
        pol = antpol[4:5]
    return ants
def bins_bdis(smin, smax, rfi_max, sbin_num, rfi_bin_num):
    # Sky portion
    smin = smin
    smax = smax
    sbin_num = sbin_num
    sbins = np.linspace(smin, smax, sbin_num)
    
    # RFI portion
    rfi_min = smax
    rfi_max = rfi_max
    rfi_bin_num = rfi_bin_num
    rfi_bins = np.logspace(np.log10(smax), np.log10(rfi_max), num=rfi_bin_num)
    
    # Combine the bins
    bins_set = np.hstack([sbins, rfi_bins])
    return bins_set


def temp_to_flux(Temperature, freqs):
    from astropy import units as u
    from astropy import constants as c
    c_ = c.c
    c_.value
    kb = c.k_B
    kb_ = kb.to('W*s/K')
    kb_.value
    flux_density = np.zeros(Temperature.shape)
    for i, freq in enumerate(freqs):
        D=13.6
        lambda_ = c_.value / freq  # meters
        omega_b = (1.22* lambda_/D)**2
        for time in range(Temperature.shape[0]):
            flux_density[time, i] = (2 * kb.value * Temperature[time, i] / lambda_**2)* omega_b/1e-26  # this is in W/hz*m^2 --->  multiply by factor 10^-26 --->  Jy
            
    return flux_density



def plot_bd(fname, data = None, cd_data = None, amp_data=None, cd=None, amp_rfi=None, cd_rfi=None, amp_norfi=None, cd_norfi=None, label= None, ax=None, figsize= None):
    if ax is None:  # Create a new figure and axes if not being, passed in as a parameter
        fig, ax = plt.subplots(1, 1, figsize=figsize)
 
    if amp_data and cd and amp_rfi and cd_rfi and amp_norfi and cd_norfi is not None:
    
        plt.step(amp_norfi[0:-1], cd_norfi, label='Data not detected as RFI', linestyle='-')
        plt.step(amp_rfi[0:-1], cd_rfi, label='RFI', linestyle='-')
        plt.step(amp_data[0:-1], cd, label='Data', linestyle='-')
        ax.set_xlabel('Flux (Jy)')
        ax.set_ylabel('Flux Count')
        ax.legend()
        plt.xscale('log')
        plt.yscale('log')
        plt.title(f'Brightness Distribution of Observation {fname}')
        plt.xlim((1e3, 1e6))
        plt.show()
    else:
        plt.step(data[0:-1], cd_data, label=label, linestyle='-')
        ax.set_xlabel('Flux (Jy)')
        ax.set_ylabel('Flux Count')
        ax.legend()
        plt.title(f'Brightness Distribution of Observation {fname}')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlim((1e3, 1e6))

         
    return ax

    