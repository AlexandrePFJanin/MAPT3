# -*- coding: utf-8 -*-
"""
@author: Alexandre JANIN
@aim:    Tools for Plate Processing and Analysis

"""

# External dependencies:
import numpy as np
import matplotlib.pyplot as plt


# ----------------- FUNCTIONS -----------------


def clustering(threshold,x,y,z):
    """Function returning a spatial clustering of scattered data according
    to a given critial distance threshold. Function processing ndarray data
    and a cascading (top-down) algorithm to remain efficient even on a large
    set of input data.

    Args:
        threshold (int/float): Critical distance threshold. If the L2 distance between
                               two points is above this value, the two points are considered
                               as from a different cluster.
        x (ndarray): x coordinates (flatten) of points
        y (ndarray): y coordinates (flatten) of points
        z (ndarray): z coordinates (flatten) of points

    Returns:
        list: list of list representing the ID of input points and the resulting clustering
    """
    
    nod  = len(x)
    myPID   = np.arange(nod)
    todo    = np.ones(nod,dtype=bool)
    cluster = np.zeros(nod,dtype=np.int32)

    ccid   = 1 # current cluster ID : start at 1, very important, 0 means that not yet in a cluster
    while np.count_nonzero(todo) != 0:

        m_cluster = cluster > 0
        mask = m_cluster * todo
        if np.count_nonzero(mask) == 0: # pas de cluster en cours
            ptID = 0  # takes the first point of the todo list
            cid = myPID[todo][ptID]
        else:
            ptID = 0  # takes the first point of the todo list
            cid = myPID[mask][ptID]
        
        if cluster[cid] > 0:
            ccid = cluster[cid]
        else:
            ccid = np.amax(cluster)+1
        
        xref = x[cid]
        yref = y[cid]
        zref = z[cid]
        
        cluster[cid] = ccid
        todo[cid]    = False
        
        dist = np.sqrt((xref-x[todo])**2+(yref-y[todo])**2+(zref-z[todo])**2)
        
        m = dist <= threshold
        
        if np.count_nonzero(m) != 0:
            cluster[myPID[todo][m]] = ccid

    uc = np.unique(cluster)
    temp = np.arange(len(x))
    output = []
    for i in range(len(uc)):
        output += [list(temp[cluster == uc[i]])]
    
    return cluster, output


def distribution(data,binning='log',nbins=10,step=5,small='auto',plot=False,verbose=False):
    """
    Function computing and returning the cumulative, inverse
    cumulative and PDF representing the distribution of an
    input dataset.
    The distributions are computing according to three choices
    of binning: linear, log space and step.

    Args:
        data (numpy.ndarray, shape (N,)): input dataset
        binning (str, optional): binning method:
                    "lin", linear binning with 'nbins' bins
                    "log", bin in log space with 'nbins' bins
                    "step", bin every 'step' samples
                    (Default, binning = "log").
        nbins (int, optional): number of bins if binning in
                    ['lin','log']. (Default, nbins=10)
        step (int, optional): binning every 'step' samples when
                    binning == 'step'. (Default, step=5)
        small (float, optional): define a 'small' (not significant)
                    value for the dataset (negligible quantity).
                    Added to the sample with the maximum value.
                    If small is "auto" then, define "small" as being
                    1/1000 of the smallest value in the input dataset.
                    (Default, small = "auto")
        plot (bool, optional): If True then, produce a figure
                    displaying the resulting PDF.
                    (Default, plot = False)
        verbose (bool, optional): Verbose output option.
                    (Default, verbose = False)
    
    Returns:
        bins (numpy.ndarray): used binning
        cumsumS (numpy.ndarray): resulting inverse cumulative
        cumul (numpy.ndarray): resulting cumulative (derived from cumsumS)
        pdf (numpy.ndarray): resulting PDF (derived from cumsumS)
    """
    # sort the data
    sorted_data = np.sort(data)

    # small
    if small == "auto":
        small = sorted_data[0]/1000
    elif np.isscalar(small):
        pass
    else:
        raise ValueError('Unknown value for "small". "small" must be either an integer, a float number of "auto".')

    # def the binning that will be used
    if isinstance(binning,str):
        if binning == 'lin':
            datamin = np.amin(data)
            datamax = np.amax(data)
            minbin  = datamin
            maxbin  = datamax+(datamax-datamin)*small # make sure to keep the last one
            bins  = np.linspace(minbin,maxbin,nbins)
        elif binning == 'log':
            datamin = np.amin(data)
            datamax = np.amax(data)
            minbin  = np.log10(datamin)
            maxbin  = np.log10(datamax+(datamax-datamin)*small)# make sure to keep the last one
            bins  = np.logspace(minbin,maxbin,nbins)
        elif binning == 'step':
            bins = list(sorted_data[::step])
            j = range(0,len(data),step)
            if j[-1] != len(data)-1:
                bins.append(sorted_data[-1])
            bins = np.array(bins)
            nbins = len(bins)
        else:
            print('ERROR: Unrecognized binning method')
            raise ValueError()
    elif isinstance(binning, np.ndarray):
        bins  = binning
        nbins = binning.shape[0]
    else:
        print('ERROR: Unrecognized binning method')
        raise ValueError()

    # compute the inverse cumulative
    cumsumS = np.zeros(nbins, dtype=np.float64)
    for i in range(nbins):
        cumsumS[i] = len(np.where(data >= bins[i])[0])

    # compute the pdf by derivation of the cumulative
    cumul = np.amax(cumsumS) - cumsumS	# represents the true cumulative
    pdf   = np.zeros(nbins-1,dtype=np.float64)
    norm  = 0  # norm of the pdf
    for i in range(nbins-1):
        dN = abs(cumul[i+1] - cumul[i])
        dS = abs(bins[i+1] - bins[i])
        pdf[i] = dN / dS
        norm += pdf[i] * dS
    pdf = pdf/norm  # normalize the pdf to have: \int_-inf^+inf pdf(s) ds = 1

    # Plot the PDF
    if plot:

        # For plotting, we use the midpoints between successive sorted data values
        midpoints = (bins[:-1] + bins[1:]) / 2

        # Figure
        fig = plt.figure(figsize=(8, 6))
        ax  = fig.add_subplot(111)
        ax.plot(midpoints, pdf, marker='o', linestyle='-', color='r')
        ax.set_title('Probability Density Function')
        ax.set_xlabel('Data Value')
        ax.set_ylabel('Probability Density')
        ax.grid(True)
        if binning == 'log':
            ax.set_xscale('log')
            ax.set_yscale('log')
        plt.show()

    # verify the intergal of the pdf
    integral_pdf = 0
    for i in range(len(pdf)):
        integral_pdf += pdf[i] * abs(bins[i+1] - bins[i])

    if verbose:
        print('Check: intergral PDF = ',integral_pdf)
    
    # return bins, inverse cumulative, cumulative, PDF
    return bins, cumsumS, cumul, pdf