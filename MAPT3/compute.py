# -*- coding: utf-8 -*-
"""
@author: Alexandre JANIN
@aim:    Tools for Plate Processing and Analysis

"""

# External dependencies:
import numpy as np

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

