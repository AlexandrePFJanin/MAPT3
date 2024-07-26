# -*- coding: utf-8 -*-
"""
@author: Alexandre JANIN
@aim:    Generic errors
"""

# External dependencies:
import numpy as np


# ----------------- FUNCTIONS -----------------


def im(textMessage,pName,verbose,error=False):
    """Print verbose internal message. This function depends on the
    argument verbose. If verbose, then the message will be displayed
    in the terminal.
    
    Args:
        textMessage = str, message to display
        pName = str, name of the subprogram
        verbose = bool, condition for the verbose output
    """
    if verbose and not error:
        print('>> '+pName+'| '+textMessage)
    if error:
        #print error message
        print('>> '+pName+'| --- ----- ---')
        print('>> '+pName+'| --- ERROR ---')
        print('>> '+pName+'| --- ----- ---')
        print('>> '+pName+'| '+textMessage)
        raise AssertionError()


def line_count(filename):
    """
    Returns the number of lines in a file.
    """
    import subprocess
    return int(subprocess.check_output(['wc', '-l', filename]).split()[0])


gaussian3D = lambda x,y,z,x0,y0,z0,sigma: np.exp(-((x-x0)**2/(2*sigma**2)+(y-y0)**2/(2*sigma**2)+(z-z0)**2/(2*sigma**2)))


def intstringer(iint,strl):
    """
    Transforms an input integer 'iint' into a string format according
    to a string length 'strl' (condition: strl >= len(str(iint))  ).
    e.g. >> intstringer(4000,5)
         << '04000'
    Args:
        iint = int, input integer you want to transform into a string
        strl = int, length of the output string
    Returns:
        ostr = str, output string
    """
    class BaseError(Exception):
        """Base class for exceptions raised"""
        pass
    class intstringError(BaseError):
        def __init__(self):
            super().__init__('The length of the intput integer is higher '+\
                             'than the length of the string requested length')
    ostr = str(iint)
    if len(ostr) <= strl:
        ostr = '0'*(strl-len(ostr))+ostr
        return ostr
    else:
        raise intstringError()

