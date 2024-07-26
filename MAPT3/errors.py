# -*- coding: utf-8 -*-
"""
@author: Alexandre JANIN
@aim:    Generic errors
"""


class TessellationOptimizationError(Exception):
    """ Main class for optimization errors"""
    pass

class OptimizationError(TessellationOptimizationError):
    """Raised when error on the plate tessellation optimization"""
    def __init__(self,msg):
        super().__init__('Error during the optimization of the plate tessellation!\n'+\
            '>> '+msg)

class RigidityError(TessellationOptimizationError):
    """Raised when option not yet implemented"""
    def __init__(self):
        super().__init__('Error during the optimization: Parameters for the rigidity not yet implemented.')

class OptimizationSettingsError(TessellationOptimizationError):
    """Raised when bad arguments are passed to an optimization function"""
    def __init__(self,msg=''):
        super().__init__('Error: Bad argument passed to the function\n'+msg)


class ProjectError(Exception):
    """ Main class for MAPT3 project errors"""
    pass

class ParFileExtError(ProjectError):
    """Raised when bad extension of the par file"""
    def __init__(self):
        super().__init__('Error: the input par file should have the extension .py')

class ParFileNameError(ProjectError):
    """Raised when bad par file name"""
    def __init__(self):
        super().__init__("Error: the input par file should not have another '.' except for the extension")

class ProjectCheckError(ProjectError):
    """Raised when error during the checking control of the project"""
    def __init__(self,msg=''):
        super().__init__("Error: Bad project configuration.\n"+msg)


class TimeTrackingError(Exception):
    """ Main class for MAPT3 time tracking errors"""
    pass

class LoadingFormatError(TimeTrackingError):
    """Raised when bad format passed during the loading of the tracking file"""
    def __init__(self,msg):
        super().__init__('Error: Unkown input format. The data format have to be in '+msg)

