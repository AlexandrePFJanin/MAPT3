# -*- coding: utf-8 -*-
"""
@author: Alexandre JANIN
@aim:    Geometric transformations
"""


# ----------------- CLASS -----------------


class Planet():
    """
    Main class for planetary models
    """
    pass

class EarthModel(Planet):
    def __init__(self):
        super().__init__()
        self.radius = 6371.0088 # mean radius in [km]

class VenusModel(Planet):
    def __init__(self):
        super().__init__()
        self.radius = 6051.8    # mean radius in [km]

class MarsModel(Planet):
    def __init__(self):
        super().__init__()
        self.radius = 3389.5    # mean radius in [km]


# ----------------- INSTANCES -----------------


Earth = EarthModel()
Venus = VenusModel()
Mars  = VenusModel()