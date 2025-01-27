# -*- coding: utf-8 -*-
"""
@author: Alexandre JANIN
@aim:    Planetary models data
"""


# ----------------- CLASS -----------------


class Planet():
    """
    Main class for planetary models
    """
    def __init__(self, radius=0):
        self.radius = radius # mean radius in [km]

class EarthModel(Planet):
    def __init__(self):
        super().__init__(radius=6371.0088)

class VenusModel(Planet):
    def __init__(self):
        super().__init__(radius=6051.8)

class MarsModel(Planet):
    def __init__(self):
        super().__init__(radius=3389.5)


# ----------------- INSTANCES -----------------


Earth = EarthModel()
Venus = VenusModel()
Mars  = MarsModel()
