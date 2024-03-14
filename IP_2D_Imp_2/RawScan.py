import numpy as np
from dataclasses import dataclass, field
import CartesianPose
from scipy.signal import convolve2d
import jsonpickle

@dataclass
class ScanFrame:
    """Dataclass that stores a set of points representing the scan, and a raw pose from which the scans thought to be taken""" 
    scanDistances: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    scanAngles: np.ndarray = field(default_factory=lambda: np.array([], dtype=float)) 
    pose: CartesianPose = field(default_factory=lambda: CartesianPose.zero() ) 

