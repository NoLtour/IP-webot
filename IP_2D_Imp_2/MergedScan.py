
from Chunk import Chunk
import numpy as np

class MergedScan:
    """ This is a datastructure constructed inside chunks, it is basically just the points present in rawscan(s) merged together
     into a format that's suitable for fast probability grid construction. 

     It only needs to be constructed if feature extraction or image processing needs to be done on that layer
    """

    scanPoints:np.ndarray = None
    """ Scanpoints are given relative to this scans "centre" """
    
    @staticmethod
    def constructFr(parentChunk:Chunk ):
        this = MergedScan()

        parentChunk.rawScans


