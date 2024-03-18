from __future__ import annotations

from RawScanFrame import RawScanFrame
from scipy.spatial import KDTree
from CartesianPose import CartesianPose
from ProbabilityGrid import ProbabilityGrid

from IPConfig import IPConfig

class Chunk:
    """ A datastructure which recursively contains data relevant to building a map, with functions for processing that data
     across multiple levels in a way that improves the overall maps accuracy. Following composition over inheritance. """

    """ INPUT PROPERTIES """
    isScanWrapper: bool = None
    unifiedScan: RawScanFrame = None
    rawScans:  list[RawScanFrame] = None
    centreScanIndex: int = None
    subChunks: Chunk = None
    config: IPConfig = IPConfig() # Can be changed
    
    """ MODIFICATION INFO - information relating to properties of this class and if they've been edited (hence requiring recalculation elsewhere) """
    
    """ CACHED MAPPED DATA """
    cachedProbabilityGrid: ProbabilityGrid = None
    

    """ SECTION - constructors (static) """
    @staticmethod
    def initFromRawScans( inputScans:list[RawScanFrame], centreIndex:int=-1 ) -> Chunk:
        """ The initialisation method to be used when constructing the Chunk out of frames, centreIndex is -1 to auto select a resonable one """
        this = Chunk()

        if ( centreIndex == -1 ):
            this.centreScanIndex = int(len(inputScans)/2)
        else:
            this.centreScanIndex = centreIndex

        this.isScanWrapper = True
        this.rawScans = inputScans 

        return this

    @staticmethod 
    def initEmpty() -> Chunk:
        """ The initialisation method to be used when constructing the Chunk, this is for constructing chunks to be made of sub-chunks """
        this = Chunk()

        this.isScanWrapper = False

        return this 
    

    """ SECTION - chunk navigators, for getting objects such as children or parents related to this chunk """

    def getRawFrameData( this ):
        """ returns raw frames found within this chunk structure, as well as the offsets to be applied to get them all into this chunks
         coordinate frame """
        
        if ( this.isScanWrapper ):
            ""
            
        else:
            NOTIMPLEMENTED

        return frameList, offsetList

    """ SECTION - image construction """
    def constructProbabilityGrid(this, offset:CartesianPose=-1 ):
        """ used by scan wrapper chunks to return a valid probabilty grid representing the chunk """
        
        noOffset = offset==-1
        if ( noOffset ):
            if ( this.cachedProbabilityGrid != None ):
                return this.c
            offset = CartesianPose.zero() 
        
        if ( this.isScanWrapper ):
            probGrid = ProbabilityGrid.initFromScanFramesPoly( 
                this.config.GRID_RESOLUTION,
                this.rawScans,
                this.centreScanIndex,
                this.config.MAX_LIDAR_LENGTH,
                offset
            )
            
            if ( noOffset ):
                this.cachedProbabilityGrid = probGrid
                
            return probGrid
            
            
        else:
            NOTIMPLEMENTED
        
        
        


    




