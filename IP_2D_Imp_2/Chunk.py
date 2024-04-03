from __future__ import annotations

from RawScanFrame import RawScanFrame
from scipy.spatial import KDTree
from CartesianPose import CartesianPose
from ProbabilityGrid import ProbabilityGrid
from CommonLib import gaussianKernel, fancyPlot

from scipy.signal import convolve2d
from scipy.ndimage import laplace, maximum_filter, minimum_filter, zoom
from typing import Union
import numpy as np
import matplotlib.pyplot as plt

from IPConfig import IPConfig

class Chunk:
    """ A datastructure which recursively contains data relevant to building a map, with functions for processing that data
     across multiple levels in a way that improves the overall maps accuracy. Following composition over inheritance. """

    """ INPUT PROPERTIES """
    isScanWrapper: bool = None
    unifiedScan: RawScanFrame = None
    rawScans:  list[RawScanFrame] = None
    centreScanIndex: int = None
    subChunks: list[Chunk] = None 
    centreChunkIndex: int = None
    config: IPConfig = IPConfig() # Can be changed

    """ RELATIONAL PROPERTIES - AS A CHILD """
    offsetFromParent: CartesianPose = None
    parent: Chunk = None
    """ This is given in the parent's coordinate frame! With the parent existing at 0,0 with the x axis aligned to its rotation! """
    
    """ MODIFICATION INFO - information relating to properties of this class and if they've been edited (hence requiring recalculation elsewhere) """
    
    """ CACHED MAPPED DATA """
    cachedProbabilityGrid: ProbabilityGrid = None 

    """ SECTION - constructors (static) """
    @staticmethod
    def initFromRawScans( inputScans:list[RawScanFrame], config:IPConfig, centreIndex:int=-1 ) -> Chunk:
        """ The initialisation method to be used when constructing the Chunk out of frames, centreIndex is -1 to auto select a resonable one """
        this = Chunk()
        this.config = config

        if ( centreIndex == -1 ):
            this.centreScanIndex = int(len(inputScans)/2)
        else:
            this.centreScanIndex = centreIndex

        this.isScanWrapper = True
        this.rawScans = inputScans 

        return this

    @staticmethod 
    def initEmpty( config:IPConfig ) -> Chunk:
        """ The initialisation method to be used when constructing the Chunk, this is for constructing chunks to be made of sub-chunks """
        this = Chunk()
        this.config = config

        this.isScanWrapper = False 
        this.subChunks = []

        return this 
    
    """ SECTION - mergers? """ 
    def addChunks( this, subChunks:list[Chunk] ):
        """ Adds chunks, establishing relations between them """
        this.subChunks.extend( subChunks )
        for nSubChunk in subChunks:
            nSubChunk.parent = this

        this.determineCentreChunk()

    def determineCentreChunk( this, forcedValue=-1 ):
        """ Sets the centre of this chunk to be the midpoint of the list, it then finds the subchunks relative offsets
            currently cannot change chunk centre after initilization!
           """
        if ( this.centreChunkIndex != None ):
            raise RuntimeError("Not yet implemented")

        else:
            if ( forcedValue == -1 ):
                this.centreChunkIndex = int(len(this.subChunks)/2)
            else:
                this.centreChunkIndex = forcedValue 
            
            for otherChunk in this.subChunks:
                X, Y, yaw = otherChunk.determineInitialOffset()
                 
                otherChunk.offsetFromParent = CartesianPose( X, Y, 0, 0, 0, yaw )

    """ SECTION - chunk navigators, for getting objects such as children or parents related to this chunk """

    def getMapPose( this ):
        """ This is the chunks currently believed absolute position, atleast relative to the highest parent """
        # TODO make this return the estimated pose AFTER finding offsets of this node from whatever is defined as the origin
        NotImplemented
        return this.subChunkOffsets
    
    def determineInitialOffset( this ):
        """ This makes the initial estimation of this chunks offset from it's parent, returning a pose representing the offset
          the returned pose is interms of the parent's orientation (forward, upward, alignedYaw)
              """
        
        # TODO current implementation only makes use of first raw position data
        parentCentre = this.parent.getRawCentre().pose
        thisCentre = this.getRawCentre().pose
        
        X = thisCentre.x - parentCentre.x
        Y = thisCentre.y - parentCentre.y
        alpha = thisCentre.yaw - parentCentre.yaw

        seperation = np.sqrt( X**2 + Y**2 )
        vecAngle   = np.arctan2( Y, X )- parentCentre.yaw 

        return np.array((seperation*np.cos( vecAngle ), seperation*np.sin( vecAngle ), alpha))

    def getRawCentre( this ):
        """ This recursively follows the centres of the chunk until the lowest raw middle scan is found """
        if ( this.isScanWrapper ):
            return this.rawScans[ this.centreScanIndex ]
        else:
            return this.subChunks[ this.centreChunkIndex ].getRawCentre()

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
        
        """noOffset = offset==-1
        if ( noOffset ):
            if ( this.cachedProbabilityGrid != None ):
                return this.cachedProbabilityGrid 
            offset = CartesianPose.zero() """
        
        if ( this.isScanWrapper ):
            probGrid = ProbabilityGrid.initFromScanFramesPoly( 
                this.config.GRID_RESOLUTION,
                this.rawScans,
                this.centreScanIndex,
                this.config.MAX_LIDAR_LENGTH#,
                #offset
            )
            probGrid.estimateFeatures( this.config.IE_OBJECT_DIAM, this.config.IE_SHARPNESS ) 
            
            #if ( noOffset ):
            this.cachedProbabilityGrid = probGrid 
 
            return probGrid
             
        else:
            ""
    
    """ SECTION - chunk comparison """

    def determineErrorFeatureless( this, otherChunk:Chunk ):
        """ determines the error between two images without using features """

        transOffset = otherChunk.determineInitialOffset()
        myOffset    = this.determineInitialOffset()

        toTransVector = transOffset - myOffset
        rotation = toTransVector[2]

        test = this.copyOverlaps( otherChunk, rotation, (toTransVector[0],toTransVector[1]) )


    def determineOffsetFeatureless( this, otherChunk:Chunk ):
        """ determines the offset using image comparison methods instead of feature matching """
    
    def copyOverlaps( this, transChunk:Chunk, rotation:float, translation: Union[float, float], onlyMapEstimate=False ):
        """ 
            returns a copy of the overlapping region between both grids, it takes inputs in their refrance frames 
        """


        thisProbGrid   = this.cachedProbabilityGrid
        nTransProbGrid = transChunk.cachedProbabilityGrid.copyTranslated( rotation, translation, True )
        

        cXMin = max( thisProbGrid.xAMin, nTransProbGrid.xAMin ) + 1
        cXMax = min( thisProbGrid.xAMax, nTransProbGrid.xAMax ) - 1
        cYMin = max( thisProbGrid.yAMin, nTransProbGrid.yAMin ) + 1
        cYMax = min( thisProbGrid.yAMax, nTransProbGrid.yAMax ) - 1

        cWidth  = cXMax-cXMin
        cHeight = cYMax-cYMin

        thisLCXMin, thisLCYMin = cXMin-thisProbGrid.xAMin, cYMin-thisProbGrid.yAMin
        transLCXMin, transLCYMin = cXMin-nTransProbGrid.xAMin, cYMin-nTransProbGrid.yAMin

        thisWindow  = thisProbGrid.mapEstimate[ thisLCXMin:thisLCXMin+cWidth, thisLCYMin:thisLCYMin+cHeight ]
        transWindow = nTransProbGrid.mapEstimate[ transLCXMin:transLCXMin+cWidth, transLCYMin:transLCYMin+cHeight ]

        #fancyPlot( this.cachedProbabilityGrid.mapEstimate )
        #fancyPlot( nTransProbGrid.mapEstimate )
        
        plt.figure(45)
        plt.imshow( thisProbGrid.mapEstimate, origin="lower" ) 
        plt.plot( [thisLCXMin,thisLCXMin+cWidth], [thisLCYMin,thisLCYMin+cHeight], "bx"  )
        #plt.plot( [-thisProbGrid.xAMin,translation[0]*thisProbGrid.cellRes-thisProbGrid.xAMin], [-thisProbGrid.yAMin,translation[1]*thisProbGrid.cellRes-thisProbGrid.xAMin], "ro"  )
        
        plt.figure(46)
        plt.imshow( nTransProbGrid.mapEstimate, origin="lower" ) 
        plt.plot( [transLCXMin,transLCXMin+cWidth], [transLCYMin,transLCYMin+cHeight], "bx"  )
        #plt.plot( [-nTransProbGrid.xAMin], [-nTransProbGrid.yAMin], "ro"  )
        
        fancyPlot( thisWindow )
        fancyPlot( transWindow )
        
        
        
        #fancyPlot( thisWindow-transWindow )

        plt.show(block=False)

        ""



