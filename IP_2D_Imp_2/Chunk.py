from __future__ import annotations

from RawScanFrame import RawScanFrame
from scipy.spatial import KDTree
from CartesianPose import CartesianPose
from ProbabilityGrid import ProbabilityGrid
from CommonLib import gaussianKernel, fancyPlot, solidCircle

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

    def determineErrorFeatureless( this, otherChunk:Chunk, forcedOffset:np.ndarray=np.zeros(3), showPlot=False ):
        """ estimates the poitional error between two images without using features """

        transOffset = otherChunk.determineInitialOffset()
        myOffset    = this.determineInitialOffset()

        toTransVector = transOffset - myOffset

        toTransVector += forcedOffset

        # Overlap is extracted
        thisWindow, transWindow = this.copyOverlaps( otherChunk, toTransVector[2], (toTransVector[0],toTransVector[1]) )

        # First the search region is defined
        searchKernal = solidCircle( this.config.FEATURELESS_PIX_SEARCH_DIAM )
        intrestMask = (convolve2d( thisWindow>0, searchKernal, mode="same" ) )>0*((convolve2d( transWindow>0, searchKernal, mode="same" ))>0)
        
        errorWindow = (thisWindow - transWindow) *intrestMask *np.abs(thisWindow * transWindow)

        """#fancyPlot( this.cachedProbabilityGrid.mapEstimate )
        #fancyPlot( nTransProbGrid.mapEstimate )
        
        plt.figure(45)
        plt.imshow( thisProbGrid.mapEstimate, origin="lower" ) 
        plt.plot( [thisLCXMin,thisLCXMin+cWidth], [thisLCYMin,thisLCYMin+cHeight], "bx"  )
        #plt.plot( [-thisProbGrid.xAMin,translation[0]*thisProbGrid.cellRes-thisProbGrid.xAMin], [-thisProbGrid.yAMin,translation[1]*thisProbGrid.cellRes-thisProbGrid.xAMin], "ro"  )
        
        plt.figure(46)
        plt.imshow( nTransProbGrid.mapEstimate, origin="lower" ) 
        plt.plot( [transLCXMin,transLCXMin+cWidth], [transLCYMin,transLCYMin+cHeight], "bx"  )
        #plt.plot( [-nTransProbGrid.xAMin], [-nTransProbGrid.yAMin], "ro"  )"""
        
        kernal = gaussianKernel( 1 )
         
        erDy, erDx = np.gradient( errorWindow )
        #erDy = convolve2d( erDy, kernal, mode="same" ) *(thisWindow<0)
        #erDx = convolve2d( erDx, kernal, mode="same" ) 
        
        # Imperical adjustments made to make final errors more accurate
        erDy = this.config.FEATURELESS_Y_ERROR_SCALE*(thisWindow<0)*erDy#*convolve2d( erDy, kernal, mode="same" ) 
        erDx = this.config.FEATURELESS_X_ERROR_SCALE*(thisWindow<0)*erDx#*convolve2d( erDx, kernal, mode="same" )  

        # Length scale keeps errors consistantly sized
        lengthScale = np.sqrt(np.sum(transWindow*(transWindow>0)))
        yError = np.sum(erDy)/lengthScale
        xError = np.sum(erDx)/lengthScale 

        rows, cols = errorWindow.shape  
        x_coords, y_coords = np.meshgrid(np.arange(cols), np.arange(rows)) 
        origin = (-this.cachedProbabilityGrid.xAMin, -this.cachedProbabilityGrid.yAMin)
        x_offset = (x_coords - origin[0])
        y_offset = (y_coords - origin[1]) 

        # tangential and normal vectors are scaled inversely proportional to origin seperation (to account for increased offsets assosiated with larger seperation)
        sepLen = np.maximum( ( np.square(x_offset) + np.square(y_offset) )/(this.config.GRID_RESOLUTION**2), 0.1 )
        x_tangential_vector = y_offset/sepLen
        y_tangential_vector = -x_offset/sepLen 

        angleError = -np.sum(x_tangential_vector*erDx + y_tangential_vector*erDy)/lengthScale
        angleError *= this.config.FEATURELESS_A_ERROR_SCALE

        # Reduce coupling effect
        xCompensation = (1-np.cos(angleError))
        yCompensation = (np.sin(angleError))
        aCompensation = np.arctan( np.array([yError]),np.array([xError]) )[0]  

        #xError     -= xCompensation*this.config.FEATURELESS_COMP_FACT
        #yError     -= yCompensation*this.config.FEATURELESS_COMP_FACT
        angleError -= aCompensation*this.config.FEATURELESS_COMP_FACT

        errorWindow *= thisWindow<0
        errorWindow = np.where( (errorWindow)==0, np.inf, errorWindow ) 

        if ( showPlot ):
            #fancyPlot( thisWindow )
            #fancyPlot( transWindow )
             
            fancyPlot( sepLen )
            fancyPlot( x_tangential_vector )
            fancyPlot( y_tangential_vector )
            
            fancyPlot( x_tangential_vector*erDx )
            fancyPlot( y_tangential_vector*erDy )
            
            fancyPlot( errorWindow )
            fancyPlot( thisWindow-transWindow )
            fancyPlot( intrestMask )

            plt.show(block=False)
 

        return xError, yError, angleError 

    def determineDirectDifference( this, otherChunk:Chunk, forcedOffset:np.ndarray=np.zeros(3) ):
        """ determines the error between two images without using features """

        transOffset = otherChunk.determineInitialOffset()
        myOffset    = this.determineInitialOffset()

        toTransVector = transOffset - myOffset

        toTransVector += forcedOffset

        # Overlap is extracted
        thisWindow, transWindow = this.copyOverlaps( otherChunk, toTransVector[2], (toTransVector[0],toTransVector[1]) )
  
        errorWindow = (thisWindow*transWindow)
        mArea = np.sum(np.abs(errorWindow) ) 

        errorWindow = -np.minimum( errorWindow, 0 ) 

        errorScore = 1000*np.sum(errorWindow)/mArea
 

        return errorScore, mArea

        

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

        thisWindow  = thisProbGrid.mapEstimate[ thisLCYMin:thisLCYMin+cHeight, thisLCXMin:thisLCXMin+cWidth ]
        transWindow = nTransProbGrid.mapEstimate[ transLCYMin:transLCYMin+cHeight, transLCXMin:transLCXMin+cWidth ]

        return thisWindow, transWindow



