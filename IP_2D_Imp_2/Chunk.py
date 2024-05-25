from __future__ import annotations

from RawScanFrame import RawScanFrame
from scipy.spatial import KDTree 
from ProbabilityGrid import ProbabilityGrid
from CommonLib import gaussianKernel, fancyPlot, solidCircle, generate1DGuassianDerivative, rotationMatrix

from scipy.signal import convolve2d
from scipy.ndimage import laplace, maximum_filter, minimum_filter, zoom
from scipy.optimize import minimize
from scipy.optimize import differential_evolution
from typing import Union
import numpy as np
import matplotlib.pyplot as plt

from skimage.transform import EuclideanTransform
from skimage.measure import ransac

from IPConfig import IPConfig

from GraphLib import GraphSLAM2D
from CartesianPose import CartesianPose

import cv2

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
    graphVertexID: int = None
    #offsetFromParent: np.ndarray = None
    parent: Chunk = None
    isCentre: bool = False
    """ This is given in the parent's coordinate frame! With the parent existing at 0,0 with the x axis aligned to its rotation! """
    
    """ MODIFICATION INFO - information relating to properties of this class and if they've been edited (hence requiring recalculation elsewhere) """
    
    """ CACHED MAPPED DATA """
    cachedProbabilityGrid: ProbabilityGrid = None 
    
    graphSLAM:GraphSLAM2D = None

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
        
        this.graphSLAM = GraphSLAM2D()

        return this 
    
    def exportAsRaws( this ) -> list[RawScanFrame]:
        """ This exports the current chunk stack as raws using the new position data """
        
        asRaws = []
        
        rawSubChunks, rawPositions = this.extractAllChildRaws()
        
        for rawSubChunk, rawPosition in zip(rawSubChunks, rawPositions):
            
            centreRaw = rawSubChunk.rawScans[rawSubChunk.centreScanIndex]
            
            newRaw = centreRaw.copy()
            newRaw.pose = CartesianPose( rawPosition[0], rawPosition[1], 0, 0, 0, rawPosition[2] )
            
            asRaws.append( newRaw )
        
        return asRaws
    
    def exportAsChunks( this, chunkSize:int, config:IPConfig  ):
        return Chunk.initFromProcessedScanStack( this.exportAsRaws(), chunkSize, config )
    
    @staticmethod 
    def initFromProcessedScanStack(  inpRaws:list[RawScanFrame], chunkSize, config:IPConfig  ): 
        rawStack = []
        
        asChunks = []
        
        for i in range(0, len(inpRaws)):
            rawStack.append( Chunk.initFromRawScans( [inpRaws[i]], config ) )
            
            if ( len(rawStack) > chunkSize or i==len(inpRaws)-1 ):
                nChunk = Chunk.initEmpty( config )
                nChunk.addChunks( rawStack )
                
                asChunks.append( nChunk )
                
                rawStack = []
    
        return asChunks
    
    """ SECTION - mergers? """ 
    def addChunks( this, subChunks:list[Chunk] ):
        """ Adds chunks, establishing relations between them """
        if ( len(this.subChunks) != 0 ):
            raise RuntimeError("Not implemented to add to initialized chunk groups")

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
            
            centreChunk = this.subChunks[ this.centreChunkIndex ]
            
            centreChunk.isCentre = True
            centreChunk.graphVertexID = this.graphSLAM.add_fixed_pose()
            
            prevChunkIs = [ i for i in range( this.centreChunkIndex, len(this.subChunks)-1 ) ]
            prevChunkIs.extend( [ this.centreChunkIndex-i for i in range( 0, this.centreChunkIndex ) ] )
            
            nextChunkIs = [ i+1 for i in range( this.centreChunkIndex, len(this.subChunks)-1 ) ]
            nextChunkIs.extend( [ this.centreChunkIndex-(i+1) for i in range( 0, this.centreChunkIndex ) ] )
             
            for i in range(0, len(this.subChunks)-1):
                prevChunk = this.subChunks[prevChunkIs[i]]
                nextChunk = this.subChunks[nextChunkIs[i]]
                 
                X, Y, yaw = prevChunk.determineInitialOffset( nextChunk )
                
                #prevChunk.offsetFromParent = np.array(( X, Y, yaw )) 
                nextChunk.graphVertexID = this.graphSLAM.add_pose( prevChunk.graphVertexID, np.array((X, Y, yaw)), 0.1 ) 

            this.graphSLAM.optimize() 
            
            ""
            
    """ SECTION - data integrity managers """
    def clearCache( this ):
        """ Run when there are changes to this parents children which mean cached values are no longer valid
         this is done recursively to target all grand parents """
        this.cachedProbabilityGrid = None

        if ( this.parent != None ):
            this.parent.clearCache()

    def deleteSubChunks( this, indecies ): 
        indecies = sorted(indecies, reverse=True)

        for i in indecies:
            targetChunk:Chunk = this.subChunks[i]

            if ( targetChunk.isCentre ):
                print("warning, attempt to delete centre chunk!")
            else:
                this.subChunks.pop( i )
                targetChunk.parent = None
                targetChunk.clearCache()

        for i in range(0,len(this.subChunks)):
            targetChunk:Chunk = this.subChunks[i]

            if ( targetChunk.isCentre ):
                this.centreChunkIndex = i
        
        this.clearCache()
        
    """ SECTION - graphslam integration """
    
    def updateOffset( this, targetChunk:Chunk, newOffset:np.ndarray ):
        """ updates this chunks offset from some refrance, adding it as a new constraint in the graph """

        if ( this.isCentre ):
            this.parent.graphSLAM.relate_pose(  targetChunk.graphVertexID, this.graphVertexID, -newOffset, 1 )
        else:
            this.parent.graphSLAM.relate_pose(  this.graphVertexID, targetChunk.graphVertexID, newOffset, 1 )
        
        this.parent.graphSLAM.optimize()
        this.parent.clearCache()
        
    """ SECTION - chunk navigators, for getting objects such as children or parents related to this chunk """

    def getCommonParent( this, otherChunk:Chunk ):
        """ returns the most appropriate parent for 2 chunks being compared, if they share a parent that parent is returned
         if one of the two chunks is the parent of the other, then it is returned """
        
        if ( this.parent == otherChunk.parent ):
            return this.parent
        
        if ( this.subChunks.count( otherChunk ) == 1 ):
            return this
        
        if ( otherChunk.subChunks.count( this ) == 1 ):
            return otherChunk
        
        raise RuntimeError("No close relationship found")

    def getMapPose( this ):
        """ This is the chunks currently believed absolute position, atleast relative to the highest parent """
        # TODO make this return the estimated pose AFTER finding offsets of this node from whatever is defined as the origin
        
        raise RuntimeError("Not implemented")
     

    def getNormalOffsetFromLocal( this, localOffset:np.ndarray ):
        """ returns the offset in the parents refrance frame, after being provided a vector in this chunks refrence frame """
        
        thisPosition = this.getOffset()
        
        sepLength = np.sqrt( localOffset[0]**2 + localOffset[1]**2 )
        newBeta = np.arctan2( localOffset[1], localOffset[0] )
        beta = newBeta + thisPosition[2]
        
        normalOffset = np.array(( sepLength*np.cos( beta ), sepLength*np.sin( beta ), localOffset[2] ))
        
        return normalOffset
        
    def getLocalOffsetFromNormal( this, offset:np.ndarray ):
        """ returns the offset in this chunks refrence frame, after being provided a vector in the parents refrence frame """ 

        thisPosition = this.getOffset()
        
        sepLength = np.sqrt( offset[0]**2 + offset[1]**2 )
        beta = np.arctan2( offset[1], offset[0] )
        
        newBeta = beta - thisPosition[2]
        
        localOffset = np.array((sepLength*np.cos( newBeta ), sepLength*np.sin( newBeta ), offset[2] ))
        
        return localOffset
    
    def getLocalOffsetFromTarget( this, otherChunk:Chunk ):
        """ returns the offset from this chunk to the other, within this chunks local coordinate frame """
         
        if ( this.parent != otherChunk.parent ):
            raise RuntimeError("Chunks don't share parent")
        
        thisPosition  = this.getOffset()
        otherPosition = otherChunk.getOffset()
        
        # This offsets in parents refrance frame
        offsetVector = otherPosition - thisPosition
        
        # Here it's converted into this chunks local refrance frame
        return this.getLocalOffsetFromNormal( offsetVector )
    
    def getTRUEOffsetLocal( this, otherChunk:Chunk ):
        if ( this.parent != otherChunk.parent ):
            raise RuntimeError("Chunks don't share parent")
        
        thisPosition  = this.getRawCentre().truePose 
        thisPosition = np.array( (thisPosition.x, thisPosition.y, thisPosition.yaw) )
        otherPosition = otherChunk.getRawCentre().truePose 
        otherPosition = np.array( (otherPosition.x, otherPosition.y, otherPosition.yaw) )
        
        # This offsets in parents refrance frame
        offsetVector = otherPosition - thisPosition 
        
        sepLength = np.sqrt( offsetVector[0]**2 + offsetVector[1]**2 )
        beta = np.arctan2( offsetVector[1], offsetVector[0] )
        
        newBeta = beta - thisPosition[2]
        
        localOffset = np.array((sepLength*np.cos( newBeta ), sepLength*np.sin( newBeta ), offsetVector[2] ))
        
        # Here it's converted into this chunks local refrance frame
        return localOffset
        
    
    def getOffset( this  ):
        """ returns this chunks offset from it's parent, if chunk has no parent it's simply zero """
        
        if ( this.parent == None ):
            return np.zeros(3)
        
        return this.parent.graphSLAM.vertex_pose( this.graphVertexID )
        return this.offsetFromParent.copy()

    def extractAllChildRaws( this ):
        """ returns all raw scan chunks, and their position in this chunks refrance frame. Done recursively """
        
        allRaws    = []
        allOffsets = []
        
        if ( this.isScanWrapper ):
            return [this], [this.getOffset()]
        
        for subChunk in this.subChunks: 
            rawChunks, offsets = subChunk.extractAllChildRaws()
            
            for offset in offsets:
                allOffsets.append( this.getNormalOffsetFromLocal( offset ) )
            allRaws.extend( rawChunks )
                
        return allRaws, allOffsets
    
    def findResonableRelation( this, otherRoot:Chunk ):
        """ Attempts to find over """
        
        thisRaws = this.extractAllChildRaws()
        otherRaws = otherRoot.extractAllChildRaws()
        
        minSep = 99999999999999999
        indx1 = -1
        indx2 = -1
        
        for thisRaw in thisRaws:
            for otherRaw in otherRaws:
                nIndx1 = thisRaw.getRawCentre().index
                nIndx2 = otherRaw.getRawCentre().index
                sep = abs(nIndx1 - nIndx2)
                if ( sep < minSep ):
                    minSep = sep
                    indx1 = nIndx1
                    indx2 = nIndx2
        
        if ( minSep != 1 ):
            raise RuntimeError("Hey we didn't account for this")
        
        fromCommonToThis = thisRaws[indx1].getLocalOffsetFromNormal( this )
        fromCommonToOther = otherRaws[indx2].getLocalOffsetFromNormal( otherRoot )
        
        # NOT ACCOUNTING FOR THE CHANGE BETWEEN "COMMON" FRAMES!
        
        # In this local refrance frame
        vectorToOther = fromCommonToOther - fromCommonToThis
        
        return vectorToOther
        
    def getGrandParentVector( this, grandParent:Chunk ):
        """ Returns the vector (in this chunks refrance frame) from this chunk to the set grandparent """
        if ( grandParent == this ):
            return np.zeros(3)
        if ( this.parent == None ):
            raise RuntimeError("what")
        
        return this.getLocalOffsetFromNormal( this.parent.getGrandParentVector( grandParent ) + this.getOffset() )
    
    def determineInitialOffset( this, comparisonTarget:Chunk ):
        """ This makes the initial estimation of this chunks offset from a target, returning a pose representing the offset
          the returned pose is interms of this nodes orientation
              """
         
        thisCentre = this.getRawCentre().pose
        targetCentre = comparisonTarget.getRawCentre().pose
         
        
        X = thisCentre.x - targetCentre.x
        Y = thisCentre.y - targetCentre.y
        alpha = thisCentre.yaw - targetCentre.yaw

        seperation = np.sqrt( X**2 + Y**2 )
        vecAngle   = np.arctan2( Y, X )- targetCentre.yaw 

        return -np.array((seperation*np.cos( vecAngle ), seperation*np.sin( vecAngle ), alpha))

    def getRawCentre( this ):
        """ This recursively follows the centres of the chunk until the lowest raw middle scan is found """
        if ( this.isScanWrapper ):
            return this.rawScans[ this.centreScanIndex ]
        else:
            #raw0to1 = this.subChunks[ this.centreChunkIndex ].getRawCentre().pose.asNumpy() - this.subChunks[ 0 ].getRawCentre().pose.asNumpy()
            #raw1to2 = this.subChunks[ -1 ].getRawCentre().pose.asNumpy() - this.subChunks[ this.centreChunkIndex ].getRawCentre().pose.asNumpy()
            
            
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
    def constructProbabilityGrid(this, offset:np.ndarray=None ):
        """ used by scan wrapper chunks to return a valid probabilty grid representing the chunk """
          
        if ( this.cachedProbabilityGrid == None ): 
            if ( this.isScanWrapper ):
                # Scan wrapper chunk logic 
                    probGrid = ProbabilityGrid.initFromScanFramesPoly( 
                        this.config.GRID_RESOLUTION,
                        this.rawScans,
                        this.centreScanIndex,
                        this.config.MAX_LIDAR_LENGTH 
                    )
                    probGrid.estimateFeatures( this.config.IE_OBJECT_DIAM, this.config.IE_SHARPNESS ) 
                    
                    this.cachedProbabilityGrid = probGrid 

            else:
                # Composite chunk logic
                transGrids = []
                for subChunk in this.subChunks:
                    transGrids.append( subChunk.constructProbabilityGrid(
                        subChunk.getOffset()
                    ) )

                this.cachedProbabilityGrid = ProbabilityGrid.initFromGridStack( transGrids, False )
        

        if ( offset is None ):
            return this.cachedProbabilityGrid 
        return this.cachedProbabilityGrid.copyTranslated( offset[2], (offset[0], offset[1]), False )

    
    """ SECTION - chunk comparison """

    def determineErrorFeatureless3( this, otherChunk:Chunk, forcedOffset:np.ndarray, showPlot=False ):
        """ estimates the poitional error between two images without using features, the error is given in this chunks local refrance frame """

        localToTargetVector = this.getLocalOffsetFromTarget( otherChunk ) + forcedOffset

        # Overlap is extracted
        thisWindow, transWindow = this.copyOverlaps( otherChunk, localToTargetVector[2], (localToTargetVector[0],localToTargetVector[1]) ) 
        thisWindow = thisWindow**3
        transWindow = transWindow**3
        # existMask = np.abs(thisWindow*transWindow)
        # thisWindow *= existMask
        # transWindow *= existMask
        
        # First the search region is defined  
        intrestMask = np.minimum((thisWindow>0.01)+(transWindow>0.01), 1)
        
        x1DGuas, y1DGuas = generate1DGuassianDerivative(2)
         
        errorWindow = thisWindow-transWindow 
        
        lengthScale = min( np.sum(np.maximum(0, thisWindow)), np.sum(np.maximum(0, transWindow)) ) 
        if ( lengthScale==0 ): return 0,0,0  

        errorWindow = (thisWindow-transWindow)*np.abs(thisWindow*transWindow)
        
        erDx = convolve2d( errorWindow, x1DGuas, mode="same" )*intrestMask
        erDy = convolve2d( errorWindow, y1DGuas, mode="same" )*intrestMask
         
        erDx = erDx*this.config.FEATURELESS_X_ERROR_SCALE/lengthScale 
        erDy = erDy*this.config.FEATURELESS_Y_ERROR_SCALE/lengthScale 
          
        erDyMask = (np.abs(erDy)>np.max(np.abs(erDy))*0.05)
        erDxMask = (np.abs(erDx)>np.max(np.abs(erDx))*0.05)
        
        xError = np.sum(erDx)
        yError = np.sum(erDy)
        
        if ( showPlot ):
            ""

        rows, cols = errorWindow.shape  
        x_coords, y_coords = np.meshgrid(np.arange(cols), np.arange(rows)) 
        origin = (0*localToTargetVector[0]*this.cachedProbabilityGrid.cellRes-this.cachedProbabilityGrid.xAMin, 0*localToTargetVector[1]*this.cachedProbabilityGrid.cellRes-this.cachedProbabilityGrid.yAMin)
        x_offset = (x_coords - origin[0])
        y_offset = (y_coords - origin[1]) 

        # tangential and normal vectors are scaled inversely proportional to origin seperation (to account for increased offsets assosiated with larger seperation)
        sepLen = np.maximum( ( np.square(x_offset) + np.square(y_offset) )/(this.config.GRID_RESOLUTION**2), 0.1 )
        x_tangential_vector = y_offset/sepLen
        y_tangential_vector = -x_offset/sepLen 

        erDa = -(x_tangential_vector*( erDxMask*(xError/np.sum(erDxMask)) ) + y_tangential_vector*(erDyMask*(yError/np.sum(erDyMask)) ))
        erDa = -(x_tangential_vector*erDx + y_tangential_vector*erDy) - erDa 

        angleError = np.sum(erDa) * this.config.FEATURELESS_A_ERROR_SCALE
        if ( np.isnan(angleError) ):
            angleError = 0 # TODO fix it
 

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
 

    def determineErrorFeaturelessMinimum( this, otherChunk:Chunk, iterCap, forcedOffset:np.ndarray,  scoreRequired=99999, updateOffset=False ):
        """ This finds the relative offset between two chunks without using features, using scipy.optimize.minimum """
        transOffset = otherChunk.getOffset()
        myOffset    = this.getOffset() 
        
        #print( "\n\ninitOffset:", this.getOffset()*40 )

        #toTransVector = this.getLocalOffsetFromTarget( otherChunk )
        
        
        initErrorScore, overlapArea = this.determineDirectDifference( otherChunk,  forcedOffset ) 
        
       # this.awoijd = 0
        def interpFunc( offsets ): 
            #this.awoijd+=1
            error, area = this.determineDirectDifference( otherChunk, offsets + forcedOffset )
            #print(error, offsets )

            if (np.isnan(error)):
                return 9999999999
            
            return error

        #initChange = np.array(( 5/this.config.GRID_RESOLUTION, 5/this.config.GRID_RESOLUTION, np.deg2rad(5) ))
        bounds     = ( (-0.3,0.3), (-0.3,0.3), (-np.pi/5, np.pi/5) ) 
        #nm = differential_evolution( interpFunc, bounds, x0=toTransVector, maxiter=5, strategy="currenttobest1exp")
        #nm = minimize( interpFunc, np.zeros(3),  method="COBYLA", options={ 'maxfun': iterCap, 'rhobeg': [0.05,0.05,0.05] } )
        nm = minimize( interpFunc, np.zeros(3),  method="Powell", bounds=bounds, options={ 'maxfev': iterCap, 'maxiter': iterCap, "ftol":1 } )
    
        """
        COBYLA -> 97
        Powell -> 323
        COBYLA -> 323
        """
    
        trueOffset = np.array(( nm.x[0], nm.x[1], nm.x[2] )) 
        errorScore, overlapArea = this.determineDirectDifference( otherChunk, trueOffset + forcedOffset )
        
        foundDirectionErrors =  ( trueOffset)

        if (updateOffset):
             this.updateOffset(
                 otherChunk.getIntermsOfParent( this, trueOffset )
             )
             
        #print( "change:", nm.x*40 )
        #print( "newOffset:", this.getOffset()*40 ) 
        if ( initErrorScore < errorScore or errorScore > scoreRequired ):
            print("rejected", errorScore)
            return np.nan, np.nan
        
        print( "did it:",errorScore )
        return foundDirectionErrors, errorScore 

    def determineErrorFeaturelessDirect( this, otherChunk:Chunk, iterations:int, forcedOffset:np.ndarray, updateOffset=False, scoreRequired=99999999, maxImpScore=0 ):
        """ This finds the relative offset between two chunks without using features, using the custom method """
        forcedOffset = forcedOffset.copy()
        
        errorScores = []
        offsetValues = []
        
        offsetValues.append( forcedOffset.copy() )
        errorScores.append( this.determineDirectDifference( otherChunk, forcedOffset )[0] )
        
        if ( errorScores[0] < maxImpScore ):
            print("godd enough: ", errorScores[0])
            return np.nan, np.nan
         
        for i in range(0, iterations):  
            if ( errorScores[-1] < this.config.ITERATIVE_REDUCTION_PERMITTED_ERROR or ( i>3 and errorScores[-1]>errorScores[-3]*1.4 ) ):
                break # breaks early if the error is below some permitted threshold
            
            predictedErrors = np.array(this.determineErrorFeatureless3( otherChunk, forcedOffset, False ))
            
            forcedOffset -= predictedErrors*this.config.ITERATIVE_REDUCTION_MULTIPLIER
            
            offsetValues.append( forcedOffset.copy() )
            errorScores.append( this.determineDirectDifference( otherChunk, forcedOffset )[0] )
            
        lowestErrorIndex = np.argmin( np.array( errorScores ) )
        
        offsetAdjustment = offsetValues[ lowestErrorIndex ]
        newErrorScore    = errorScores[ lowestErrorIndex ]
        
        if ( newErrorScore > scoreRequired ):
            print("rejected", newErrorScore)
            return np.nan, np.nan
        
        if ( updateOffset ): 
            toTargetVector = ( this.getLocalOffsetFromTarget( otherChunk ) + offsetAdjustment )
            
            #this.plotDifference( otherChunk, offsetAdjustment )
            
            #targetNewPosition = toTargetVector + this.getOffset()
            #this.plotDifference( otherChunk, this.getLocalOffsetFromTarget( otherChunk ), True )
            #this.plotDifference( otherChunk, toTargetVector, True )
            
            this.updateOffset( otherChunk, toTargetVector )
        
        print( "did it:",newErrorScore )
        return offsetAdjustment, newErrorScore
    
    def determineHybridErrorReduction( this, otherChunk:Chunk, forcedOffset:np.ndarray, scoreRequired=99999999, maxImpScore=0 ):
        """ sequentially through all children of this chunk, reducing the error using the selected reduction method
        it applies the reduction by comparing all children to the centre
           """  
              
        adjustmentOffset, errorScore = this.determineOffsetKeypoints( otherChunk, forcedOffset, False, returnOnPoorScore=True )
        
        if ( np.isnan(errorScore) ):
            #rootChunk.determineErrorFeaturelessDirect( targetChunk, 6, np.zeros(3), True, scoreRequired=140 )
            return adjustmentOffset, errorScore
        else:
            return this.determineErrorFeaturelessDirect( otherChunk, 9, adjustmentOffset, False, scoreRequired=scoreRequired, maxImpScore=maxImpScore )
    
    def determineOffsetKeypoints( this, otherChunk:Chunk, forcedOffset:np.ndarray, updateOffset=False, scoreRequired=99999999, returnOnPoorScore=False, maxImpScore=0 ):
        """ This finds the relative offset between two chunks using features  """
          
        initErrorScore = ( this.determineDirectDifference( otherChunk, forcedOffset )[0] )
        
        if ( initErrorScore < maxImpScore ):
            print("godd enough: ", initErrorScore)
            return np.nan, np.nan
        
        newOffset, wasSuccess = this.determineErrorKeypoints( otherChunk, forcedOffset )
        if ( not wasSuccess or ( np.sum( np.isnan( newOffset ) ) ) ):
            return np.nan, np.nan
        
        newErrorScore, overlapRegion  = this.determineDirectDifference( otherChunk, newOffset, True )
        adjustmentOffset = newOffset - this.getLocalOffsetFromTarget( otherChunk )
        
        if ( newErrorScore > initErrorScore or overlapRegion<10 or newErrorScore > scoreRequired ):
            if ( returnOnPoorScore ): 
                return adjustmentOffset, newErrorScore
            return np.nan, np.nan
         
        
        if ( updateOffset ):
            # this is found in parent refrance frame 
            #targetNewPosition = toTargetVector + this.getOffset()
            
            this.updateOffset( otherChunk, newOffset )
        
        return adjustmentOffset, newErrorScore
    
    def determineDirectDifference( this, otherChunk:Chunk, forcedOffset:np.ndarray, completeOffsetOverride=False ):
        """ determines the error between two images without using features """
        
        localToTargetVector = this.getLocalOffsetFromTarget( otherChunk ) + forcedOffset
  
        # Overlap is extracted
        thisWindow, transWindow = this.copyOverlaps( otherChunk, localToTargetVector[2], (localToTargetVector[0],localToTargetVector[1]) )

        thisPositive  = np.maximum( thisWindow,  0 )
        transPositive = np.maximum( transWindow, 0 )
        
        #thisNeg  = -np.minimum( thisWindow,  0 )
        #transNeg = -np.minimum( transWindow, 0 ) 
        

        errorWindow = (thisWindow*transWindow)
        
        mArea = np.sum( (thisPositive+transPositive  )*np.abs(errorWindow) ) 
        #mArea = np.sum( (thisPositive+transPositive + 0.01*(thisNeg+transNeg) )*np.abs(errorWindow) ) 
        
        #if (mArea<10): return np.nan, np.nan 
        misMatchWindow = -np.minimum( errorWindow, 0 ) 
        
        if ( mArea == 0 or mArea<20 ):
            return 1000000000, 0

        errorScore = 1000*np.sum(misMatchWindow)/mArea

        return errorScore, mArea
        
    def plotDifference( this, otherChunk:Chunk, forcedOffset:np.ndarray, offsetTotalOverwrite=False ):
        """ determines the error between two images without using features """

        transOffset = otherChunk.getOffset()
        myOffset    = this.getOffset()

        toTransVector = transOffset - myOffset

        toTransVector += forcedOffset
        if ( offsetTotalOverwrite ):
            toTransVector = forcedOffset

        # Overlap is extracted
        thisWindow, transWindow = this.copyOverlaps( otherChunk, toTransVector[2], (toTransVector[0],toTransVector[1]) )
 

        errorWindow = (thisWindow*transWindow)
        mArea = np.sum(np.abs(errorWindow) ) 
        if (mArea<10): return np.nan, np.nan

        #errorWindow = -np.minimum( errorWindow, 0 ) 

        errorScore = 1000*np.sum(errorWindow)/mArea

        """fancyPlot( thisWindow )
        fancyPlot( transWindow )
        fancyPlot( errorWindow )"""
        fancyPlot( errorWindow ) 

        return errorScore, mArea
 
    def copyOverlaps( this, transChunk:Chunk, rotation:float, translation: Union[float, float], onlyMapEstimate=False ):
        """ 
            returns a copy of the overlapping region between both grids, it takes inputs in their local refrance frames 
        """


        thisProbGrid   = this.constructProbabilityGrid()
        nTransProbGrid = transChunk.constructProbabilityGrid().copyTranslated( rotation, translation, True )
        

        cXMin = max( thisProbGrid.xAMin, nTransProbGrid.xAMin ) + 1
        cXMax = min( thisProbGrid.xAMax, nTransProbGrid.xAMax ) - 1
        cYMin = max( thisProbGrid.yAMin, nTransProbGrid.yAMin ) + 1
        cYMax = min( thisProbGrid.yAMax, nTransProbGrid.yAMax ) - 1
        
        if ( cXMin >= cXMax or cYMin >= cYMax ):
            return np.zeros(4).reshape((2,2)), np.zeros(4).reshape((2,2))

        cWidth  = cXMax-cXMin
        cHeight = cYMax-cYMin

        thisLCXMin, thisLCYMin = cXMin-thisProbGrid.xAMin, cYMin-thisProbGrid.yAMin
        transLCXMin, transLCYMin = cXMin-nTransProbGrid.xAMin, cYMin-nTransProbGrid.yAMin

        thisWindow  = thisProbGrid.mapEstimate[ thisLCYMin:thisLCYMin+cHeight, thisLCXMin:thisLCXMin+cWidth ]
        transWindow = nTransProbGrid.mapEstimate[ transLCYMin:transLCYMin+cHeight, transLCXMin:transLCXMin+cWidth ]

        return thisWindow, transWindow

    def determineErrorKeypoints( this, otherChunk:Chunk, forcedOffset:np.ndarray, showPlot=False ): 
        this.cachedProbabilityGrid.extractDescriptors()
        otherChunk.cachedProbabilityGrid.extractDescriptors()

        image1 = np.uint8(255*(this.cachedProbabilityGrid.mapEstimate+1)/2)
        image2 = np.uint8(255*(otherChunk.cachedProbabilityGrid.mapEstimate+1)/2)

        thisKeypoints, thisDescriptors = this.cachedProbabilityGrid.asKeypoints, this.cachedProbabilityGrid.featureDescriptors
        otherKeypoints, otherDescriptors = otherChunk.cachedProbabilityGrid.asKeypoints, otherChunk.cachedProbabilityGrid.featureDescriptors
        
        transSet = this.getLocalOffsetFromTarget( otherChunk ) + forcedOffset 
        
        transRotation = rotationMatrix( -transSet[2] )
        transVector = np.array((transSet[0], transSet[1]))*this.cachedProbabilityGrid.cellRes
        
        origin1 = np.array(( this.cachedProbabilityGrid.xAMin, this.cachedProbabilityGrid.yAMin ))
        origin2 = np.array(( otherChunk.cachedProbabilityGrid.xAMin, otherChunk.cachedProbabilityGrid.yAMin ))
 
        if ( showPlot ):
            rawKP1 = np.array([ keyPoint.pt for keyPoint in thisKeypoints ]) 
            rawKP2 = np.array([ keyPoint.pt for keyPoint in otherKeypoints ]) 
            
            trnsKP2 = np.dot(rawKP2 + origin2, transRotation) + ( transVector - origin1 )
            
            plt.figure(1934)
            plt.imshow( this.cachedProbabilityGrid.mapEstimate, origin="lower" )
            plt.plot( rawKP1[:,0], rawKP1[:,1], "rx" )
            plt.plot( trnsKP2[:,0], trnsKP2[:,1], "bx" )
            plt.show(block=False)
            ""
        
        if ( showPlot ):
            # Draw keypoints on the image
            image_with_keypoints = cv2.drawKeypoints(image1, thisKeypoints, None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS) 
            # Draw lines representing angles of keypoints
            for kp in thisKeypoints:
                angle = kp.angle
                x, y = kp.pt
                # Calculate endpoint for the line
                endpoint_x = int(x + 10 * np.cos((angle)))
                endpoint_y = int(y + 10 * np.sin((angle)))
                # Draw the line
                cv2.line(image_with_keypoints, (int(x), int(y)), (endpoint_x, endpoint_y), (0, 0, 255), 1)
                # Draw an 'x' at the keypoint
                cv2.drawMarker(image_with_keypoints, (int(x), int(y)), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=5, thickness=1) 
            # Display the image
            cv2.imshow('Image 1 with keypoints and angles', image_with_keypoints)
            
            # Draw keypoints on the image
            image_with_keypoints = cv2.drawKeypoints(image2, otherKeypoints, None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS) 
            # Draw lines representing angles of keypoints
            for kp in otherKeypoints:
                angle = kp.angle
                x, y = kp.pt
                # Calculate endpoint for the line
                endpoint_x = int(x + 10 * np.cos((angle)))
                endpoint_y = int(y + 10 * np.sin((angle)))
                # Draw the line
                cv2.line(image_with_keypoints, (int(x), int(y)), (endpoint_x, endpoint_y), (0, 0, 255), 1)
                # Draw an 'x' at the keypoint
                cv2.drawMarker(image_with_keypoints, (int(x), int(y)), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=5, thickness=1) 
            # Display the image
            cv2.imshow('Image 2 with keypoints and angles', image_with_keypoints)
        
        ""

        # Perform keypoint matching
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=150)  # or pass empty dictionary
         
        kVal = 5 if this.config.FEATURE_COMPARISON_FILTER_DISTANCE else 2
        
        if ( thisDescriptors.size<5 or otherDescriptors.size<5 ): 
            return False, False
        
        bf = cv2.BFMatcher() 
        matches = bf.knnMatch(thisDescriptors, otherDescriptors, k=kVal)  
        
        # Extract keypoints
        src_I   = []
        src_pts = []
        
        dst_pts = [] 
        flatMatches = []
        
        
        offSetScale = 0
        
        for matchSet in matches: 
            if ( this.config.FEATURE_COMPARISON_FILTER_DISTANCE ):
                for match in matchSet:
                    if (  match.distance*0.92 > matchSet[0].distance or match.distance>5 ):#
                        break 
                    
                    srcKeypoint = thisKeypoints[match.queryIdx]
                    dstKeypoint = otherKeypoints[match.trainIdx]
                    
                    src_I.append( match.queryIdx )
                    src_pts.append( srcKeypoint.pt )
                    dst_pts.append( dstKeypoint.pt )
                    flatMatches.append( match )
            else:
                if (  matchSet[1].distance*0.8 > matchSet[0].distance ):#  
                    srcKeypoint = thisKeypoints[matchSet[0].queryIdx]
                    dstKeypoint = otherKeypoints[matchSet[0].trainIdx]
                    
                    src_I.append( matchSet[0].queryIdx )
                    src_pts.append( srcKeypoint.pt )
                    dst_pts.append( dstKeypoint.pt )
                    flatMatches.append( matchSet[0] )
        
            
        
        src_pts = np.float32(src_pts).reshape(-1, 2)
        dst_pts = np.float32(dst_pts).reshape(-1, 2) 
        
        if ( src_pts.size < 3 or dst_pts.size < 3 ):
            return False, False
        
        dst_ptsInFrame = np.dot(dst_pts + origin2, transRotation) + ( transVector - origin1 )
        
        sepsSquared = np.sum( (src_pts - dst_ptsInFrame)**2, axis=1)
        
        src_pts_filt = []
        dst_pts_filt = []
        flat_match_filt = []
        
        # This is a disgusting design, but it works
        listIndx = 0
        while ( listIndx < len(src_I) ):
            targQIdx = src_I[listIndx]
            
            bestSeperation = 99999999999
            bestSepIndex   = -1
            
            while (listIndx < len(src_I) and targQIdx == src_I[listIndx]):
                if ( sepsSquared[listIndx] < bestSeperation ):
                    bestSeperation = sepsSquared[listIndx]
                    bestSepIndex = listIndx
                listIndx += 1
                
            src_pts_filt.append( src_pts[ bestSepIndex ] )
            dst_pts_filt.append( dst_pts[ bestSepIndex ] )
            flat_match_filt.append( flatMatches[ bestSepIndex ] )
                
        src_pts_filt = np.array(src_pts_filt)
        dst_pts_filt = np.array(dst_pts_filt)
        
        if ( src_pts_filt.size < 3 or dst_pts_filt.size < 3 ):
            return False, False
        
        model = EuclideanTransform()
        model.estimate( src_pts_filt, dst_pts_filt )
        
        model_robust, inliers = ransac(
            (src_pts_filt, dst_pts_filt), EuclideanTransform, min_samples=2, residual_threshold=2, max_trials=100
        )
        
        if ( showPlot ):
            # Draw filtered matches
            matched_image = cv2.drawMatches(image1, thisKeypoints, image2, otherKeypoints, flatMatches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS) 
            # Display the matched image
            cv2.imshow('Matches flat', matched_image)
            
            # Draw filtered matches
            matched_image = cv2.drawMatches(image1, thisKeypoints, image2, otherKeypoints, flat_match_filt, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS) 
            # Display the matched image
            cv2.imshow('Matches flat filt1', matched_image)
            
        if ( (inliers is None) or np.sum(inliers) < 3 ):
            return False, False
        
        accRotation = -( model_robust.rotation )
        accTrans = -model_robust.translation
        
        newTransVector = ( origin1+np.dot(accTrans-origin2, rotationMatrix(-accRotation)) )/ this.cachedProbabilityGrid.cellRes
        newTransVector = np.array(( newTransVector[0], newTransVector[1], accRotation ))
        
        if ( showPlot ): 
            rawKP1 = np.array([ keyPoint.pt for keyPoint in thisKeypoints ]) 
            rawKP2 = np.array([ keyPoint.pt for keyPoint in otherKeypoints ]) 
            
            trnsKP2 = np.dot(rawKP2 + origin2, rotationMatrix(-newTransVector[2])) + ( newTransVector[0:2]*this.cachedProbabilityGrid.cellRes - origin1 )
            
            plt.figure(1937)
            plt.imshow( this.cachedProbabilityGrid.mapEstimate, origin="lower" )
            plt.plot( rawKP1[:,0], rawKP1[:,1], "rx" )
            plt.plot( trnsKP2[:,0], trnsKP2[:,1], "bx" )
            plt.show(block=False)
            ""
 
        inMatches = [flat_match_filt[i] for i, inlier in enumerate(inliers) if inlier]
        
        if ( showPlot ):
            this.plotDifference( otherChunk, transSet, True )
            this.plotDifference( otherChunk, newTransVector, True )
            plt.show(block=False)
            
            # Draw filtered matches
            matched_image = cv2.drawMatches(image1, thisKeypoints, image2, otherKeypoints, inMatches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS) 
            # Display the matched image
            cv2.imshow('Matches', matched_image)
 
            
            #cv2.imshow('Matches flat', cv2.drawMatches(image1, thisKeypoints, image2, otherKeypoints, flat_match_filt, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS))
  
        return newTransVector, True




    """ SECTION - full chunk layer manipulation """

    def centredFeaturelessErrorReduction( this, minMethod:bool=False ):
        """ sequentially through all children of this chunk, reducing the error using the selected reduction method
        it applies the reduction by comparing all children to the centre
           """ 
        centreChunk = this.subChunks[this.centreChunkIndex]

        if ( minMethod ):
            for targetChunk in this.subChunks:
                if ( targetChunk != centreChunk ):
                    centreChunk.plotDifference( targetChunk )
                    centreChunk.determineErrorFeaturelessMinimum( targetChunk, np.zeros(3), True )
                    centreChunk.plotDifference( targetChunk )
                    plt.show()
                    ""
        else:
            for targetChunk in this.subChunks:
                if ( targetChunk != centreChunk ):
                    #centreChunk.plotDifference( targetChunk )
                    centreChunk.determineErrorFeaturelessDirect( targetChunk, 8, np.zeros(3), True, scoreRequired=80 )
                    #centreChunk.plotDifference( targetChunk )
                    #plt.show(block=False)
                    ""

    def linearFeaturelessErrorReduction( this, skipSize, scoreRequired=80, maxImpScore=80  ):
        """ sequentially through all children of this chunk, reducing the error using the selected reduction method
        it applies the reduction by comparing all children to the centre
           """  
        for i in range(skipSize, len(this.subChunks)):
            targetChunk = this.subChunks[i]
                
            rootChunk:Chunk = this.subChunks[i - skipSize]
            
            rootChunk.determineErrorFeaturelessDirect( targetChunk, 8, np.zeros(3), True, scoreRequired=scoreRequired, maxImpScore=maxImpScore )
    
    def centredHybridErrorReduction( this ):
        centreChunk = this.subChunks[this.centreChunkIndex]
 
        for targetChunk in this.subChunks:
            if ( targetChunk != centreChunk ):
                #centreChunk.plotDifference( targetChunk )
                errorScore = centreChunk.determineOffsetKeypoints( targetChunk, np.zeros(3), True )[1]
                
                if ( not np.isnan(errorScore) ):
                    centreChunk.determineErrorFeaturelessDirect( targetChunk, 9, np.zeros(3), True, scoreRequired=80 )
                
                #centreChunk.plotDifference( targetChunk )
                #plt.show(block=False)
                "" 
    
    def linearHybridErrorReduction( this, skipSize, scoreRequired=80, maxImpScore=80 ):
        """ sequentially through all children of this chunk, reducing the error using the selected reduction method
        it applies the reduction by comparing all children to the centre
           """  
        for i in range(skipSize, len(this.subChunks)):
            targetChunk = this.subChunks[i]
                
            rootChunk:Chunk = this.subChunks[i-skipSize] #this.subChunks[i + (skipSize if i<this.centreChunkIndex else -skipSize)] 
            
            adjustmentOffset, errorScore = rootChunk.determineOffsetKeypoints( targetChunk, np.zeros(3), False, returnOnPoorScore=True )
            
            if ( np.isnan(errorScore) ):
                #rootChunk.determineErrorFeaturelessDirect( targetChunk, 6, np.zeros(3), True, scoreRequired=140 )
                ""
            else:
                rootChunk.determineErrorFeaturelessDirect( targetChunk, 9, adjustmentOffset, True, scoreRequired=scoreRequired, maxImpScore=maxImpScore )
    
    def randomHybridErrorReduction( this, scoreRequired=80, maxImpScore=80  ):
        """ sequentially through all children of this chunk, reducing the error using the selected reduction method
        it applies the reduction by comparing all children to the centre
           """  
        for i in range(0, len(this.subChunks)):
            targetChunk = this.subChunks[i]
            rootChunk:Chunk = this.subChunks[ int(np.random.random()*len(this.subChunks)) ]
            if ( rootChunk != targetChunk ):
                
                adjustmentOffset, errorScore = rootChunk.determineOffsetKeypoints( targetChunk, np.zeros(3), False, returnOnPoorScore=True )
                
                if ( np.isnan(errorScore) ):
                    #rootChunk.determineErrorFeaturelessDirect( targetChunk, 6, np.zeros(3), True, scoreRequired=140 )
                    ""
                else:
                    rootChunk.determineErrorFeaturelessDirect( targetChunk, 6, adjustmentOffset, True, scoreRequired=scoreRequired, maxImpScore=maxImpScore  )
     
    def linearKeypointErrorReduction( this, skipSize ):
        """ sequentially through all children of this chunk, reducing the error using the selected reduction method
        it applies the reduction by comparing all children to the centre
           """  
        for i in range(0, len(this.subChunks)):
            targetChunk = this.subChunks[i]
            if ( not targetChunk.isCentre ):
                rootChunk:Chunk = this.subChunks[i + (skipSize if i<this.centreChunkIndex else -skipSize)]
                
                rootChunk.determineOffsetKeypoints( targetChunk, np.zeros(3), True, scoreRequired=150 )
    
    def centredPrune( this, pruneMult=1.5, overlapMult=1.5 ):
        centreChunk = this.subChunks[this.centreChunkIndex]
 
        errors = []
        overlaps = []

        for i in range(0, len(this.subChunks)):
            targetChunk = this.subChunks[i]
                
            error, overlap = centreChunk.determineDirectDifference( targetChunk ) 
            errors.append( error )
            overlaps.append( overlap )

        errors = np.array( errors )
        overlaps = np.array( overlaps )
        
        maxError = np.median( errors )*pruneMult
        minOverlap = np.median( overlap )/overlapMult

        pruneTargets = np.where( (errors>maxError) | (overlaps<minOverlap) )[0]

        print("deleting: ",pruneTargets)
        
        this.deleteSubChunks( pruneTargets.tolist() )

    def linearPrune( this, skipSize, pruneMult=1.5 ):
        """ sequentially through all children of this chunk, reducing the error using the selected reduction method
        it applies the reduction by comparing all children to the centre
           """  
        
        errors = []

        for i in range(skipSize, len(this.subChunks)):
            targetChunk = this.subChunks[i]
            if ( not targetChunk.isCentre ):
                rootChunk:Chunk = this.subChunks[i - skipSize]
                
                error, overlap = rootChunk.determineDirectDifference( targetChunk )

                errors.append( error )
            else:
                errors.append(0)

        errors = np.array( errors )
        maxError = np.median( errors )*pruneMult

        pruneTargets = np.where( errors>maxError )[0]

        this.deleteSubChunks( pruneTargets.tolist() )

    def repeatingHybridPrune( this, minFrameError:float, mergeMethod:list[int], maxIterations=20, errorCompSep=3 ):
        """ sequentially through all children of this chunk, reducing the error using the selected reduction method
        it applies the reduction by comparing all children to the centre
           """  
        
        if ( len(this.subChunks) == 1 ):
            return 
        
        errorCompSep = min( len(this.subChunks)-1, errorCompSep )
        
        for I in range(0, maxIterations):
            errors = []
            for i in range(errorCompSep, len(this.subChunks)):
                targetChunk = this.subChunks[i]
                    
                error, overlap = targetChunk.determineDirectDifference( this.subChunks[i-errorCompSep], forcedOffset=np.zeros(3) )
                errors.append( error ) 
            
            maxError = max( errors )
            
            if ( np.mean(np.array(errors)) < minFrameError ):
                print( "final max error ", maxError, "  mean: ", np.mean(np.array(errors)) )
                return
            
            print( "I ", I, ":   ", maxError, "  mean: ", np.mean(np.array(errors)) )
            ""
            for cM in mergeMethod:
                #this.graphSLAM.plot()
                if ( len(this.subChunks) < abs(cM)-1 ):
                    print("would fail to error check, len: ",len(this.subChunks), " to min len ", abs(cM)-1)
                else:
                    if ( cM < 0 ):
                        this.linearFeaturelessErrorReduction( -cM, minFrameError*0.7, minFrameError*0.6 ) 
                    else:
                        this.linearHybridErrorReduction( cM, minFrameError*0.7, minFrameError*0.6 ) 

        print( "ran out of iterations: ", maxError, "  mean: ", np.mean(np.array(errors)) )
        # errors = np.array( errors )
        # maxError = np.median( errors )*pruneMult

        # pruneTargets = np.where( errors>maxError )[0]

        # this.deleteSubChunks( pruneTargets.tolist() )

        
        


