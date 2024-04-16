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
    offsetFromParent: np.ndarray = None
    parent: Chunk = None
    isCentre: bool = False
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
                
            this.subChunks[ this.centreChunkIndex ].isCentre = True
            
            for otherChunk in this.subChunks:
                X, Y, yaw = otherChunk.determineInitialOffset()
                 
                otherChunk.offsetFromParent = np.array(( X, Y, yaw ))
    
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
    
    def updateOffset( this, newOffset:np.ndarray ):
        """ updates this chunks offset from parent, setting it to the new value """

        if ( this.isCentre ):
            raise RuntimeError("Cannot update the offset of the centre chunk!")
        
        this.offsetFromParent = newOffset
        this.parent.clearCache()

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
    
    def getOffset( this  ):
        """ returns this chunks offset from it's parent, if chunk has no parent it's simply zero """
        
        if ( this.parent == None ):
            return np.zeros(3)
        
        return this.offsetFromParent.copy()

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

    def determineErrorFeatureless3( this, otherChunk:Chunk, forcedOffset:np.ndarray=np.zeros(3), showPlot=False ):
        """ estimates the poitional error between two images without using features, the error is given in this chunks local refrance frame """

        localToTargetVector = this.getLocalOffsetFromTarget( otherChunk ) + forcedOffset

        # Overlap is extracted
        thisWindow, transWindow = this.copyOverlaps( otherChunk, localToTargetVector[2], (localToTargetVector[0],localToTargetVector[1]) ) 
        
        # First the search region is defined  
        intrestMask = np.minimum((thisWindow>0.01)+(transWindow>0.01), 1)
        
        x1DGuas, y1DGuas = generate1DGuassianDerivative(this.config.FEATURELESS_PIX_SEARCH_DIAM/2)
        
        errorWindow = -np.minimum(thisWindow*transWindow, 0)
        errorWindow = -(thisWindow*transWindow)  

        conflictWindow = (thisWindow*transWindow) 
        conflictWindow = -np.minimum( conflictWindow, 0 )  
 
        erDx = convolve2d( errorWindow, x1DGuas, mode="same" )
        erDy = convolve2d( errorWindow, y1DGuas, mode="same" ) 
        
        lengthScale = np.sum(thisWindow*intrestMask*(thisWindow>0))/this.config.OBJECT_PIX_DIAM
        if ( lengthScale==0 ): return 0,0,0 
        
        erDx = conflictWindow*np.where(thisWindow<0,erDx,-erDx)/lengthScale 
        erDy = conflictWindow*np.where(thisWindow<0,erDy,-erDy)/lengthScale  
         
        erDx = erDx*this.config.FEATURELESS_X_ERROR_SCALE
        erDy = erDy*this.config.FEATURELESS_Y_ERROR_SCALE
          
        erDyMask = (np.abs(erDy)>0)
        erDxMask = (np.abs(erDx)>0)
        
        xError = np.sum(erDx)
        yError = np.sum(erDy)

        rows, cols = errorWindow.shape  
        x_coords, y_coords = np.meshgrid(np.arange(cols), np.arange(rows)) 
        origin = (localToTargetVector[0]*this.cachedProbabilityGrid.cellRes-this.cachedProbabilityGrid.xAMin, localToTargetVector[1]*this.cachedProbabilityGrid.cellRes-this.cachedProbabilityGrid.yAMin)
        x_offset = (x_coords - origin[0])
        y_offset = (y_coords - origin[1]) 

        # tangential and normal vectors are scaled inversely proportional to origin seperation (to account for increased offsets assosiated with larger seperation)
        sepLen = np.maximum( ( np.square(x_offset) + np.square(y_offset) )/(this.config.GRID_RESOLUTION**2), 0.1 )
        x_tangential_vector = y_offset/sepLen
        y_tangential_vector = -x_offset/sepLen 

        erDa = -(x_tangential_vector*( erDxMask*(np.sum(erDx)/np.sum(erDxMask)) ) + y_tangential_vector*(erDyMask*(np.sum(erDy)/np.sum(erDyMask)) ))
        erDa = -(x_tangential_vector*erDx + y_tangential_vector*erDy) - erDa 

        angleError = np.sum(erDa) * this.config.FEATURELESS_A_ERROR_SCALE
        if ( np.isnan(angleError) ):
            angleError = 0 # TODO fix it


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

    # def determineErrorFeatureless2( this, otherChunk:Chunk, forcedOffset:np.ndarray=np.zeros(3), showPlot=False ):
    #     """ estimates the poitional error between two images without using features """

    #     transOffset = otherChunk.getOffset()
    #     myOffset    = this.getOffset()

    #     toTransVector = transOffset - myOffset

    #     toTransVector += forcedOffset

    #     # Overlap is extracted
    #     thisWindow, transWindow = this.copyOverlaps( otherChunk, toTransVector[2], (toTransVector[0],toTransVector[1]) )

    #     #thisWindow = convolve2d( thisWindow, gaussianKernel(this.config.FEATURELESS_PIX_SEARCH_DIAM/4, 0.05), mode="same" )
    #     #transWindow = convolve2d( transWindow, gaussianKernel(this.config.FEATURELESS_PIX_SEARCH_DIAM/4, 0.05), mode="same" )
        
    #     # First the search region is defined
    #     searchKernal = solidCircle( this.config.FEATURELESS_PIX_SEARCH_DIAM )
    #     #convTrans = convolve2d( transWindow, gaussianKernel(this.config.FEATURELESS_PIX_SEARCH_DIAM/4), mode="same" )
    #     intrestMask = ((convolve2d( thisWindow>0, searchKernal, mode="same" ) )>0)*((convolve2d( transWindow>0, searchKernal, mode="same" ))>0)
        
    #     errorWindow = (thisWindow - transWindow) * intrestMask*np.abs(thisWindow * transWindow)
    #     errorWindow = convolve2d( errorWindow, gaussianKernel(this.config.FEATURELESS_PIX_SEARCH_DIAM/4, 0.05), mode="same" )

    #     conflictWindow = (thisWindow*transWindow*intrestMask)
        
    #     conflictWindow = -np.minimum( conflictWindow, 0 )  


    #     """#fancyPlot( this.cachedProbabilityGrid.mapEstimate )
    #     #fancyPlot( nTransProbGrid.mapEstimate )
        
    #     plt.figure(45)
    #     plt.imshow( thisProbGrid.mapEstimate, origin="lower" ) 
    #     plt.plot( [thisLCXMin,thisLCXMin+cWidth], [thisLCYMin,thisLCYMin+cHeight], "bx"  )
    #     #plt.plot( [-thisProbGrid.xAMin,translation[0]*thisProbGrid.cellRes-thisProbGrid.xAMin], [-thisProbGrid.yAMin,translation[1]*thisProbGrid.cellRes-thisProbGrid.xAMin], "ro"  )
        
    #     plt.figure(46)
    #     plt.imshow( nTransProbGrid.mapEstimate, origin="lower" ) 
    #     plt.plot( [transLCXMin,transLCXMin+cWidth], [transLCYMin,transLCYMin+cHeight], "bx"  )
    #     #plt.plot( [-nTransProbGrid.xAMin], [-nTransProbGrid.yAMin], "ro"  )"""
         
    #     erDy, erDx = np.gradient( errorWindow ) 
    #     #erDy = convolve2d( erDy, kernal, mode="same" ) *(thisWindow<0)
    #     #erDx = convolve2d( erDx, kernal, mode="same" ) 

    #     # Length scale keeps errors consistantly sized, here the y and x error are non-dimensionalised
    #     #lengthScale = np.sqrt(np.sum(thisWindow*(thisWindow>0)))
    #     #lengthScale = np.sum(np.abs(erDy))+np.sum(np.abs(erDx))
    #     lengthScale = np.sum(thisWindow*intrestMask*(thisWindow>0))/this.config.OBJECT_PIX_DIAM
    #     if ( lengthScale==0 ): return 0,0,0
    #     conflictMultiplier = 1 + this.config.CONFLICT_MULT_GAIN*np.sum(conflictWindow)/lengthScale
        
    #     #conflictMultiplier = 1
        
    #     """fancyPlot( errorWindow )
    #     fancyPlot( (thisWindow<0)*erDy/lengthScale ) 
    #     fancyPlot( -((thisWindow>0)*erDy/lengthScale) )
    #     fancyPlot( np.where(thisWindow<0,erDy,-erDy)/lengthScale ) 
    #     plt.show()"""

    #     # Imperical adjustments made to make final errors more accurate
    #     #erDx = (thisWindow<0)*erDx/lengthScale 
    #     #erDy = (thisWindow<0)*erDy/lengthScale 
        
        
    #     erDx = np.where(thisWindow<0,erDx,-erDx)/lengthScale 
    #     erDy = np.where(thisWindow<0,erDy,-erDy)/lengthScale  
        
    #     #erDx = convolve2d( erDx, gaussianKernel(this.config.FEATURELESS_PIX_SEARCH_DIAM/4, 0.05), mode="same" )*this.config.FEATURELESS_X_ERROR_SCALE*conflictMultiplier
    #     #erDy = convolve2d( erDy, gaussianKernel(this.config.FEATURELESS_PIX_SEARCH_DIAM/4, 0.05), mode="same" )*this.config.FEATURELESS_Y_ERROR_SCALE*conflictMultiplier
    #     erDx = erDx*this.config.FEATURELESS_X_ERROR_SCALE*conflictMultiplier
    #     erDy = erDy*this.config.FEATURELESS_Y_ERROR_SCALE*conflictMultiplier
         
    #     maxErr = max(np.max( erDx ),np.max( erDy ))
    #     erDyMask = (np.abs(erDy)>maxErr*0.025)
    #     erDxMask = (np.abs(erDx)>maxErr*0.025)
    #     """erDx *= erDxMask
    #     erDy *= erDyMask"""
        
    #     xError = np.sum(erDx)
    #     yError = np.sum(erDy)

    #     rows, cols = errorWindow.shape  
    #     x_coords, y_coords = np.meshgrid(np.arange(cols), np.arange(rows)) 
    #     origin = (-this.cachedProbabilityGrid.xAMin, -this.cachedProbabilityGrid.yAMin)
    #     x_offset = (x_coords - origin[0])
    #     y_offset = (y_coords - origin[1]) 

    #     # tangential and normal vectors are scaled inversely proportional to origin seperation (to account for increased offsets assosiated with larger seperation)
    #     sepLen = np.maximum( ( np.square(x_offset) + np.square(y_offset) )/(this.config.GRID_RESOLUTION**2), 0.1 )
    #     x_tangential_vector = y_offset/sepLen
    #     y_tangential_vector = -x_offset/sepLen 

    #     """fancyPlot( errorWindow )
    #     fancyPlot( erDx )
    #     fancyPlot( erDy )
    #     fancyPlot( erDx - erDxMask*(np.sum(erDx)/np.sum(erDxMask)) )
    #     fancyPlot( erDy - erDyMask*(np.sum(erDy)/np.sum(erDyMask)) )
    #     plt.show(block=False)"""
        

    #     #angleError = -np.sum(x_tangential_vector*( erDx - erDxMask*(np.sum(erDx)/np.sum(erDxMask)) ) + y_tangential_vector*( erDy - erDyMask*(np.sum(erDy)/np.sum(erDyMask)) ))
    #     #angleError = -np.sum(x_tangential_vector*erDx + y_tangential_vector*erDy)

    #     erDa = -(x_tangential_vector*( erDxMask*(np.sum(erDx)/np.sum(erDxMask)) ) + y_tangential_vector*(erDyMask*(np.sum(erDy)/np.sum(erDyMask)) ))
    #     erDa = -(x_tangential_vector*erDx + y_tangential_vector*erDy) - erDa

    #     angleError = np.sum(erDa) * this.config.FEATURELESS_A_ERROR_SCALE

    #     """ratio = erDx/erDy
    #     aErDy = erDa/( x_tangential_vector*ratio + y_tangential_vector )
    #     aErDx = aErDy*ratio
        
    #     xError -= np.sum( aErDx )
    #     yError -= np.sum( aErDy )"""
        
    #     """angleError *= conflictMultiplier
    #     xError *= conflictMultiplier
    #     yError *= conflictMultiplier"""

    #     """if ( abs(angleError) > this.config.ANGLE_OVERWIRTE_THRESHOLD ):
    #         xError *= this.config.ANGLE_OVERWIRTE_THRESHOLD/angleError
    #         yError *= this.config.ANGLE_OVERWIRTE_THRESHOLD/angleError"""

    #     #aCompensation = np.arctan( np.array([yError]),np.array([xError]) )[0]  
    #     #angleError -= aCompensation*this.config.FEATURELESS_COMP_FACT

    #     # Reduce coupling effect
    #     xCompensation = (1-np.cos(angleError))
    #     yCompensation = (np.sin(angleError))

    #     #xError     -= xCompensation*this.config.FEATURELESS_COMP_FACT
    #     #yError     -= yCompensation*this.config.FEATURELESS_COMP_FACT

    #     errorWindow *= thisWindow<0
    #     errorWindow = np.where( (errorWindow)==0, np.inf, errorWindow ) 

    #     if ( showPlot ):
    #         #fancyPlot( thisWindow )
    #         #fancyPlot( transWindow )
             
    #         fancyPlot( sepLen )
    #         fancyPlot( x_tangential_vector )
    #         fancyPlot( y_tangential_vector )
            
    #         fancyPlot( x_tangential_vector*erDx )
    #         fancyPlot( y_tangential_vector*erDy )
            
    #         fancyPlot( errorWindow )
    #         fancyPlot( thisWindow-transWindow )
    #         fancyPlot( intrestMask )

    #         plt.show(block=False)
 

    #     return xError, yError, angleError
    
    # def determineErrorFeatureless( this, otherChunk:Chunk, forcedOffset:np.ndarray=np.zeros(3), showPlot=False ):
    #     """ estimates the poitional error between two images without using features """

    #     transOffset = otherChunk.getOffset()
    #     myOffset    = this.getOffset()

    #     toTransVector = transOffset - myOffset

    #     toTransVector += forcedOffset

    #     # Overlap is extracted
    #     thisWindow, transWindow = this.copyOverlaps( otherChunk, toTransVector[2], (toTransVector[0],toTransVector[1]) )

    #     # First the search region is defined
    #     searchKernal = solidCircle( this.config.FEATURELESS_PIX_SEARCH_DIAM )
    #     intrestMask = (convolve2d( thisWindow>0, searchKernal, mode="same" ) )>0*((convolve2d( transWindow>0, searchKernal, mode="same" ))>0)
        
    #     errorWindow = (thisWindow - transWindow) *intrestMask *np.abs(thisWindow * transWindow)

         

    #     """#fancyPlot( this.cachedProbabilityGrid.mapEstimate )
    #     #fancyPlot( nTransProbGrid.mapEstimate )
        
    #     plt.figure(45)
    #     plt.imshow( thisProbGrid.mapEstimate, origin="lower" ) 
    #     plt.plot( [thisLCXMin,thisLCXMin+cWidth], [thisLCYMin,thisLCYMin+cHeight], "bx"  )
    #     #plt.plot( [-thisProbGrid.xAMin,translation[0]*thisProbGrid.cellRes-thisProbGrid.xAMin], [-thisProbGrid.yAMin,translation[1]*thisProbGrid.cellRes-thisProbGrid.xAMin], "ro"  )
        
    #     plt.figure(46)
    #     plt.imshow( nTransProbGrid.mapEstimate, origin="lower" ) 
    #     plt.plot( [transLCXMin,transLCXMin+cWidth], [transLCYMin,transLCYMin+cHeight], "bx"  )
    #     #plt.plot( [-nTransProbGrid.xAMin], [-nTransProbGrid.yAMin], "ro"  )"""
        
    #     kernal = gaussianKernel( 1 )
         
    #     erDy, erDx = np.gradient( errorWindow )
    #     #erDy = convolve2d( erDy, kernal, mode="same" ) *(thisWindow<0)
    #     #erDx = convolve2d( erDx, kernal, mode="same" ) 
        
    #     # Imperical adjustments made to make final errors more accurate
    #     erDy = this.config.FEATURELESS_Y_ERROR_SCALE*(thisWindow<0)*erDy#*convolve2d( erDy, kernal, mode="same" ) 
    #     erDx = this.config.FEATURELESS_X_ERROR_SCALE*(thisWindow<0)*erDx#*convolve2d( erDx, kernal, mode="same" )  

    #     # Length scale keeps errors consistantly sized
    #     lengthScale = np.sqrt(np.sum(thisWindow*(thisWindow>0)))
    #     yError = np.sum(erDy)/lengthScale
    #     xError = np.sum(erDx)/lengthScale 

    #     rows, cols = errorWindow.shape  
    #     x_coords, y_coords = np.meshgrid(np.arange(cols), np.arange(rows)) 
    #     origin = (-this.cachedProbabilityGrid.xAMin, -this.cachedProbabilityGrid.yAMin)
    #     x_offset = (x_coords - origin[0])
    #     y_offset = (y_coords - origin[1]) 

    #     # tangential and normal vectors are scaled inversely proportional to origin seperation (to account for increased offsets assosiated with larger seperation)
    #     sepLen = np.maximum( ( np.square(x_offset) + np.square(y_offset) )/(this.config.GRID_RESOLUTION**2), 0.1 )
    #     x_tangential_vector = y_offset/sepLen
    #     y_tangential_vector = -x_offset/sepLen 

    #     angleError = -np.sum(x_tangential_vector*erDx + y_tangential_vector*erDy)/lengthScale
    #     angleError *= this.config.FEATURELESS_A_ERROR_SCALE

    #     # Reduce coupling effect
    #     xCompensation = (1-np.cos(angleError))
    #     yCompensation = (np.sin(angleError))
    #     aCompensation = np.arctan( np.array([yError]),np.array([xError]) )[0]  

    #     #xError     -= xCompensation*this.config.FEATURELESS_COMP_FACT
    #     #yError     -= yCompensation*this.config.FEATURELESS_COMP_FACT
    #     angleError -= aCompensation*this.config.FEATURELESS_COMP_FACT

    #     errorWindow *= thisWindow<0
    #     errorWindow = np.where( (errorWindow)==0, np.inf, errorWindow ) 

    #     if ( True or showPlot ):
    #         #fancyPlot( thisWindow )
    #         #fancyPlot( transWindow )
             
    #         fancyPlot( sepLen )
    #         fancyPlot( x_tangential_vector )
    #         fancyPlot( y_tangential_vector )
            
    #         fancyPlot( x_tangential_vector*erDx )
    #         fancyPlot( y_tangential_vector*erDy )
            
    #         fancyPlot( errorWindow )
    #         fancyPlot( thisWindow-transWindow )
    #         fancyPlot( intrestMask )

    #         plt.show(block=False)
 

    #     return xError, yError, angleError

    # def determineErrorFeaturelessMinimumOLD( this, otherChunk:Chunk, forcedOffset:np.ndarray=np.zeros(3), updateOffset=False ):
    #     """ This finds the relative offset between two chunks without using features, using scipy.optimize.minimum """
    #     transOffset = otherChunk.getOffset()
    #     myOffset    = this.getOffset()
        
    #     #print( "\n\ninitOffset:", this.getOffset()*40 )

    #     toTransVector = transOffset - myOffset + forcedOffset
        
    #     initErrorScore, overlapArea = this.determineDirectDifference( otherChunk, toTransVector, True ) 
 
    #     def interpFunc( offsets ):
    #         error, area = this.determineDirectDifference( otherChunk, offsets, True )

    #         if (np.isnan(error)):
    #             return 9999999999
            
    #         return error

    #     initChange = np.array(( 5/this.config.GRID_RESOLUTION, 5/this.config.GRID_RESOLUTION, np.deg2rad(5) ))
    #     nm = minimize( interpFunc, toTransVector,  method="COBYLA", options={ 'maxiter': this.config.MINIMISER_MAX_LOOP } )
    
    #     trueOffset = ( nm.x[0], nm.x[1], nm.x[2] ) 
    #     errorScore, overlapArea = this.determineDirectDifference( otherChunk, trueOffset, True )
        
    #     foundDirectionErrors =  toTransVector - trueOffset

    #     if (updateOffset):
    #          this.updateOffset(
    #              this.getIntermsOfParent( otherChunk, nm.x )
    #          )
             
    #     #print( "change:", nm.x*40 )
    #     #print( "newOffset:", this.getOffset()*40 ) 
    #     if ( initErrorScore < errorScore ):
    #         return np.zeros(3), initErrorScore, 1
        
    #     return foundDirectionErrors, errorScore, 1

    # def determineErrorFeaturelessMinimum( this, otherChunk:Chunk, forcedOffset:np.ndarray=np.zeros(3), updateOffset=False ):
        """ This finds the relative offset between two chunks without using features, using scipy.optimize.minimum """
        transOffset = otherChunk.getOffset()
        myOffset    = this.getOffset()
        
        raise RuntimeError("Not implemented properly don't use")
        
        #print( "\n\ninitOffset:", this.getOffset()*40 )

        toTransVector = transOffset - myOffset 
        
        initErrorScore, overlapArea = this.determineDirectDifference( otherChunk, toTransVector + forcedOffset, True ) 
        
       # this.awoijd = 0
        def interpFunc( offsets ): 
            #this.awoijd+=1
            error, area = this.determineDirectDifference( otherChunk, offsets + forcedOffset, True )
            #print(this.awoijd, error, offsets )

            if (np.isnan(error)):
                return 9999999999
            
            return error

        #initChange = np.array(( 5/this.config.GRID_RESOLUTION, 5/this.config.GRID_RESOLUTION, np.deg2rad(5) ))
        bounds     = ( (-0.5,0.5), (-0.5,0.5), (-np.pi/2, np.pi/2) ) 
        #nm = differential_evolution( interpFunc, bounds, x0=toTransVector, maxiter=5, strategy="currenttobest1exp")
        #nm = minimize( interpFunc, toTransVector,  method="COBYLA", options={ 'maxiter': this.config.MINIMISER_MAX_LOOP } )
        nm = minimize( interpFunc, toTransVector,  method="Powell", bounds=bounds, options={ 'maxfev': this.config.MINIMISER_MAX_LOOP, 'maxiter': this.config.MINIMISER_MAX_LOOP, "ftol":1 } )
    
        """
        COBYLA -> 97
        Powell -> 323
        COBYLA -> 323
        """
    
        trueOffset = np.array(( nm.x[0], nm.x[1], nm.x[2] )) 
        errorScore, overlapArea = this.determineDirectDifference( otherChunk, trueOffset + forcedOffset, True )
        
        foundDirectionErrors =  toTransVector - trueOffset

        if (updateOffset):
             this.updateOffset(
                 otherChunk.getIntermsOfParent( this, trueOffset )
             )
             
        #print( "change:", nm.x*40 )
        #print( "newOffset:", this.getOffset()*40 ) 
        if ( initErrorScore < errorScore ):
            return np.zeros(3), initErrorScore, 1
        
        return foundDirectionErrors, errorScore, 1

    def determineErrorFeaturelessDirect( this, otherChunk:Chunk, iterations:int, forcedOffset:np.ndarray=np.zeros(3), updateOffset=False ):
        """ This finds the relative offset between two chunks without using features, using the custom method """
        
        errorScores = []
        offsetValues = []
        
        offsetValues.append( forcedOffset.copy() )
        errorScores.append( this.determineDirectDifference( otherChunk, forcedOffset )[0] )
        
        for i in range(0, iterations):  
            if ( errorScores[-1] < this.config.ITERATIVE_REDUCTION_PERMITTED_ERROR or ( i>3 and errorScores[-1]>errorScores[-4] ) ):
                break # breaks early if the error is below some permitted threshold
            
            predictedErrors = np.array(this.determineErrorFeatureless3( otherChunk, forcedOffset ))
            
            forcedOffset -= predictedErrors*this.config.ITERATIVE_REDUCTION_MULTIPLIER
            
            offsetValues.append( forcedOffset.copy() )
            errorScores.append( this.determineDirectDifference( otherChunk, forcedOffset )[0] )
            
        lowestErrorIndex = np.argmin( np.array( errorScores ) )
        
        offsetAdjustment = offsetValues[ lowestErrorIndex ]
        newErrorScore    = errorScores[ lowestErrorIndex ]
        
        if ( updateOffset ):
            # this is found in parent refrance frame
            toTargetVector = this.getNormalOffsetFromLocal( this.getLocalOffsetFromTarget( otherChunk ) + offsetAdjustment )
            
            targetNewPosition = toTargetVector + this.getOffset()
            
            otherChunk.updateOffset( targetNewPosition )
            
        return offsetAdjustment, newErrorScore
    
    def determineDirectDifference( this, otherChunk:Chunk, forcedOffset:np.ndarray=np.zeros(3), completeOffsetOverride=False ):
        """ determines the error between two images without using features """
        
        localToTargetVector = this.getLocalOffsetFromTarget( otherChunk ) + forcedOffset
  
        # Overlap is extracted
        thisWindow, transWindow = this.copyOverlaps( otherChunk, localToTargetVector[2], (localToTargetVector[0],localToTargetVector[1]) )

        thisPositive = thisWindow>0
        thisWindow = np.where( thisPositive, thisWindow*10, thisWindow )
        #posThisWin = np.where( thisWindow>0, thisWindow, 0 )
        #posTransWin = np.where( transWindow>0, thisWindow, 0 )

        #possibleOverlap = min( np.sum(posThisWin), np.sum(posTransWin) )

        errorWindow = (thisWindow*transWindow)
        #mArea = np.sum(np.abs(errorWindow) ) 
        mArea = np.sum(np.abs(thisPositive*errorWindow) ) 
        
        #if (mArea<10): return np.nan, np.nan

        errorWindow = -np.minimum( errorWindow, 0 ) 
        
        if ( mArea == 0 ):
            return 1000000000, 0

        errorScore = 1000*np.sum(errorWindow)/mArea

        return errorScore, mArea
        
    def plotDifference( this, otherChunk:Chunk, forcedOffset:np.ndarray=np.zeros(3) ):
        """ determines the error between two images without using features """

        transOffset = otherChunk.getOffset()
        myOffset    = this.getOffset()

        toTransVector = transOffset - myOffset

        toTransVector += forcedOffset

        # Overlap is extracted
        thisWindow, transWindow = this.copyOverlaps( otherChunk, toTransVector[2], (toTransVector[0],toTransVector[1]) )
 

        errorWindow = (thisWindow*transWindow)
        mArea = np.sum(np.abs(errorWindow) ) 
        if (mArea<10): return np.nan, np.nan

        errorWindow = -np.minimum( errorWindow, 0 ) 

        errorScore = 1000*np.sum(errorWindow)/mArea

        """fancyPlot( thisWindow )
        fancyPlot( transWindow )
        fancyPlot( errorWindow )"""
        fancyPlot( (thisWindow-transWindow) ) 

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

        cWidth  = cXMax-cXMin
        cHeight = cYMax-cYMin

        thisLCXMin, thisLCYMin = cXMin-thisProbGrid.xAMin, cYMin-thisProbGrid.yAMin
        transLCXMin, transLCYMin = cXMin-nTransProbGrid.xAMin, cYMin-nTransProbGrid.yAMin

        thisWindow  = thisProbGrid.mapEstimate[ thisLCYMin:thisLCYMin+cHeight, thisLCXMin:thisLCXMin+cWidth ]
        transWindow = nTransProbGrid.mapEstimate[ transLCYMin:transLCYMin+cHeight, transLCXMin:transLCXMin+cWidth ]

        return thisWindow, transWindow

    def determineErrorKeypoints( this, otherChunk:Chunk ):

        this.cachedProbabilityGrid.extractDescriptors()
        otherChunk.cachedProbabilityGrid.extractDescriptors()

        image1 = np.uint8(255*(this.cachedProbabilityGrid.mapEstimate+1)/2)
        image2 = np.uint8(255*(otherChunk.cachedProbabilityGrid.mapEstimate+1)/2)

        thisKeypoints, thisDescriptors = this.cachedProbabilityGrid.asKeypoints, this.cachedProbabilityGrid.featureDescriptors
        otherKeypoints, otherDescriptors = otherChunk.cachedProbabilityGrid.asKeypoints, otherChunk.cachedProbabilityGrid.featureDescriptors
        
        transSet = this.getLocalOffsetFromTarget( otherChunk )
        transRotation = rotationMatrix( -transSet[2] )
        transVector = np.array((transSet[0], transSet[1]))*this.cachedProbabilityGrid.cellRes
        
        origin1 = np.array(( this.cachedProbabilityGrid.xAMin, this.cachedProbabilityGrid.yAMin ))
        origin2 = np.array(( otherChunk.cachedProbabilityGrid.xAMin, otherChunk.cachedProbabilityGrid.yAMin ))

        if ( True ):
            rawKP1 = np.array([ keyPoint.pt for keyPoint in thisKeypoints ]) 
            rawKP2 = np.array([ keyPoint.pt for keyPoint in otherKeypoints ]) 
            
            trnsKP2 = np.dot(rawKP2 + origin2, transRotation) + ( transVector - origin1 )
            
            plt.figure(1934)
            plt.imshow( this.cachedProbabilityGrid.mapEstimate, origin="lower" )
            plt.plot( rawKP1[:,0], rawKP1[:,1], "rx" )
            plt.plot( trnsKP2[:,0], trnsKP2[:,1], "bx" )
            plt.show(block=False)
            ""
        
        if ( True ):
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
         
        kVal = 3
        
        #flann = cv2.FlannBasedMatcher(index_params, search_params)
        #matches = flann.knnMatch(thisDescriptors, otherDescriptors, k=kVal)
        
        bf = cv2.BFMatcher() 
        matches = bf.knnMatch(thisDescriptors, otherDescriptors, k=kVal)  
        
        # Extract keypoints
        src_I   = []
        src_pts = []
        #np.float32([thisKeypoints[m.queryIdx].pt for m, n in matches]).reshape(-1, 1, 2)
        dst_pts = [] 
        flatMatches = []
        #np.float32([otherKeypoints[m.trainIdx].pt for m, n in matches]).reshape(-1, 1, 2)
        
        offSetScale = 0
        
        for matchSet in matches:
            for match in matchSet:
                srcKeypoint = thisKeypoints[match.queryIdx]
                dstKeypoint = otherKeypoints[match.trainIdx]
                
                #src_pts.append( (srcKeypoint.pt[0] + offSetScale*np.cos( srcKeypoint.angle ), srcKeypoint.pt[1] + offSetScale*np.sin( srcKeypoint.angle )) )
                #dst_pts.append( (dstKeypoint.pt[0] + offSetScale*np.cos( dstKeypoint.angle ), dstKeypoint.pt[1] + offSetScale*np.sin( dstKeypoint.angle )) )
                src_I.append( match.queryIdx )
                src_pts.append( srcKeypoint.pt )
                dst_pts.append( dstKeypoint.pt )
                flatMatches.append( match )
                
        src_pts = np.float32(src_pts).reshape(-1, 2)
        dst_pts = np.float32(dst_pts).reshape(-1, 2) 
        
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
            
        
        # Apply RANSAC to estimate affine transformation
        #affine_matrix, inliers = cv2.estimateAffinePartial2D(src_pts, dst_pts, ransacReprojThreshold=10.0, method=cv2.RANSAC )  
        
        model = EuclideanTransform()
        model.estimate( src_pts_filt, dst_pts_filt )
        
        model_robust, inliers = ransac(
            (src_pts_filt, dst_pts_filt), EuclideanTransform, min_samples=2, residual_threshold=2, max_trials=100
        )
        
        accRotation = np.rad2deg( model_robust.rotation )
        accTrans = model_robust.translation
        
        #tform = transform.estimate_transform('euclidean', src_pts.reshape(-1, 2), dst_pts.reshape(-1, 2)  )
 
        #x_translation = affine_matrix[0, 2]
        #y_translation = affine_matrix[1, 2]
        #angle = np.arctan2(affine_matrix[1, 0], affine_matrix[0, 0]) * 180.0 / np.pi
 
        inMatches = [flat_match_filt[i] for i, inlier in enumerate(inliers) if inlier]
        
        

        # Draw filtered matches
        matched_image = cv2.drawMatches(image1, thisKeypoints, image2, otherKeypoints, inMatches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS) 
        # Display the matched image
        cv2.imshow('Matches', matched_image)

        # Draw filtered matches
        matched_image = cv2.drawMatches(image1, thisKeypoints, image2, otherKeypoints, flatMatches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS) 
        # Display the matched image
        cv2.imshow('Matches flat', matched_image)

        ""




    """ SECTION - full chunk layer manipulation """

    def centredFeaturelessErrorReduction( this, minMethod:bool ):
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
                    centreChunk.determineErrorFeaturelessDirect( targetChunk, 8, np.zeros(3), True )
                    #centreChunk.plotDifference( targetChunk )
                    #plt.show(block=False)
                    ""

    def linearFeaturelessErrorReduction( this, skipSize ):
        """ sequentially through all children of this chunk, reducing the error using the selected reduction method
        it applies the reduction by comparing all children to the centre
           """  
        for i in range(0, len(this.subChunks)):
            targetChunk = this.subChunks[i]
            if ( not targetChunk.isCentre ):
                rootChunk = this.subChunks[i + (skipSize if i<this.centreChunkIndex else -skipSize)]
                
                rootChunk.determineErrorFeaturelessDirect( targetChunk, 8, np.zeros(3), True )
        

    def linearPrune( this, skipSize, pruneMult=1.5 ):
        """ sequentially through all children of this chunk, reducing the error using the selected reduction method
        it applies the reduction by comparing all children to the centre
           """  
        
        errors = []

        for i in range(0, len(this.subChunks)):
            targetChunk = this.subChunks[i]
            if ( not targetChunk.isCentre ):
                rootChunk:Chunk = this.subChunks[i + (skipSize if i<this.centreChunkIndex else -skipSize)]
                
                error, overlap = rootChunk.determineDirectDifference( targetChunk )

                errors.append( error )

        errors = np.array( errors )
        maxError = np.median( errors )*pruneMult

        pruneTargets = np.where( errors>maxError )[0]

        this.deleteSubChunks( pruneTargets.tolist() )

        
        


