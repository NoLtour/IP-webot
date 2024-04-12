from __future__ import annotations

from RawScanFrame import RawScanFrame
from scipy.spatial import KDTree 
from ProbabilityGrid import ProbabilityGrid
from CommonLib import gaussianKernel, fancyPlot, solidCircle, generate1DGuassianDerivative

from scipy.signal import convolve2d
from scipy.ndimage import laplace, maximum_filter, minimum_filter, zoom
from scipy.optimize import minimize
from scipy.optimize import differential_evolution
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
    offsetFromParent: np.ndarray = None
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
            
            for otherChunk in this.subChunks:
                X, Y, yaw = otherChunk.determineInitialOffset()
                 
                otherChunk.offsetFromParent = np.array(( X, Y, yaw ))
    
    """ SECTION - data integrity managers """
    def clearCache( this ):
        """ Run when there are changes to this parents children which mean cached values are no longer valid
         this is done recursively to target all grand parents """
        this.cachedProbabilityGrid = None

        if ( this.parent != None ):
            this.parent.clearCache

    
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
    
    def getIntermsOfParent( this, otherChunk:Chunk, poseFromOther:np.ndarray ):
        """ returns the cartesian offset of this chunk from the parent given a target chunk with our offset from it, both chunks must
         hold the same parent """
        
        if ( this.parent != otherChunk.parent ):
            raise RuntimeError("Chunks don't share parent")
        
        return otherChunk.getOffset( ) + ( poseFromOther )
    
    def updateOffset( this, newOffset:np.ndarray ):
        """ updates this chunks offset from parent, setting it to the new value """

        this.offsetFromParent = newOffset
        this.parent.clearCache()

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
        """ estimates the poitional error between two images without using features """

        transOffset = otherChunk.getOffset()
        myOffset    = this.getOffset()

        toTransVector = transOffset - myOffset

        toTransVector += forcedOffset

        # Overlap is extracted
        thisWindow, transWindow = this.copyOverlaps( otherChunk, toTransVector[2], (toTransVector[0],toTransVector[1]) ) 
        
        # First the search region is defined
        searchKernal = solidCircle( this.config.FEATURELESS_PIX_SEARCH_DIAM )
        #convTrans = convolve2d( transWindow, gaussianKernel(this.config.FEATURELESS_PIX_SEARCH_DIAM/4), mode="same" )
        intrestMask = np.minimum((thisWindow>0.01)+(transWindow>0.01), 1)
        
        x1DGuas, y1DGuas = generate1DGuassianDerivative(this.config.FEATURELESS_PIX_SEARCH_DIAM/2)
        
        errorWindow = -np.minimum(thisWindow*transWindow, 0)
        errorWindow = -(thisWindow*transWindow) 
        #fancyPlot( errorWindow )
        #errorWindow = convolve2d( errorWindow, gaussianKernel(this.config.FEATURELESS_PIX_SEARCH_DIAM/4, 0.05), mode="same" )

        conflictWindow = (thisWindow*transWindow) 
        conflictWindow = -np.minimum( conflictWindow, 0 )  

        #fancyPlot( errorWindow )
        erDx = convolve2d( errorWindow, x1DGuas, mode="same" )
        erDy = convolve2d( errorWindow, y1DGuas, mode="same" )
        #erDy, erDx = np.gradient( errorWindow )*intrestMask
        
        lengthScale = np.sum(thisWindow*intrestMask*(thisWindow>0))/this.config.OBJECT_PIX_DIAM
        if ( lengthScale==0 ): return 0,0,0
        
        #fancyPlot( erDx )
        
        erDx = conflictWindow*np.where(thisWindow<0,erDx,-erDx)/lengthScale 
        erDy = conflictWindow*np.where(thisWindow<0,erDy,-erDy)/lengthScale  
        
        #fancyPlot( erDx )
        #fancyPlot( erDy )
        #fancyPlot( erDy )
        #plt.show( block=False )
        
        erDx = erDx*this.config.FEATURELESS_X_ERROR_SCALE
        erDy = erDy*this.config.FEATURELESS_Y_ERROR_SCALE
         
        maxErr = max(np.max( erDx ),np.max( erDy ))
        erDyMask = (np.abs(erDy)>0)
        erDxMask = (np.abs(erDx)>0)
        
        xError = np.sum(erDx)
        yError = np.sum(erDy)

        rows, cols = errorWindow.shape  
        x_coords, y_coords = np.meshgrid(np.arange(cols), np.arange(rows)) 
        origin = (-this.cachedProbabilityGrid.xAMin, -this.cachedProbabilityGrid.yAMin)
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

    def determineErrorFeatureless2( this, otherChunk:Chunk, forcedOffset:np.ndarray=np.zeros(3), showPlot=False ):
        """ estimates the poitional error between two images without using features """

        transOffset = otherChunk.getOffset()
        myOffset    = this.getOffset()

        toTransVector = transOffset - myOffset

        toTransVector += forcedOffset

        # Overlap is extracted
        thisWindow, transWindow = this.copyOverlaps( otherChunk, toTransVector[2], (toTransVector[0],toTransVector[1]) )

        #thisWindow = convolve2d( thisWindow, gaussianKernel(this.config.FEATURELESS_PIX_SEARCH_DIAM/4, 0.05), mode="same" )
        #transWindow = convolve2d( transWindow, gaussianKernel(this.config.FEATURELESS_PIX_SEARCH_DIAM/4, 0.05), mode="same" )
        
        # First the search region is defined
        searchKernal = solidCircle( this.config.FEATURELESS_PIX_SEARCH_DIAM )
        #convTrans = convolve2d( transWindow, gaussianKernel(this.config.FEATURELESS_PIX_SEARCH_DIAM/4), mode="same" )
        intrestMask = ((convolve2d( thisWindow>0, searchKernal, mode="same" ) )>0)*((convolve2d( transWindow>0, searchKernal, mode="same" ))>0)
        
        errorWindow = (thisWindow - transWindow) * intrestMask*np.abs(thisWindow * transWindow)
        errorWindow = convolve2d( errorWindow, gaussianKernel(this.config.FEATURELESS_PIX_SEARCH_DIAM/4, 0.05), mode="same" )

        conflictWindow = (thisWindow*transWindow*intrestMask)
        
        conflictWindow = -np.minimum( conflictWindow, 0 )  


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
         
        erDy, erDx = np.gradient( errorWindow ) 
        #erDy = convolve2d( erDy, kernal, mode="same" ) *(thisWindow<0)
        #erDx = convolve2d( erDx, kernal, mode="same" ) 

        # Length scale keeps errors consistantly sized, here the y and x error are non-dimensionalised
        #lengthScale = np.sqrt(np.sum(thisWindow*(thisWindow>0)))
        #lengthScale = np.sum(np.abs(erDy))+np.sum(np.abs(erDx))
        lengthScale = np.sum(thisWindow*intrestMask*(thisWindow>0))/this.config.OBJECT_PIX_DIAM
        if ( lengthScale==0 ): return 0,0,0
        conflictMultiplier = 1 + this.config.CONFLICT_MULT_GAIN*np.sum(conflictWindow)/lengthScale
        
        #conflictMultiplier = 1
        
        """fancyPlot( errorWindow )
        fancyPlot( (thisWindow<0)*erDy/lengthScale ) 
        fancyPlot( -((thisWindow>0)*erDy/lengthScale) )
        fancyPlot( np.where(thisWindow<0,erDy,-erDy)/lengthScale ) 
        plt.show()"""

        # Imperical adjustments made to make final errors more accurate
        #erDx = (thisWindow<0)*erDx/lengthScale 
        #erDy = (thisWindow<0)*erDy/lengthScale 
        
        
        erDx = np.where(thisWindow<0,erDx,-erDx)/lengthScale 
        erDy = np.where(thisWindow<0,erDy,-erDy)/lengthScale  
        
        #erDx = convolve2d( erDx, gaussianKernel(this.config.FEATURELESS_PIX_SEARCH_DIAM/4, 0.05), mode="same" )*this.config.FEATURELESS_X_ERROR_SCALE*conflictMultiplier
        #erDy = convolve2d( erDy, gaussianKernel(this.config.FEATURELESS_PIX_SEARCH_DIAM/4, 0.05), mode="same" )*this.config.FEATURELESS_Y_ERROR_SCALE*conflictMultiplier
        erDx = erDx*this.config.FEATURELESS_X_ERROR_SCALE*conflictMultiplier
        erDy = erDy*this.config.FEATURELESS_Y_ERROR_SCALE*conflictMultiplier
         
        maxErr = max(np.max( erDx ),np.max( erDy ))
        erDyMask = (np.abs(erDy)>maxErr*0.025)
        erDxMask = (np.abs(erDx)>maxErr*0.025)
        """erDx *= erDxMask
        erDy *= erDyMask"""
        
        xError = np.sum(erDx)
        yError = np.sum(erDy)

        rows, cols = errorWindow.shape  
        x_coords, y_coords = np.meshgrid(np.arange(cols), np.arange(rows)) 
        origin = (-this.cachedProbabilityGrid.xAMin, -this.cachedProbabilityGrid.yAMin)
        x_offset = (x_coords - origin[0])
        y_offset = (y_coords - origin[1]) 

        # tangential and normal vectors are scaled inversely proportional to origin seperation (to account for increased offsets assosiated with larger seperation)
        sepLen = np.maximum( ( np.square(x_offset) + np.square(y_offset) )/(this.config.GRID_RESOLUTION**2), 0.1 )
        x_tangential_vector = y_offset/sepLen
        y_tangential_vector = -x_offset/sepLen 

        """fancyPlot( errorWindow )
        fancyPlot( erDx )
        fancyPlot( erDy )
        fancyPlot( erDx - erDxMask*(np.sum(erDx)/np.sum(erDxMask)) )
        fancyPlot( erDy - erDyMask*(np.sum(erDy)/np.sum(erDyMask)) )
        plt.show(block=False)"""
        

        #angleError = -np.sum(x_tangential_vector*( erDx - erDxMask*(np.sum(erDx)/np.sum(erDxMask)) ) + y_tangential_vector*( erDy - erDyMask*(np.sum(erDy)/np.sum(erDyMask)) ))
        #angleError = -np.sum(x_tangential_vector*erDx + y_tangential_vector*erDy)

        erDa = -(x_tangential_vector*( erDxMask*(np.sum(erDx)/np.sum(erDxMask)) ) + y_tangential_vector*(erDyMask*(np.sum(erDy)/np.sum(erDyMask)) ))
        erDa = -(x_tangential_vector*erDx + y_tangential_vector*erDy) - erDa

        angleError = np.sum(erDa) * this.config.FEATURELESS_A_ERROR_SCALE

        """ratio = erDx/erDy
        aErDy = erDa/( x_tangential_vector*ratio + y_tangential_vector )
        aErDx = aErDy*ratio
        
        xError -= np.sum( aErDx )
        yError -= np.sum( aErDy )"""
        
        """angleError *= conflictMultiplier
        xError *= conflictMultiplier
        yError *= conflictMultiplier"""

        """if ( abs(angleError) > this.config.ANGLE_OVERWIRTE_THRESHOLD ):
            xError *= this.config.ANGLE_OVERWIRTE_THRESHOLD/angleError
            yError *= this.config.ANGLE_OVERWIRTE_THRESHOLD/angleError"""

        #aCompensation = np.arctan( np.array([yError]),np.array([xError]) )[0]  
        #angleError -= aCompensation*this.config.FEATURELESS_COMP_FACT

        # Reduce coupling effect
        xCompensation = (1-np.cos(angleError))
        yCompensation = (np.sin(angleError))

        #xError     -= xCompensation*this.config.FEATURELESS_COMP_FACT
        #yError     -= yCompensation*this.config.FEATURELESS_COMP_FACT

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
    
    def determineErrorFeatureless( this, otherChunk:Chunk, forcedOffset:np.ndarray=np.zeros(3), showPlot=False ):
        """ estimates the poitional error between two images without using features """

        transOffset = otherChunk.getOffset()
        myOffset    = this.getOffset()

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
        lengthScale = np.sqrt(np.sum(thisWindow*(thisWindow>0)))
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

        if ( True or showPlot ):
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

    def determineErrorFeaturelessMinimumOLD( this, otherChunk:Chunk, forcedOffset:np.ndarray=np.zeros(3), updateOffset=False ):
        """ This finds the relative offset between two chunks without using features, using scipy.optimize.minimum """
        transOffset = otherChunk.getOffset()
        myOffset    = this.getOffset()
        
        #print( "\n\ninitOffset:", this.getOffset()*40 )

        toTransVector = transOffset - myOffset + forcedOffset
        
        initErrorScore, overlapArea = this.determineDirectDifference( otherChunk, toTransVector, True ) 
 
        def interpFunc( offsets ):
            error, area = this.determineDirectDifference( otherChunk, offsets, True )

            if (np.isnan(error)):
                return 9999999999
            
            return error

        initChange = np.array(( 5/this.config.GRID_RESOLUTION, 5/this.config.GRID_RESOLUTION, np.deg2rad(5) ))
        nm = minimize( interpFunc, toTransVector,  method="COBYLA", options={ 'maxiter': this.config.MINIMISER_MAX_LOOP } )
    
        trueOffset = ( nm.x[0], nm.x[1], nm.x[2] ) 
        errorScore, overlapArea = this.determineDirectDifference( otherChunk, trueOffset, True )
        
        foundDirectionErrors =  toTransVector - trueOffset

        if (updateOffset):
             this.updateOffset(
                 this.getIntermsOfParent( otherChunk, nm.x )
             )
             
        #print( "change:", nm.x*40 )
        #print( "newOffset:", this.getOffset()*40 ) 
        if ( initErrorScore < errorScore ):
            return np.zeros(3), initErrorScore, 1
        
        return foundDirectionErrors, errorScore, 1

    def determineErrorFeaturelessMinimum( this, otherChunk:Chunk, forcedOffset:np.ndarray=np.zeros(3), updateOffset=False ):
        """ This finds the relative offset between two chunks without using features, using scipy.optimize.minimum """
        transOffset = otherChunk.getOffset()
        myOffset    = this.getOffset()
        
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

    def determineDirectDifference( this, otherChunk:Chunk, forcedOffset:np.ndarray=np.zeros(3), completeOffsetOverride=False ):
        """ determines the error between two images without using features """

        toTransVector = 0
        if ( completeOffsetOverride ):
            toTransVector = forcedOffset
        else:
            transOffset = otherChunk.getOffset()
            myOffset    = this.getOffset()

            toTransVector = transOffset - myOffset
            toTransVector += forcedOffset

        # Overlap is extracted
        thisWindow, transWindow = this.copyOverlaps( otherChunk, toTransVector[2], (toTransVector[0],toTransVector[1]) )

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
            # TODO implement for other method
            raise RuntimeError("not implemented yet")


