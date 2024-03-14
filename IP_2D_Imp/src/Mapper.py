

import numpy as np
from ProbabilityGrid import ProbabilityGrid, ScanFrame
from Navigator import Navigator, CartesianPose
from dataclasses import dataclass, field

from ImageProcessor import ImageProcessor 
from typing import Union
from random import random, seed
from scipy.signal import convolve2d

from matplotlib import pyplot as plt 

def plot_image_gradient(  xG, yG):
    # Get the dimensions of the image
    height, width = xG.shape[:2]

    # Create X and Y arrays using numpy.meshgrid
    x = np.linspace(0, width - 1, width)
    y = np.linspace(0, height - 1, height)
    X, Y = np.meshgrid(x, y) 

    # Create a quiver plot to visualize the gradient
    plt.figure(15233)
    plt.imshow(xG, extent=(x.min(), x.max(), y.min(), y.max()), origin='lower', cmap='viridis')
    plt.colorbar(label='Image Intensity')

    plt.quiver(X, Y, xG, yG, color='red', scale=0.1, scale_units='xy', angles='xy')
    plt.title('2D Image Gradient')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')

def acuteAngle( a1, a2 ) -> float:
    """Returns the acute angle between a1 and a2 in radians"""
    absD = abs(a1-a2)%(np.pi*2)
    return min( absD, 2*np.pi - absD )

@dataclass
class MapperConfig:
    # Core mapper settings
    GRID_RESOLUTION = 40
    MAX_FRAMES_MERGE = 7
    MAX_INTER_FRAME_ANGLE =  np.rad2deg(20)

    # Image estimation
    IE_OBJECT_SIZE = 0.25
    IE_SHARPNESS   = 2.6

    # Corner detector
    CORN_PEAK_SEARCH_RADIUS = 3
    CORN_PEAK_MIN_VALUE     = 0.15/1000
    CORN_DET_KERN           = ImageProcessor.gaussian_kernel(7, 2)

    # Corner descriptor
    DESCRIPTOR_RADIUS   = 5
    DESCRIPTOR_CHANNELS = 12

    # Feature descriptor comparison
    #DCOMP_COMPARISON_RADIUS    = 1 # The permittable mismatch between the descriptors keychannels when initially comparing 2 descriptors
    DCOMP_COMPARISON_COUNT     = 1 # The number of descriptor keychannels to used when initially comparing 2 descriptors 

class ProcessedScan:
    rawScans: list[ScanFrame]

    constructedProbGrid: ProbabilityGrid = None
    estimatedMap: np.ndarray = None
    featurePositions : np.ndarray = None
    featureDescriptors: np.ndarray = None
    scanPose: CartesianPose = None
    
    offsetPose: CartesianPose = None

    # Maps feature descriptors by key channels, structure: dict[channelHash] = [ [x, y], [channel data] ]
    featureDict: dict[int, list[np.ndarray]] = None

    def __init__(this, inpScanFrames: list[ScanFrame], offsetPose:CartesianPose=CartesianPose.zero() ) -> None:
        this.rawScans = inpScanFrames
        this.offsetPose = offsetPose

    #, clearExcessData:bool = False
    def estimateMap( this, gridResolution:float, estimatedWidth:float, sharpnessMult=2.5 ) -> None:
        """ This constructs a probability grid from the scan data then preforms additional processing to account for object infill """
        this.constructedProbGrid, midScan = ProbabilityGrid.initFromScanFramesPoly( gridResolution, this.rawScans, 8, this.offsetPose )
        this.scanPose = midScan.pose.copy() 

        this.estimatedMap = ImageProcessor.estimateFeatures( this.constructedProbGrid, estimatedWidth, sharpnessMult ) - this.constructedProbGrid.negativeData/2
    
    # TODO consider applying a guassian weighting to the features extracted, such that orientations are weighted higher near the centre
    def extractFeatures( this, keychannelNumb, cornerKernal=ImageProcessor.gaussian_kernel(9, 4), maximaSearchRad=3, minMaxima=-1, featureRadius=4, descriptorChannels=12 ):
        """ Extracts the positions and descriptors of intrest points """
        lambda_1, lambda_2, Rval = ImageProcessor.guassianCornerDist( this.estimatedMap, cornerKernal )
        
        this.featurePositions, vals = ImageProcessor.findMaxima( Rval, maximaSearchRad, minMaxima )  
        
 
        this.featureDescriptors = ImageProcessor.extractOrientations( this.estimatedMap, this.featurePositions[:,0], this.featurePositions[:,1], featureRadius, descriptorChannels )

        this.featureDict = {}

        for descriptor, position in zip(this.featureDescriptors, this.featurePositions):
                # It gets the indecies of the largest descriptor channels for DCOMP_COMPARISON_COUNT channels
                keypointIndecies = np.argpartition(descriptor, -keychannelNumb)[-keychannelNumb:]

                # It uses these channels to create a key, then adds to the map
                descriptorKey = hash(keypointIndecies.tobytes())

                if descriptorKey in this.featureDict:
                    this.featureDict[descriptorKey].append( [position, descriptor] )
                else:
                    this.featureDict[descriptorKey] = [ [position, descriptor] ]

class Mapper:
    navigator: Navigator
    allScans: list[ProcessedScan]
    scanBuffer: list[ScanFrame] 
     
    allRawScans: list[ScanFrame]
    config: MapperConfig
    
    def __init__(this, navigator:Navigator, config:MapperConfig ) -> None:
        this.navigator = navigator
        this.allScans  = []
        this.scanBuffer = []
        this.allRawScans = []
        this.config = config
    
    def analyseRecentScan( this ) -> Union[None, ProcessedScan]:
        """ Gets the most recent scan from the scan buffer and performs analysis on it """

        if ( len( this.allScans ) == 0 ):
            return None

        recentScan = this.allScans[-1]
        
        if ( recentScan.featureDescriptors is None ):
            recentScan.estimateMap( this.config.GRID_RESOLUTION, this.config.IE_OBJECT_SIZE, this.config.IE_SHARPNESS )
            recentScan.extractFeatures( this.config.DCOMP_COMPARISON_COUNT, this.config.CORN_DET_KERN, this.config.CORN_PEAK_SEARCH_RADIUS, this.config.CORN_PEAK_MIN_VALUE, this.config.DESCRIPTOR_RADIUS, this.config.DESCRIPTOR_CHANNELS )

            return recentScan
        return None
        
    

    def pushLidarScan( this, lidarOup ) -> None: 
        """ Pushed the scan frame onto the scan buffer and merges to make a completed image if conditions met """
        
        angles  = -lidarOup.angle_min - np.arange( 0, len(lidarOup.ranges) )*lidarOup.angle_increment
        lengths = np.array(lidarOup.ranges)
        cPose   = this.navigator.currentPose.copy()
        
        this.pushScanFrame( ScanFrame( scanAngles=angles, scanDistances=lengths, pose=cPose )  )

    def pushScanFrame( this, newScan:ScanFrame  ) -> None: 
        this.allRawScans.append( newScan )

        # Under target conditions the scan buffer gets merged to make a new frame
        if ( len(this.scanBuffer) >= this.config.MAX_FRAMES_MERGE or ( len(this.scanBuffer) > 0 and acuteAngle( newScan.pose.yaw, this.scanBuffer[0].pose.yaw )  > this.config.MAX_INTER_FRAME_ANGLE) ): 
            this.allScans.append( ProcessedScan( this.scanBuffer ) )
            
            this.scanBuffer = [ newScan ] 
        else:
            this.scanBuffer.append( newScan ) 
    
    def compareScans2( this, scan1: ProcessedScan, scan2: ProcessedScan, RANSACattempts=5, maxSQSepError=0.4**2 ):  
        matchingSet1 = [] 
        matchingSet2 = []  

        set1size = 0
        set2size = 0

        for scanKey in scan1.featureDict.keys():
            if ( scanKey in scan2.featureDict ):
                matchingSet1.append( scan1.featureDict[scanKey] )
                matchingSet2.append( scan2.featureDict[scanKey] )

                set1size += len(matchingSet1[-1])
                set2size += len(matchingSet2[-1])

        # Done for debugging purposes (repeatability)
        seed( set1size*set2size + set1size )

        """# Ensures that set1 is the larger one
        flipped = False
        if ( set1size < set2size ):
            set1size, set2size = set2size, set1size
            matchingSet1, matchingSet2 = matchingSet2, matchingSet1
            flipped = True"""

        # These are randomly sampled values from set1, set2 
        sampleSet1  = []
        sampleSet2  = []
        sampleCount = RANSACattempts*2
        for i in range(0, sampleCount):
            # Selects a random group, weighted according to group size
            point1SumIndex = int(random()*set1size)
            pointGroup = 0
            while ( point1SumIndex > 0 ): 
                point1SumIndex -= len(matchingSet1[pointGroup])
                pointGroup += 1
            pointGroup -= 1
            
            
            # Selects a random group, weighted according to group size
            point1SumIndex = int(random()*set1size)
            pointGroup2 = 0
            while ( point1SumIndex > 0 ): 
                point1SumIndex -= len(matchingSet1[pointGroup2])
                pointGroup2 += 1
            pointGroup2 -= 1
            
            #sampleGroup.append( pointGroup )
            # Selects random points from within that group TODO fix groups to work well, currently not using group system
            sampleSet1.append( matchingSet1[pointGroup][ int(len(matchingSet1[pointGroup])*random()) ] ) 
            sampleSet2.append( matchingSet2[pointGroup2][ int(len(matchingSet2[pointGroup2])*random()) ] ) 
        
        # Actually performs ransac
        # TODO fix sample generation to ensure duplicates can't happen and overlaps can't happen
        for i in range(0, RANSACattempts):
            index1 = int( (random()+1)*sampleCount/2 )
            index2 = int(random()*sampleCount/2)
            
            set1Point1 = sampleSet1[ index1 ]
            set1Point2 = sampleSet1[ index2 ]
            set2Point1 = sampleSet2[ index1 ]
            set2Point2 = sampleSet2[ index2 ]
            
            # Ensure no duplicate sample points
            if ( not (np.array_equiv(set1Point1[0], set1Point2[0]) or np.array_equiv(set2Point1[0], set2Point2[0])) ):
                
                # Checks that the match is resonable, (the distance between sets of points is simular)
                seperationErrorSQ = abs(np.sum(np.square(set1Point1[0]-set1Point2[0])) - np.sum(np.square(set2Point1[0]-set2Point2[0]))) / scan1.constructedProbGrid.cellRes**2
                if ( seperationErrorSQ < maxSQSepError ):
                    vec1 = set1Point2[0]-set1Point1[0]
                    vec2 = set2Point2[0]-set2Point1[0]
                    
                    angleChange    = np.arctan2( vec2[1], vec2[0] ) - np.arctan2( vec1[1], vec1[0] )
                    positionChange = (set2Point1[0]-set1Point1[0]) / scan1.constructedProbGrid.cellRes
                    
                    #matchValues, matchIndecies = this.determineImageMatchSuccess( scan2, scan1, angleChange, positionChange )
                    
                    ""
                
            else:
                ""
        ""
                
        
    
    def computeAllFeatureMatches( this, scan1: ProcessedScan, scan2: ProcessedScan, minMatch=-1 ):  
        """ Naive approach """
        matchScores   = []
        matchIndecies = [] 

        for scan1Feature, ind1 in zip(scan1.featureDescriptors, range(0, scan1.featureDescriptors.shape[0])):
            for scan2Feature, ind2 in zip(scan2.featureDescriptors, range(0, scan2.featureDescriptors.shape[0])):
                matchScores.append( np.sum(np.abs(scan1Feature - scan2Feature )) )
                matchIndecies.append( (ind1, ind2) )
                

        matchScores = np.array(matchScores)
        matchIndecies = np.array(matchIndecies)

        if ( minMatch == -1 ):
            return matchScores, matchIndecies
        
        mask = matchScores > minMatch
        return matchScores[mask], matchIndecies[mask]
        
    def copyScanWithOffset( this, targetScan:ProcessedScan, offsetPose:CartesianPose ):
        offsetScan = ProcessedScan( targetScan.rawScans, offsetPose )
        
        offsetScan.estimateMap( this.config.GRID_RESOLUTION, this.config.IE_OBJECT_SIZE, this.config.IE_SHARPNESS )
         
        return offsetScan
        
    
    def determineImageTranslation( this, modScan: ProcessedScan, refScan: ProcessedScan, rotation:float, translation: Union[float, float], show=False ):
        """ compares two scans by translating them as specified, then calculating ... """
 
        transScan = this.copyScanWithOffset( modScan, CartesianPose( translation[0], translation[1], 0, 0, 0, rotation ) )
         
        # +-1 offsets done to discriminate error prone edges
        xAMin = max(refScan.constructedProbGrid.xAMin, transScan.constructedProbGrid.xAMin)+1
        xAMax = min(refScan.constructedProbGrid.xAMax, transScan.constructedProbGrid.xAMax)-1
        yAMin = max(refScan.constructedProbGrid.yAMin, transScan.constructedProbGrid.yAMin)+1
        yAMax = min(refScan.constructedProbGrid.yAMax, transScan.constructedProbGrid.yAMax)-1
        
        # The overlap frames limits in grid cooridiantes (an integer scaled by resolution)
        absLimits = np.array( [ [xAMin, yAMin], 
                                [xAMax, yAMax] ] )
        
        # Translates overlap coordiantes to be in localised probability grid coordiantes
        transLimits = (absLimits - np.array([transScan.constructedProbGrid.xAMin, transScan.constructedProbGrid.yAMin])) 
        refLimits   = (absLimits - np.array([refScan.constructedProbGrid.xAMin, refScan.constructedProbGrid.yAMin])) 
        
        # Extracts the region of intrest from both images
        transWindow = transScan.estimatedMap.copy()[ transLimits[0][1]:transLimits[1][1], transLimits[0][0]:transLimits[1][0] ]
        refWindow   = refScan.estimatedMap.copy()[ refLimits[0][1]:refLimits[1][1], refLimits[0][0]:refLimits[1][0] ]
         
        
        mask = ((transWindow!=0) * (refWindow!=0))
        transWindow *= mask
        refWindow   *= mask 
        
        intrestMask = (convolve2d( refWindow>0, np.ones((13,13)), mode="same" )>0)*(convolve2d( transWindow>0, np.ones((13,13)), mode="same" )>0)
        
        errorWindow = (refWindow - transWindow) *intrestMask *np.abs(refWindow * transWindow)
         
        if (show):
            plt.figure(20)
            plt.imshow(refWindow - transWindow, origin='lower' )   
            plt.show() 
        
        kernal = ImageProcessor.gaussian_kernel( 7, 2 )
         
        erDy, erDx = np.gradient( errorWindow )  
        
        erDy = convolve2d( erDy, kernal, mode="same" ) *(refWindow<0)
        erDx = convolve2d( erDx, kernal, mode="same" ) *(refWindow<0)
        
        # Imperical adjustments made to make final errors more accurate
        erDy = 0.3*convolve2d( erDy, kernal, mode="same" ) 
        erDx = 0.7*convolve2d( erDx, kernal, mode="same" ) 
        
        # Length scale keeps errors consistantly sized
        lengthScale = np.sqrt(np.sum(transWindow*(transWindow>0)))
        yError = np.sum(erDy)/lengthScale
        xError = np.sum(erDx)/lengthScale 
 

        rows, cols = errorWindow.shape  
        x_coords, y_coords = np.meshgrid(np.arange(cols), np.arange(rows)) 
        origin = (refScan.scanPose.x*refScan.constructedProbGrid.cellRes - xAMin, refScan.scanPose.y*refScan.constructedProbGrid.cellRes - yAMin)
        x_offset = x_coords - origin[0]
        y_offset = y_coords - origin[1]

        # tangential and normal vectors are scaled inversely proportional to origin seperation (to account for increased offsets assosiated with larger seperation)
        sepLen = ( np.square(x_offset) + np.square(y_offset) )
        x_tangential_vector = y_offset/sepLen
        y_tangential_vector = -x_offset/sepLen 

        angleError = -np.sum(x_tangential_vector*erDx + y_tangential_vector*erDy)/lengthScale
        angleError *= 60

        xError -= (1-np.cos(angleError))
        yError -= (np.sin(angleError))   

        errorWindow *= refWindow<0
        errorWindow = np.where( (errorWindow)==0, np.inf, errorWindow ) 

        return xError, yError, angleError
    
    def determineImageTranslationLEGACY( this, modScan: ProcessedScan, refScan: ProcessedScan, rotation:float, translation: Union[float, float], show=False ):
        """ compares two scans by translating them as specified, then calculating ... """
 
        transScan = this.copyScanWithOffset( modScan, CartesianPose( translation[0], translation[1], 0, 0, 0, rotation ) )
         
        # +-1 offsets done to discriminate error prone edges
        xAMin = max(refScan.constructedProbGrid.xAMin, transScan.constructedProbGrid.xAMin)+1
        xAMax = min(refScan.constructedProbGrid.xAMax, transScan.constructedProbGrid.xAMax)-1
        yAMin = max(refScan.constructedProbGrid.yAMin, transScan.constructedProbGrid.yAMin)+1
        yAMax = min(refScan.constructedProbGrid.yAMax, transScan.constructedProbGrid.yAMax)-1
        
        # The overlap frames limits in grid cooridiantes (an integer scaled by resolution)
        absLimits = np.array( [ [xAMin, yAMin], 
                                [xAMax, yAMax] ] )
        
        # Translates overlap coordiantes to be in localised probability grid coordiantes
        transLimits = (absLimits - np.array([transScan.constructedProbGrid.xAMin, transScan.constructedProbGrid.yAMin])) 
        refLimits   = (absLimits - np.array([refScan.constructedProbGrid.xAMin, refScan.constructedProbGrid.yAMin])) 
        
        # Extracts the region of intrest from both images
        transWindow = transScan.estimatedMap.copy()[ transLimits[0][1]:transLimits[1][1], transLimits[0][0]:transLimits[1][0] ]
        refWindow   = refScan.estimatedMap.copy()[ refLimits[0][1]:refLimits[1][1], refLimits[0][0]:refLimits[1][0] ]
         
        
        mask = ((transWindow!=0) * (refWindow!=0))
        transWindow *= mask
        refWindow   *= mask 
        
        intrestMask = (convolve2d( refWindow>0, np.ones((13,13)), mode="same" )>0)*(convolve2d( transWindow>0, np.ones((13,13)), mode="same" )>0)
        
        errorWindow = (refWindow - transWindow) *intrestMask *np.abs(refWindow * transWindow)
        #plt.figure(25)
        #plt.imshow(errorWindow, origin='lower' ) 
        if (show):
            plt.figure(20)
            plt.imshow(refWindow - transWindow, origin='lower' )   
            plt.show()
        #errorWindow *= np.abs(errorWindow)
        
        kernal = ImageProcessor.gaussian_kernel( 7, 2 )
         
        erDy, erDx = np.gradient( errorWindow )

        curl = (np.gradient(erDy)[0] - np.gradient(erDx)[1] ) *(refWindow<0) 

        """rows, cols = errorWindow.shape  
        x_coords, y_coords = np.meshgrid(np.arange(cols), np.arange(rows))
        x_coords = (x_coords+(xAMin - refScan.scanPose.x*refScan.constructedProbGrid.cellRes)).flatten()
        y_coords = (y_coords+(yAMin - refScan.scanPose.y*refScan.constructedProbGrid.cellRes)).flatten()
        e_vals   = errorWindow.flatten()
        pflat_mask = (e_vals)>0.1
        nflat_mask = (e_vals)<-0.1
        mags = np.sqrt( np.square( x_coords ) + np.square( y_coords ) ) 
        rads = np.arctan2( y_coords, x_coords ) 
        
        plt.figure(40)
        plt.xlabel( "magnitude" )
        plt.ylabel( "angle" )
        plt.plot( mags[nflat_mask], rads[nflat_mask], "rx" )
        plt.plot( mags[pflat_mask], rads[pflat_mask], "bx" )"""
        
        erDy = convolve2d( erDy, kernal, mode="same" ) *(refWindow<0)
        erDx = convolve2d( erDx, kernal, mode="same" ) *(refWindow<0)
        
        # Imperical adjustments made to make final errors more accurate
        erDy = 0.3*convolve2d( erDy, kernal, mode="same" ) 
        erDx = 0.7*convolve2d( erDx, kernal, mode="same" ) 
        
        lengthScale = np.sqrt(np.sum(transWindow*(transWindow>0)))
        yError = np.sum(erDy)/lengthScale
        xError = np.sum(erDx)/lengthScale 
 

        rows, cols = errorWindow.shape  
        x_coords, y_coords = np.meshgrid(np.arange(cols), np.arange(rows)) 
        origin = (refScan.scanPose.x*refScan.constructedProbGrid.cellRes - xAMin, refScan.scanPose.y*refScan.constructedProbGrid.cellRes - yAMin)
        x_offset = x_coords - origin[0]
        y_offset = y_coords - origin[1]

        sepLen = ( np.square(x_offset) + np.square(y_offset) )
        x_tangential_vector = y_offset/sepLen
        y_tangential_vector = -x_offset/sepLen

        refCOM = (np.sum( np.maximum(refWindow,0)*intrestMask*x_coords )/np.sum( np.maximum(refWindow,0)*intrestMask ), np.sum( np.maximum(refWindow,0)*intrestMask*y_coords )/np.sum( np.maximum(refWindow,0)*intrestMask ))
        traCOM = (np.sum( np.maximum(transWindow,0)*intrestMask*x_coords )/np.sum( np.maximum(transWindow,0)*intrestMask ), np.sum( np.maximum(transWindow,0)*intrestMask*y_coords )/np.sum( np.maximum(transWindow,0)*intrestMask ))
        plt.plot( [ origin[0], refCOM[0] ], [ origin[1], refCOM[1] ], "bx-" )   
        plt.plot( [ origin[0], traCOM[0] ], [ origin[1], traCOM[1] ], "rx-" )    
        
        #plt.figure(99) 
        #plt.colorbar( plt.imshow( x_tangential_vector*erDx + y_tangential_vector*erDy, origin='lower' ) )

        angleError = -np.sum(x_tangential_vector*erDx + y_tangential_vector*erDy)/lengthScale
        angleError *= 60

        xError -= (1-np.cos(angleError))
        yError -= (np.sin(angleError)) 
         
        #print( "dx:",xError,"  dy:",yError,"  da:", np.rad2deg(angleError) )

        """lump = np.sum( np.maximum( refWindow*intrestMask, 0 ), axis=0 )

        plt.figure(40) 
        plt.plot( np.sum( erDy, axis=0 )/lump, "r-" )
        plt.plot( np.sum( erDx, axis=0 )/lump, "b-" ) 
 
        plt.figure(41) 
        plt.plot( lump, "r-" ) """
        
        # applying exponentials leads to non zero sums of gradients (steep gradients are bonused)
        
        #plot_image_gradient( erDx, erDy )  

        errorWindow *= refWindow<0
        errorWindow = np.where( (errorWindow)==0, np.inf, errorWindow )
        
        """plt.figure(21)
        plt.imshow(  np.where( erDy==0, np.inf, erDy ), origin='lower' ) 
        
        plt.figure(22)
        plt.imshow(  np.where( erDx==0, np.inf, erDx ), origin='lower' ) 
        
        
        plt.figure(23)
        plt.imshow(  transWindow, origin='lower' ) 
        
        plt.figure(24) 
        plt.imshow(  refWindow, origin='lower' )  """
         
        #plt.show()
        """"""
        return xError, yError, angleError
    
    def determineImageMatchSuccess( this, modScan: ProcessedScan, refScan: ProcessedScan, rotation:float, translation: Union[float, float] ):
        """ compares two scans by translating them as specified, then calculating the average diffecence between the scans """
 
        transScan = this.copyScanWithOffset( modScan, CartesianPose( translation[0], translation[1], 0, 0, 0, rotation ) )
         
        # +-1 offsets done to discriminate error prone edges
        xAMin = max(refScan.constructedProbGrid.xAMin, transScan.constructedProbGrid.xAMin)+1
        xAMax = min(refScan.constructedProbGrid.xAMax, transScan.constructedProbGrid.xAMax)-1
        yAMin = max(refScan.constructedProbGrid.yAMin, transScan.constructedProbGrid.yAMin)+1
        yAMax = min(refScan.constructedProbGrid.yAMax, transScan.constructedProbGrid.yAMax)-1
        
        # The overlap frames limits in grid cooridiantes (an integer scaled by resolution)
        absLimits = np.array( [ [xAMin, yAMin], 
                                [xAMax, yAMax] ] )
        
        # Translates overlap coordiantes to be in localised probability grid coordiantes
        transLimits = (absLimits - np.array([transScan.constructedProbGrid.xAMin, transScan.constructedProbGrid.yAMin])) 
        refLimits   = (absLimits - np.array([refScan.constructedProbGrid.xAMin, refScan.constructedProbGrid.yAMin])) 
        
        # Extracts the region of intrest from both images
        transWindow = transScan.estimatedMap.copy()[ transLimits[0][1]:transLimits[1][1], transLimits[0][0]:transLimits[1][0] ]
        refWindow   = refScan.estimatedMap.copy()[ refLimits[0][1]:refLimits[1][1], refLimits[0][0]:refLimits[1][0] ]
        
        mask = ((transWindow!=0) * (refWindow!=0))
        transWindow *= mask
        refWindow   *= mask
        
        mArea = np.sum(np.abs( transWindow ) + np.abs( refWindow ))
        
        errorWindow = (transWindow*refWindow)
        errorWindow = np.minimum( errorWindow, 0 )
        errorWindow *= np.abs(errorWindow)
        
        errorScore = -1000*np.sum(errorWindow)/mArea
        
        errorWindow = np.where( errorWindow==0, np.inf, errorWindow )
        
        """plt.figure(21)
        plt.imshow(  np.where( transWindow==0, np.inf, transWindow ), origin='lower' ) 
        
        plt.figure(22)
        plt.imshow(  np.where( refWindow==0, np.inf, refWindow ), origin='lower' ) 
        
        
        plt.figure(23)
        plt.imshow(  transScan.estimatedMap, origin='lower' ) 
        
        plt.figure(24)
        plt.imshow(  refScan.estimatedMap, origin='lower' ) 
        
        plt.figure(25)
        plt.colorbar(plt.imshow(  errorWindow, origin='lower' ))  
        plt.show()"""
        """"""
        return errorScore, mArea

    def compareScans( this, scan1: ProcessedScan, scan2: ProcessedScan ):
        
        if ( len(scan1.featureDescriptors) == 0 or len(scan2.featureDescriptors) == 0 ):
            return -1
            #raise RuntimeError( "atleast one scan has no features" )
        
        if ( scan1.featurePositions.size < scan2.featurePositions.size ):
            scan2, scan1 = scan1, scan2
 
        closestIndexes = []
        closestValues  = []
        for feat1, pos1 in zip( scan1.featureDescriptors, scan1.featurePositions  ):
            
            closestMatchIndex = -1
            closestMatchValue = 999999999999

            for feat2, pos2, I in zip( scan2.featureDescriptors, scan2.featurePositions, range( 0, scan2.featurePositions.size ) ):
                difference = np.sum(np.abs(feat2-feat1))

                if ( difference < closestMatchValue ):
                    closestMatchValue = difference
                    closestMatchIndex = I

            closestIndexes.append( closestMatchIndex ) 
            closestValues.append( closestMatchValue )       
        
        return 

            





        
        
        
        
        
        
        
        
        







