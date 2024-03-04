

import numpy as np
from ProbabilityGrid import ProbabilityGrid, ScanFrame
from Navigator import Navigator
from dataclasses import dataclass, field

from ImageProcessor import ImageProcessor
from typing import Union
import random

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

    # Maps feature descriptors by key channels, structure: dict[channelHash] = [ [x, y], [channel data] ]
    featureDict: dict[int, list[np.ndarray]] = None

    def __init__(this, inpScanFrames: list[ScanFrame]) -> None:
        this.rawScans = inpScanFrames

    #, clearExcessData:bool = False
    def estimateMap( this, gridResolution:float, estimatedWidth:float, sharpnessMult=2.5 ) -> None:
        """ This constructs a probability grid from the scan data then preforms additional processing to account for object infill """
        this.constructedProbGrid = ProbabilityGrid.initFromScanFramesPoly( gridResolution, this.rawScans, 1, -0.2, 8 )

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
    
    def compareScans2( this, scan1: ProcessedScan, scan2: ProcessedScan, RANSACattempts=5 ):  
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
        random.seed( set1size*set2size + set1size )

        # Ensures that set1 is the larger one
        flipped = False
        if ( set1size < set2size ):
            set1size, set2size = set2size, set1size
            matchingSet1, matchingSet2 = matchingSet2, matchingSet1
            flipped = True

        # No checks to prevent duplicate checks
        for i in range(0, RANSACattempts):


            point1SumIndex = int(random()*set1size)

            sumSize    = 0
            pointGroup = 0
            while ( sumSize <= point1SumIndex ): 
                sumSize    += len(matchingSet1[pointGroup])
                pointGroup += 1
            pointGroup -= 1

            point1 = matchingSet1[pointGroup][point1SumIndex - sumSize]

        

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

            





        
        
        
        
        
        
        
        
        







