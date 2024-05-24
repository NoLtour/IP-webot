
import numpy as np

class IPConfig:
    # Core mapper settings
    GRID_RESOLUTION = 40
    OBJECT_DIAMETER = 0.5
    OBJECT_PIX_RADI = int(GRID_RESOLUTION*OBJECT_DIAMETER/2)
    OBJECT_PIX_DIAM = OBJECT_PIX_RADI*2 + 1
    MAX_FRAMES_MERGE = 1
    MAX_INTER_FRAME_ANGLE = np.deg2rad(5)

    # Featureless adjuster
    FEATURELESS_SEARCH_DIST = 0.14 # Given in generic distance
    FEATURELESS_PIX_SEARCH_DIAM = int(GRID_RESOLUTION*FEATURELESS_SEARCH_DIST*2+1)
    FEATURELESS_X_ERROR_SCALE   = 1
    FEATURELESS_Y_ERROR_SCALE   = 1
    FEATURELESS_A_ERROR_SCALE   = 1
    #CHANGE_SCALING_LIMIT_LIN    = 0.1
    #CHANGE_SCALING_LIMIT_MUL    = 3
    
    ITERATIVE_REDUCTION_MULTIPLIER = 0.4
    ITERATIVE_REDUCTION_PERMITTED_ERROR = 5
    #FEATURELESS_COMP_FACT       = 0
    #ANGLE_OVERWIRTE_THRESHOLD   = np.deg2rad( 4 )
    #CONFLICT_MULT_GAIN          = 0

    # Minimiser featureless adjuster
    MINIMISER_MAX_LOOP = 150

    # Scan propreties
    MAX_LIDAR_LENGTH = 8

    # Image estimation
    IE_OBJECT_DIAM = 0.35 *OBJECT_DIAMETER
    IE_SHARPNESS   = 4.6

    # Corner detector
    CORN_PEAK_SEARCH_RADIUS = int( OBJECT_PIX_RADI*0.5 + 0.5 )
    CORN_PEAK_MIN_VALUE     = 0.15/1000 # TODO, scaling

    # Corner descriptor
    DESCRIPTOR_RADIUS   = CORN_PEAK_SEARCH_RADIUS
    DESCRIPTOR_CHANNELS = 12

    # Feature descriptor comparison
    #DCOMP_COMPARISON_RADIUS    = 1 # The permittable mismatch between the descriptors keychannels when initially comparing 2 descriptors
    DCOMP_COMPARISON_COUNT     = 1 # The number of descriptor keychannels to used when initially comparing 2 descriptors 
 
    FEATURE_COMPARISON_FILTER_DISTANCE = True
""
