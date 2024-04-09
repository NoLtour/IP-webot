
import numpy as np

class IPConfig:
    # Core mapper settings
    GRID_RESOLUTION = 40
    OBJECT_DIAMETER = 0.5
    OBJECT_PIX_RADI = int(GRID_RESOLUTION*OBJECT_DIAMETER/2)
    OBJECT_PIX_DIAM = OBJECT_PIX_RADI*2 + 1
    MAX_FRAMES_MERGE = 8
    MAX_INTER_FRAME_ANGLE = np.rad2deg(2220)

    # Featureless adjuster
    FEATURELESS_SEARCH_DIST = 0.14 # Given in generic distance
    FEATURELESS_PIX_SEARCH_DIAM = int(GRID_RESOLUTION*FEATURELESS_SEARCH_DIST*2+1)
    FEATURELESS_X_ERROR_SCALE   = 0.1#0.06
    FEATURELESS_Y_ERROR_SCALE   = 0.07
    FEATURELESS_A_ERROR_SCALE   = 16#0.06
    FEATURELESS_COMP_FACT       = 0
    ANGLE_OVERWIRTE_THRESHOLD   = np.deg2rad( 4 )
    CONFLICT_MULT_GAIN          = 0.5

    # Scan propreties
    MAX_LIDAR_LENGTH = 8

    # Image estimation
    IE_OBJECT_DIAM = 0.5 *OBJECT_DIAMETER
    IE_SHARPNESS   = 2.6

    # Corner detector
    CORN_PEAK_SEARCH_RADIUS = int( OBJECT_PIX_RADI*0.5 + 0.5 )
    CORN_PEAK_MIN_VALUE     = 0.15/1000 # TODO, scaling

    # Corner descriptor
    DESCRIPTOR_RADIUS   = CORN_PEAK_SEARCH_RADIUS
    DESCRIPTOR_CHANNELS = 12

    # Feature descriptor comparison
    #DCOMP_COMPARISON_RADIUS    = 1 # The permittable mismatch between the descriptors keychannels when initially comparing 2 descriptors
    DCOMP_COMPARISON_COUNT     = 1 # The number of descriptor keychannels to used when initially comparing 2 descriptors 
 

""
