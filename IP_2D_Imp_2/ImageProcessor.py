
import numpy as np
 
from scipy.signal import convolve2d
from scipy.ndimage import laplace, maximum_filter, minimum_filter, zoom
from CommonLib import solidCircle, gaussianKernel, convolveWithEdgeWrap
import random

import matplotlib.pyplot as plt

class ImageProcessor: 

    @staticmethod
    def guassianCornerDist( wallArray:np.ndarray, kernal  ):   
        """ Extracts the eigenvalues """ 

        Iy, Ix = np.gradient( wallArray ) 
        
        IxIx = convolve2d( np.square( Ix ), kernal, mode="same" ) 
        IxIy =  convolve2d( 2 * Ix * Iy , kernal, mode="same" )   
        IyIy =  convolve2d(np.square( Iy ) , kernal, mode="same" )
        
        #return eigvals( np.stack((IxIx, IxIy, IxIy, IyIy), axis=-1).reshape((*IxIx.shape, 2, 2)) )
        #return eigvals(np.array([[IxIx, IxIy], [IxIy, IyIy]]))
        
        ApC = IxIx + IyIy
        sqBAC = np.sqrt( np.square(IxIy) + np.square( IxIx - IyIy ) )
        
        lambda_1 = 0.5 * ( ApC + sqBAC )
        lambda_2 = 0.5 * ( ApC - sqBAC ) 
        
        Rval = lambda_1 * lambda_2 - 0.05*np.square( lambda_1 + lambda_2 ) 
        
        return lambda_1, lambda_2, Rval
 
    
    @staticmethod
    def findMaxima( inpArray:np.ndarray, maskSize = 3, threshHold = -1 ):
        filterDims = (maskSize,)*inpArray.ndim
        # in this format it's y,x
        localMaximums = np.where( (inpArray == maximum_filter(inpArray, size=filterDims, mode='constant'))  ) #  | (inpArray == minimum_filter(inpArray, size=filterDims, mode='constant')) 
        localIntensities = inpArray[ localMaximums ]
  
        if ( threshHold == -1 ):
            threshHold = np.max( np.abs(localIntensities) ) * 0.2
        
        mask = np.where( np.abs(localIntensities) > threshHold )
 
        return np.array(localMaximums[1])[ mask ], np.array(localMaximums[0])[ mask ], localIntensities[mask]

    @staticmethod
    def extractOrientations( inpArray:np.ndarray, pointXs:np.ndarray, pointYs:np.ndarray, radius:int, oRes=12 ):
        """ Produces a histogram with the centre shifted to be inline with the exponentially weighted "centre of mass" """

        # Fix ranges to fit
        yMins = np.maximum( pointYs-radius, 0 )  
        yMaxs = np.minimum( pointYs+radius, inpArray.shape[0]-1 )
        xMins = np.maximum( pointXs-radius, 0 )  
        xMaxs = np.minimum( pointXs+radius, inpArray.shape[1]-1 )
        
        outputs = []
        #guassian = gaussian_kernel2( radius*2 + 1 ) 
        
        angleArrange = np.arange( 0, oRes, 1 )*(2*np.pi/oRes)
        
        vectorX = np.cos( angleArrange )
        vectorY = np.sin( angleArrange )
        
        # Iterates through each search window
        for yMin, yMax, xMin, xMax, i in zip(yMins, yMaxs, xMins, xMaxs, range(0, xMins.size)):
            if ( yMax-yMin > 2 and xMax-xMin > 2 ):

                windDy, windDx = np.gradient( inpArray[ yMin:yMax, xMin:xMax ] )
                
                windDy = windDy.flatten()
                windDx = windDx.flatten()
                
                gain = np.sqrt(np.square(windDx) + np.square(windDy))
                
                # extract angles within the specified window after normalising about the primary direction
                angles = np.mod(np.arctan2( windDy, windDx ) + 2*np.pi, 2*np.pi)
                
                # extract occurances of angles after converting them into the specified resolution
                #types, freqs = np.unique( (angles*oRes/(2*np.pi)).astype(int), return_index=True )
                
                nAngles = (angles*oRes/(2*np.pi)).astype(int)
                
                # insert the frequency of occurances into the output array adjusted by gain
                angleDist = np.zeros( oRes )
                np.add.at( angleDist, nAngles, gain ) 
                
                netX = np.sum( vectorX*(angleDist**2) )
                netY = np.sum( vectorY*(angleDist**2) )
                # Finds the square weighted average vectors angle
                avrgAngle = np.arctan2( netY, netX )
                #avrgAngle = np.sum(( angleDist==np.max(angleDist) ) * np.arctan2( vectorY, vectorX ))
                
                # Shifts the angle distribution array such that the average angle lies at index zero
                angleDist = np.roll( angleDist, - int( avrgAngle*(oRes*0.5/np.pi) - 0.5) )
                
                outputs.append( angleDist )
        
        return np.array(outputs)
 
    @staticmethod
    def extractThicknesses( inpArray:np.ndarray, pointXs:np.ndarray, pointYs:np.ndarray, radius:int, oRes=12 ):
        """ Produces a histogram with the centre shifted to be inline with the exponentially weighted "centre of mass" """
   
        outputs = []
        positions = []
        mainAngle = []
        #guassian = gaussian_kernel2( radius*2 + 1 )

        x_coords, y_coords = np.meshgrid(np.arange(radius*2+1), np.arange(radius*2+1))
        x_coords = x_coords - radius
        y_coords = y_coords - radius

        smoothKern = np.array((0.054,0.242,0.398,0.242,0.054)) 
        smoothKern = np.array((0.154,0.242,0.398,0.242,0.154)) 
        
        intrestMask = (x_coords**2 + y_coords**2) < ((radius+0.5)**2)
        intrestMask[radius,radius] = 0
 
        angleMap = np.arctan2( y_coords, x_coords )+np.pi
        angleMap = (angleMap*oRes/(2*np.pi) + 0.5 ).astype(int)%oRes

        scaleFactors = np.zeros( (oRes) )
        np.add.at( scaleFactors, angleMap, intrestMask )
        
        angleSet = np.arange( 0, np.pi*2, np.pi*2/oRes )
        xVecFlat = np.cos( angleSet )
        yVecFlat = np.sin( angleSet )
        

        # Iterates through each search window
        for i in range(0, pointXs.size):
            centX = pointXs[i]
            centY = pointYs[i]

            xMin, xMax = centX-radius, centX+radius+1
            yMin, yMax = centY-radius, centY+radius+1

            if ( xMin > 0 and yMin > 0 and xMax < inpArray.shape[1] and yMax < inpArray.shape[0] ): 
                windowImage = ( inpArray[ yMin:yMax, xMin:xMax ] )  
                
                windowImage = np.where( np.abs( windowImage )<0.1, 0, np.where( windowImage > 0, 1, -1 ) )
                
                # Discriminator to ensure images have a high level of certainty assosiated with them
                if ( np.average( np.abs(windowImage) ) > 0.8 ):

                    extThickness = np.zeros( (oRes) )
                    np.add.at( extThickness, angleMap, windowImage )
                    extThickness = extThickness/scaleFactors
                    extThickness = convolveWithEdgeWrap( extThickness, smoothKern )
    
                    avrgAngle = np.arctan2( np.sum(yVecFlat*(extThickness**3)), np.sum(xVecFlat*(extThickness**3)) )
                    avrgAngle = np.pi*2+avrgAngle if avrgAngle<0 else avrgAngle
                    avrgIndex = int(0.5+oRes*avrgAngle/(np.pi*2))
                    mainAngle.append( avrgAngle )    

                    alignedThickness = np.roll( extThickness, -avrgIndex )
                    
                    outputs.append( alignedThickness )
                    positions.append( ( pointXs[i], pointYs[i] ) )

                
                #     plt.figure( 415 )
                #     plt.clf()
                #     plt.imshow( windowImage + np.where( intrestMask,0, np.inf), origin="lower" )  
                #     plt.figure( 4135 )
                #     plt.clf()
                #     plt.plot( alignedThickness ) 
                #     plt.show( block=False )
                # else:
                #     plt.figure( 415 )
                #     plt.imshow( windowImage, origin="lower" )  
                #     plt.figure( 4135 )
                #     plt.clf() 
                #     plt.show( block=False ) 
                # ""
        
        return np.array(outputs), np.array(positions), np.array( mainAngle )
    
    
    @staticmethod
    def extractGradients( inpArray:np.ndarray, pointXs:np.ndarray, pointYs:np.ndarray, radius:int, oRes=12 ):
        """ Produces a histogram with the centre shifted to be inline with the exponentially weighted "centre of mass" """
   
        outputs = []
        positions = []
        mainAngle = []
        #guassian = gaussian_kernel2( radius*2 + 1 )

        x_coords, y_coords = np.meshgrid(np.arange(radius*2+1), np.arange(radius*2+1))
        x_coords = x_coords - radius
        y_coords = y_coords - radius
 
        smoothKern = np.array((0.154,0.242,0.398,0.242,0.154))  
        #smoothKern = np.array((0,1,0)) 
        gaussian_array = np.array([
            [0.002969, 0.013306, 0.021938, 0.013306, 0.002969],
            [0.013306, 0.059634, 0.098320, 0.059634, 0.013306],
            [0.021938, 0.098320, 0.162103, 0.098320, 0.021938],
            [0.013306, 0.059634, 0.098320, 0.059634, 0.013306],
            [0.002969, 0.013306, 0.021938, 0.013306, 0.002969]
        ])
        gaussian_array = np.array([ 
            [0.059634, 0.098320, 0.059634],
            [0.098320, 0.162103, 0.098320],
            [0.059634, 0.098320, 0.059634] 
        ])
        
        intrestMask = (x_coords**2 + y_coords**2) < ((radius+0.5)**2)
        intrestMask[radius,radius] = 0  
        
        angleSet = np.arange( 0, np.pi*2, np.pi*2/oRes )
        xVecFlat = np.cos( angleSet )
        yVecFlat = np.sin( angleSet )
         
        absImage = np.where( inpArray < 0, -1, 1 ) 
        gDy, gDx = np.gradient(convolve2d(  absImage , gaussian_array, mode="same") )
        #gDy, gDx = np.gradient( absImage )
        #gDy, gDx = convolve2d( gDy, gaussian_array, mode="same" ), convolve2d( gDx, gaussian_array, mode="same" ) 
         
        # Iterates through each search window
        for i in range(0, pointXs.size):
            centX = pointXs[i]
            centY = pointYs[i]

            xMin, xMax = centX-radius, centX+radius+1
            yMin, yMax = centY-radius, centY+radius+1

            if ( xMin > 0 and yMin > 0 and xMax < inpArray.shape[1] and yMax < inpArray.shape[0] ): 
                windowAbsImage = ( absImage[ yMin:yMax, xMin:xMax ] )   
                windowImage = ( inpArray[ yMin:yMax, xMin:xMax ] )   
                
                # Discriminator to ensure images have a high level of certainty assosiated with them
                if ( np.average( np.abs(windowImage) ) > 0.65 ):
                    dx, dy = ( gDx[ yMin:yMax, xMin:xMax ]*intrestMask ), ( gDy[ yMin:yMax, xMin:xMax ]*intrestMask )
                     
                    magnitudes = np.sqrt(dy**2 + dx**2)  
                    angles = np.mod(np.arctan2( dy, dx ) + 2*np.pi, 2*np.pi) 
                    nAngles = (angles*oRes/(2*np.pi)).astype(int)
                    
                    gradHist = np.zeros( (oRes) )
                    np.add.at( gradHist, nAngles, magnitudes )  
                    gradHist = convolveWithEdgeWrap( gradHist, smoothKern )
    
                    avrgAngle = np.arctan2( np.sum( dy**3 ), np.sum( dx**3 ) ) + np.pi
                    avrgAngle = np.pi*2+avrgAngle if avrgAngle<0 else avrgAngle
                    avrgIndex = int(0.5+oRes*avrgAngle/(np.pi*2))
                    mainAngle.append( avrgAngle )    

                    alignedGradHist = np.roll( gradHist, -avrgIndex )
                    
                    outputs.append( alignedGradHist )
                    positions.append( ( pointXs[i], pointYs[i] ) )

                
                #     plt.figure( 415 )
                #     plt.clf()
                #     plt.imshow( windowAbsImage + np.where( intrestMask,0, np.inf), origin="lower" )  
                #     plt.figure( 4135 )
                #     plt.clf()
                #     plt.plot( alignedGradHist ) 
                #     plt.show( block=False )
                # else:
                #     plt.figure( 415 )
                #     plt.imshow( windowAbsImage, origin="lower" )  
                #     plt.figure( 4135 )
                #     plt.clf() 
                #     plt.show( block=False ) 
                # ""
        
        return np.array(outputs), np.array(positions), np.array( mainAngle )
    
    @staticmethod
    def structuralCombine(  descriptors:np.ndarray, positions:np.ndarray, angles:np.ndarray, minSeperation, minAngleDifference ):
        """ 
            Positions in format   [ [x,y], [x,y] ... ]
            Descriptors in format [ [...], [...] ... ] (floats)
            Angles in format      [ a, a ... ]
        
            This function iterates through each point, then it finds other points which are above minSeperation and have an angle difference of greater than minAngleDifference
            from this set it then selects 4 at random (if enough exist). Now it generates a new set of points which hold the following data from it's two parents:
                - Exist at the midpoint
                - Angle is from parent 1
                - Descriptor is the result of summing parent 1 and 2's descriptor then multiplying by the positional seperation
            
            After doing this the new points: position, descriptors and angles are returned
        """
        
        new_positions = []
        new_descriptors = []
        new_angles = []

        for i, pos in enumerate(positions):
            # Find candidate points for combination
            candidates = []
            for j, other_pos in enumerate(positions):
                if i != j:
                    separation = np.linalg.norm(pos - other_pos)
                    angle_difference = np.abs(angles[i] - angles[j])
                    angle_difference = min( 2*np.pi-angle_difference, angle_difference )
                    if separation > minSeperation and angle_difference > minAngleDifference:
                        candidates.append(j)

            # Select 4 random candidates if enough exist
            if len(candidates) >= 30:
                selected_candidates = random.sample(candidates, 30)
            else:
                selected_candidates = candidates

            # Generate new points
            for candidate_index in selected_candidates:
                mid_point = (pos + positions[candidate_index]) / 2
                new_positions.append(mid_point)
                new_descriptors.append((descriptors[i] - descriptors[candidate_index]) * np.linalg.norm(pos - positions[candidate_index]))
                new_angles.append(angles[i])

        return  np.array(new_descriptors), np.array(new_positions), np.array(new_angles)
        
        
        
    
    
        
    














