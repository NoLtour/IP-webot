import cv2
import numpy as np

# Load images
image1 = cv2.imread('C:\\IP webot\\IP_2D_Imp_2\\CVTesting\\Capture1.png', cv2.COLOR_BGR2GRAY)
image2 = cv2.imread('C:\\IP webot\\IP_2D_Imp_2\\CVTesting\\Capture2.png', cv2.COLOR_BGR2GRAY)


# 1. Extract Descriptors (Custom method)
def extract_custom_descriptors(image):
    # Your custom descriptor extraction method
    # This could be SIFT, SURF, ORB, or any custom method
    # Make sure to return descriptors
    # For demonstration, let's use ORB as an example
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(image, None)
    return keypoints, descriptors 

# Extract descriptors for both images
keypoints1, descriptors1 = extract_custom_descriptors(image1)
keypoints2, descriptors2 = extract_custom_descriptors(image2)

# 2. Match descriptors
# For example, using Brute-Force Matcher
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(descriptors1, descriptors2)

# Sort matches based on their distances
matches = sorted(matches, key=lambda x: x.distance)

# Extract matched keypoints
src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

# Find perspective transformation
M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

# Extract translation and rotation
dx = M[0, 2]
dy = M[1, 2]
theta = np.arctan2(M[1, 0], M[0, 0]) * 180 / np.pi

# Print offset (including rotation)
print("Offset (x, y):", dx, dy)
print("Rotation (degrees):", theta)

# 3. Draw matches
matched_image = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Display the matched image
cv2.imshow('Matches', matched_image)
cv2.waitKey(0)
cv2.destroyAllWindows()