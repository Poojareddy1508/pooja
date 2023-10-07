import cv2 as c

# Loading the image
img = c.imread(r'C:\Users\anush\OneDrive\Pictures\dingo\Images\Pembroke im\n02113023_1659.jpg', c.IMREAD_GRAYSCALE)

# Creating an ORB object with specified parameters
orb = c.ORB_create(
    edgeThreshold=30,   
    patchSize=30,        
    nlevels=8,           
    fastThreshold=20,    
    scaleFactor=1.2,     
    WTA_K=2,             
    scoreType=c.ORB_HARRIS_SCORE, 
    firstLevel=0,       
    nfeatures=60         
)

# Detecting keypoints using ORB
keypoints = orb.detect(img, None)

# We can limit the number of keypoints by setting the desired range
min_keypoints = 25
max_keypoints = 75
if len(keypoints) > max_keypoints:
    keypoints = keypoints[:max_keypoints]
elif len(keypoints) < min_keypoints:
    keypoints = orb.detect(img, None)  

# Drawing only keypoints' locations on the image
image_with_keypoints = c.drawKeypoints(img, keypoints, None, color=(0, 255, 0), flags=0)

# Printing the number of keypoints extracted and parameter values
print(f"Number of Keypoints Extracted: {len(keypoints)}")
print(f"Edge Threshold: {orb.getEdgeThreshold()}")
print(f"Patch Size: {orb.getPatchSize()}")

# Displaying the image with keypoints
c.imshow("Image with Keypoints", image_with_keypoints)
c.waitKey(0)
c.destroyAllWindows()
