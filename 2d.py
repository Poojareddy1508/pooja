import cv2 as c
import numpy as np

# Function to compute histogram comparison metrics
def compare_histograms(hist1, hist2, metric):
    if metric == 'euclidean':
        return np.linalg.norm(hist1 - hist2)
    elif metric == 'manhattan':
        return np.sum(np.abs(hist1 - hist2))
    elif metric == 'bhattacharyya':
        return c.compareHist(hist1, hist2, c.HISTCMP_BHATTACHARYYA)
    elif metric == 'intersection':
        return c.compareHist(hist1, hist2, c.HISTCMP_INTERSECT)
    else:
        raise ValueError("Invalid metric")
print()

# Loading the images (two from the same class, one from a different class)
img1 = c.imread(r'C:\Users\anush\OneDrive\Pictures\dingo\Images\Staffordshire_bullterrier im\n02093256_911.jpg')
img2 = c.imread(r'C:\Users\anush\OneDrive\Pictures\dingo\Images\Staffordshire_bullterrier im\n02093256_2405.jpg')
img3 = c.imread(r'C:\Users\anush\OneDrive\Pictures\dingo\Images\newfoundland im\n02111277_2524.jpg')

# Converting the images to grayscale
gray_img1 = c.cvtColor(img1, c.COLOR_BGR2GRAY)
gray_img2 = c.cvtColor(img2, c.COLOR_BGR2GRAY)
gray_img3 = c.cvtColor(img3, c.COLOR_BGR2GRAY)

# Computing histograms for the grayscale images
hist1 = c.calcHist([gray_img1], [0], None, [256], [0, 256])
hist2 = c.calcHist([gray_img2], [0], None, [256], [0, 256])
hist3 = c.calcHist([gray_img3], [0], None, [256], [0, 256])

# Normalizing histograms
hist1 /= hist1.sum()
hist2 /= hist2.sum()
hist3 /= hist3.sum()

# Calculating histogram comparison metrics
metrics = ['euclidean', 'manhattan', 'bhattacharyya', 'intersection']

for metric in metrics:
    same_class_distance = compare_histograms(hist1, hist2, metric)
    diff_class_distance = compare_histograms(hist1, hist3, metric)
    
    print(f'{metric.capitalize()} Distance (Same Class): {same_class_distance:.2f}')
    print(f'{metric.capitalize()} Distance (Different Class): {diff_class_distance:.2f}')
print()
