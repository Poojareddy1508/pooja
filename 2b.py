import cv2 as c
import matplotlib.pyplot as mplot

def equalize_histogram(img):
    equalized_img = c.equalizeHist(img)
    return equalized_img

img_paths = [
    r'C:\Users\anush\OneDrive\Pictures\dingo\Images\dingo im\n02115641_630.jpg', r'C:\Users\anush\OneDrive\Pictures\dingo\Images\dingo im\n02115641_1560.jpg',
    r'C:\Users\anush\OneDrive\Pictures\dingo\Images\newfoundland im\n02111277_392.jpg', r'C:\Users\anush\OneDrive\Pictures\dingo\Images\newfoundland im\n02111277_1023.jpg',
    r'C:\Users\anush\OneDrive\Pictures\dingo\Images\Pembroke im\n02113023_13297.jpg', r'C:\Users\anush\OneDrive\Pictures\dingo\Images\Pembroke im\n02113023_10636.jpg',
    r'C:\Users\anush\OneDrive\Pictures\dingo\Images\Staffordshire_bullterrier im\n02093256_911.jpg', r'C:\Users\anush\OneDrive\Pictures\dingo\Images\Staffordshire_bullterrier im\n02093256_1602.jpg'
]

# Lists to store grayscale images and histograms
gscale_imgs = []
equalized_imgs = []
gscale_histograms = []
equalized_histograms = []

# Loading, converting to grayscale, and compute=ing histograms
for path in img_paths:
    color_img = c.imread(path)
    gscale_img = c.cvtColor(color_img, c.COLOR_BGR2GRAY)
    gscale_imgs.append(gscale_img)
    equalized_img = equalize_histogram(gscale_img)
    equalized_imgs.append(equalized_img)
    
    gcale_histogram = c.calcHist([gscale_img], [0], None, [256], [0, 256])
    gscale_histograms.append(gscale_histograms)
    equalized_histogram = c.calcHist([equalized_img], [0], None, [256], [0, 256])
    equalized_histograms.append(equalized_histogram)

# Plot grayscale images and histograms
mplot.figure(figsize=(12, 12))
for p in range(len(gscale_imgs)):
    mplot.subplot(4, 4, p + 1)
    mplot.imshow(gscale_imgs[p], cmap='gray')
    mplot.title(f'Grayscale Image {p + 1}',pad=20, backgroundcolor='grey', color='white', size='10')
   
    mplot.subplot(4, 4, p + 9)
    mplot.hist(gscale_imgs[p].ravel(), bins=256, range=(0, 256), density=True, color='gray', alpha=0.7)
    mplot.title(f'Grayscale Hist. {p + 1}', pad=20, backgroundcolor='grey', color='white', size='10')

mplot.subplots_adjust(hspace=1, wspace=1)

# Plot equalized images and histograms
mplot.figure(figsize=(14, 14))
for p in range(len(equalized_imgs)):
    mplot.subplot(4, 4, p + 1)
    mplot.imshow(equalized_imgs[p], cmap='gray')
    mplot.title(f'Equalized Image {p + 1}',pad="20",backgroundcolor='grey', color='white', size='10')
   
    mplot.subplot(4, 4, p + 9)
    mplot.hist(equalized_imgs[p].ravel(), bins=256, range=(0, 256), density=True, color='gray', alpha=0.7)
    mplot.title(f'Equalized Hist. {p + 1}',pad="20",backgroundcolor='grey', color='white', size='10')

mplot.subplots_adjust(hspace=1, wspace=1)

# Pick one grayscale image and its corresponding equalized image for observation
selected_index = 0 
selected_gscale_img = gscale_imgs[selected_index]
selected_equalized_img = equalized_imgs[selected_index]

mplot.figure(figsize=(14, 7))
mplot.subplot(1, 2, 1)
mplot.imshow(selected_gscale_img, cmap='gray')
mplot.title('Original Grayscale Image',pad="20", backgroundcolor='grey',color='white', size='20')

mplot.subplot(1, 2, 2)
mplot.imshow(selected_equalized_img, cmap='gray')
mplot.title('Equalized Grayscale Image',pad="20", backgroundcolor='grey', color='white', size='20')

mplot.show()

