import cv2 as c
import matplotlib.pyplot as mplot

# Functioninf to plot RGB histogram
def plot_rgb_histogram(img, class_name):
    colors = ('r', 'g', 'b')
    mplot.figure(figsize=(10, 6))
    for p, color in enumerate(colors):
        histogram = c.calcHist([img], [p], None, [256], [0, 256])
        mplot.plot(histogram, color=color, label=f'{color.upper()} Channel')

    mplot.title(f'RGB Histogram for {class_name}', color="black", size='25', backgroundcolor="orange", pad=20)
    mplot.xlabel('Intensity', color="orange", size='15')
    mplot.ylabel('Pixel Count',color="orange", size='15')
    mplot.legend(loc='upper center', edgecolor ='black', facecolor="yellow")
    mplot.show()

# List of image paths (one image from each class)
img_paths = [
    r'C:\Users\anush\OneDrive\Pictures\dingo\Images\Staffordshire_bullterrier im\n02093256_1854.jpg',
    r'C:\Users\anush\OneDrive\Pictures\dingo\Images\Pembroke im\n02113023_1571.jpg',
    r'C:\Users\anush\OneDrive\Pictures\dingo\Images\newfoundland im\n02111277_2411.jpg',
    r'C:\Users\anush\OneDrive\Pictures\dingo\Images\dingo im\n02115641_2871.jpg'
]

# List of class names (replace with actual class names)
class_names = [
    'Staffordshire_bullterrier im',
    'newfoundland im',
    'Pembroke im',
    'dingo im'
]

# Load and plot RGB histograms for each image
for p, img_path in enumerate(img_paths):
    class_name = class_names[p]
    img = c.imread(img_path)
    plot_rgb_histogram(img, class_name)

