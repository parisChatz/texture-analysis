import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from plots import specific_window_hists, window_hists

# open the camera image
path = '/home/paris/PycharmProjects/Texture-Analysis/docu/ImgPIA.jpg'
# Reading the image
image = plt.imread(path)
cropped_image = image[29:868, 86:1569]
(winW, winH) = (187, 187)

print_hist = False
if print_hist:
    fig = plt.figure(constrained_layout=True)
    spec = gridspec.GridSpec(ncols=2, nrows=2, figure=fig)
    ax1 = fig.add_subplot(spec[0, 0])
    plt.imshow(image)
    ax2 = fig.add_subplot(spec[0, 1])
    plt.hist(image.ravel(), 256, [0, 256])
    ax3 = fig.add_subplot(spec[1, 0])
    plt.imshow(cropped_image)
    ax4 = fig.add_subplot(spec[1, 1])
    plt.hist(cropped_image.ravel(), 256, [0, 256])
    plt.show()

# Plot distribution moments of windows
window_hists(cropped_image, winW, winH)

# Plot distribution moments of specific windows
window_number_list = [14, 45, 69, 71]
specific_window_hists(cropped_image, winW, winH, window_number_list)
