import matplotlib.pyplot as plt
import cv2
from skimage.feature import greycomatrix, greycoprops
from skimage import data
import numpy as np

PATCH_SIZE = 21

# open the camera image
path = '/home/paris/PycharmProjects/Texture-Analysis/docu/ImgPIA.jpg'
# Reading the image
image = plt.imread(path)
cropped_image = image[29:868, 86:1569]

image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

mountain_locations = [(70, 66), (70, 175), (70, 345), (70, 531)]
mountain_patches = []
for loc in mountain_locations:
    mountain_patches.append(image[loc[0]:loc[0] + PATCH_SIZE, loc[1]:loc[1] + PATCH_SIZE])

smooth_mnt_top_locations = [(180, 755), (180, 877), (180, 1011), (180, 1103)]
smooth_mnt_top_patches = []
for loc in smooth_mnt_top_locations:
    smooth_mnt_top_patches.append(image[loc[0]:loc[0] + PATCH_SIZE, loc[1]:loc[1] + PATCH_SIZE])

linear_slope_locations = [(476, 885), (476, 995), (476, 1127), (476, 1233)]
linear_slope_patches = []
for loc in linear_slope_locations:
    linear_slope_patches.append(image[loc[0]:loc[0] + PATCH_SIZE, loc[1]:loc[1] + PATCH_SIZE])

rocky_plains_locations = [(585, 81), (585, 273), (703, 75), (703, 273)]
rocky_plains_patches = []
for loc in rocky_plains_locations:
    rocky_plains_patches.append(image[loc[0]:loc[0] + PATCH_SIZE, loc[1]:loc[1] + PATCH_SIZE])

# compute some GLCM properties each patch
xs = []
ys = []
for patch in (mountain_patches + smooth_mnt_top_patches + linear_slope_patches + rocky_plains_patches):
    glcm = greycomatrix(patch, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    xs.append(greycoprops(glcm, 'dissimilarity')[0, 0])
    ys.append(greycoprops(glcm, 'correlation')[0, 0])

# create the figure
fig = plt.figure(figsize=(8, 8))

# display original image with locations of patches
ax = fig.add_subplot(5, 2, 1)
ax.imshow(image, cmap='gray',
          vmin=0, vmax=255)
for (y, x) in mountain_locations:
    ax.plot(x + PATCH_SIZE / 2, y + PATCH_SIZE / 2, 'gs')
for (y, x) in smooth_mnt_top_locations:
    ax.plot(x + PATCH_SIZE / 2, y + PATCH_SIZE / 2, 'bs')
for (y, x) in linear_slope_locations:
    ax.plot(x + PATCH_SIZE / 2, y + PATCH_SIZE / 2, 'rs')
for (y, x) in rocky_plains_locations:
    ax.plot(x + PATCH_SIZE / 2, y + PATCH_SIZE / 2, 'ys')
ax.set_xlabel('Original Image')
ax.set_xticks([])
ax.set_yticks([])
ax.axis('image')

# for each patch, plot (dissimilarity, correlation)
ax = fig.add_subplot(5, 2, 2)
mnt_len = len(mountain_patches)
smooth_mnt_len = len(smooth_mnt_top_patches)
lin_mnt_len = len(linear_slope_patches)
ax.plot(xs[:mnt_len], ys[:mnt_len], 'go', label='mountain')
ax.plot(xs[smooth_mnt_len:mnt_len + smooth_mnt_len],
        ys[smooth_mnt_len:mnt_len + smooth_mnt_len], 'bo', label='smooth_mnt_top')
ax.plot(xs[mnt_len + smooth_mnt_len:mnt_len + smooth_mnt_len + lin_mnt_len],
        ys[mnt_len + smooth_mnt_len:mnt_len + smooth_mnt_len + lin_mnt_len], 'ro', label='linear_slope')
ax.plot(xs[mnt_len + smooth_mnt_len + lin_mnt_len:],
        ys[mnt_len + smooth_mnt_len + lin_mnt_len:], 'yo', label='rocky_plains')
ax.set_xlabel('GLCM Dissimilarity')
ax.set_ylabel('GLCM Correlation')
# ax.legend()

# display the image patches
for i, patch in enumerate(mountain_patches):
    ax = fig.add_subplot(5, len(mountain_patches), len(mountain_patches) * 1 + i + 1)
    ax.imshow(patch, cmap=plt.cm.gray,
              vmin=0, vmax=255)
    ax.set_xlabel('mountain %d' % (i + 1))

for i, patch in enumerate(smooth_mnt_top_patches):
    ax = fig.add_subplot(5, len(smooth_mnt_top_patches), len(smooth_mnt_top_patches) * 2 + i + 1)
    ax.imshow(patch, cmap=plt.cm.gray,
              vmin=0, vmax=255)
    ax.set_xlabel('smooth_mnt_top %d' % (i + 1))

for i, patch in enumerate(linear_slope_patches):
    ax = fig.add_subplot(5, len(linear_slope_patches), len(linear_slope_patches) * 3 + i + 1)
    ax.imshow(patch, cmap=plt.cm.gray,
              vmin=0, vmax=255)
    ax.set_xlabel('linear_slope %d' % (i + 1))

for i, patch in enumerate(rocky_plains_patches):
    ax = fig.add_subplot(5, len(rocky_plains_patches), len(rocky_plains_patches) * 4 + i + 1)
    ax.imshow(patch, cmap=plt.cm.gray,
              vmin=0, vmax=255)
    ax.set_xlabel('rocky_plains %d' % (i + 1))

# display the patches and plot
fig.suptitle('Grey level co-occurrence matrix features', fontsize=14, y=1.05)
plt.tight_layout()
plt.show()

