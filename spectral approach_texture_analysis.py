import matplotlib.pyplot as plt
import cv2
import numpy as np
from plots import barchart_all_orient, barchart_particular_orient


def collapsing_freq_in_all_orient(magnitude):
    """
    Script for summing the values for different radius from the center of the fft
    :param magnitude: Array of magnitude of shifted fft
    :return: An array of size 4 with the sums of values of the fft for 10 radius
    """
    total_sum = []
    radius = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    for r in radius:
        sum = 0
        for i, value in enumerate(magnitude):
            for j, value2 in enumerate(value):
                d = np.sqrt(np.power(i - 100, 2) + np.power(j - 100, 2))
                # print(i, j, value2, d)
                if d <= r:
                    sum += value2
        total_sum.append(sum)
    return total_sum


def collapsing_freq_in_particular_orient(magnitude):
    """
    Script for summing the values for different directons of the fft
    :param magnitude: Array of magnitude of shifted fft
    :return: an array of size 4 with the sums of values of the fft in 4 different directions
    """
    diag = magnitude.diagonal()
    inv_diag = np.fliplr(magnitude).diagonal()
    centerx, centery = np.divmod(magnitude.shape[0], 2)[0], np.divmod(magnitude.shape[1], 2)[0]

    zero_angle_sum = np.sum(magnitude[centerx, :])
    ninty_angle_sum = np.sum(magnitude[:, centery])
    diag_angle_sum = np.sum(diag)
    inv_diag_angle_sum = np.sum(inv_diag)
    return [zero_angle_sum, inv_diag_angle_sum, ninty_angle_sum, diag_angle_sum]


PATCH_SIZE = 151
# open the camera image
path = '/home/paris/PycharmProjects/Texture-Analysis/docu/ImgPIA.jpg'
# Reading the image
image = plt.imread(path)
cropped_image = image[29:868, 86:1569]
image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

mountain_locations = [(561, 900)]
rocky_plains_locations = [(431, 431)]

mountain_patches = []
for loc in mountain_locations:
    mountain_patches.append(image[loc[0]:loc[0] + PATCH_SIZE,
                            loc[1]:loc[1] + PATCH_SIZE])
rocky_plains_patches = []
for loc in rocky_plains_locations:
    rocky_plains_patches.append(image[loc[0]:loc[0] + PATCH_SIZE, loc[1]:loc[1] + PATCH_SIZE])

for i, patch in enumerate(mountain_patches):
    mount = patch
for i, patch in enumerate(rocky_plains_patches):
    rock = patch

img_rock = rock
f_rock = np.fft.fft2(img_rock)
fshift_rock = np.fft.fftshift(f_rock)
magnitude_spectrum_rock = 20 * np.log(np.abs(fshift_rock))

img_mount = mount
f_mount = np.fft.fft2(img_mount)
fshift_mount = np.fft.fftshift(f_mount)
magnitude_spectrum_mount = 20 * np.log(np.abs(fshift_mount))

plt.subplot(221), plt.imshow(img_rock, cmap='gray')
plt.title('Rock Image'), plt.xticks([]), plt.yticks([])
plt.subplot(222), plt.imshow(magnitude_spectrum_rock, cmap='gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.subplot(223), plt.imshow(img_mount, cmap='gray')
plt.title('Mount Image'), plt.xticks([]), plt.yticks([])
plt.subplot(224), plt.imshow(magnitude_spectrum_mount, cmap='gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()

# Calculate sums
mount_total_sum = collapsing_freq_in_all_orient(magnitude_spectrum_mount)
rock_total_sum = collapsing_freq_in_all_orient(magnitude_spectrum_rock)
mount_angles = collapsing_freq_in_particular_orient(magnitude_spectrum_mount)
rock_angles = collapsing_freq_in_particular_orient(magnitude_spectrum_rock)

# Plot barcharts
barchart_all_orient(mount_total_sum, rock_total_sum)
barchart_particular_orient(mount_angles, rock_angles)
