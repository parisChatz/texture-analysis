import matplotlib.pyplot as plt
import cv2
from skimage.feature import greycomatrix, greycoprops
import numpy as np
import pandas as pd

PATCH_SIZE = 21
con = []
dis = []
hom = []
asm = []
eng = []
cor = []

metrics = {"con": [], "dis": [], "hom": [], "asm": [], "eng": [], "cor": []}
cols = ['con', 'dis', 'hom', 'asm', 'eng', 'cor']
metrics = pd.DataFrame(data=metrics)

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


distances = [1,2,3]
angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
# distances = [1]
# angles = [0]

depth = 8
for patch in (mountain_patches + smooth_mnt_top_patches + linear_slope_patches + rocky_plains_patches):
    division = int(256/(2**depth))
    patch = np.divide(patch, division)
    patch = np.round(patch, decimals=0)
    patch = patch.astype(np.uint8)

    glcm = greycomatrix(patch, distances=distances, angles=angles, levels=255, symmetric=True, normed=True)
    con = np.average(greycoprops(glcm, 'contrast'))
    dis = np.average(greycoprops(glcm, 'dissimilarity'))
    hom = np.average(greycoprops(glcm, 'homogeneity'))
    asm = np.average(greycoprops(glcm, 'ASM'))
    eng = np.average(greycoprops(glcm, 'energy'))
    cor = np.average(greycoprops(glcm, 'correlation'))
    temp_metrics = pd.Series([con, dis, hom, asm, eng, cor], index=cols)
    metrics = metrics.append(temp_metrics, ignore_index=True)


print(metrics.head(20))
metrics.to_csv("/home/paris/PycharmProjects/Texture-Analysis/docu/metrics_8_bit.csv")