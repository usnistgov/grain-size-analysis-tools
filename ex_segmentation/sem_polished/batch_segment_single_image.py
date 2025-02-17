# Import external dependencies
import sys, os
import numpy as np
import cv2 as cv
import skimage.feature as feature
from skimage import morphology as morph
from skimage import filters as sfilt
from skimage.util import img_as_ubyte, img_as_float, img_as_bool
from skimage.util import invert as ski_invert
from matplotlib import pyplot as plt
from matplotlib.widgets import TextBox, Slider, Button, CheckButtons

# Import local modules
# Ensure this is the correct path to the functions folder
sys.path.insert(1, '../../imppy3d_functions') 
import import_export as imex
import plt_wrappers as pwrap
import ski_driver_functions as sdrv
import cv_processing_wrappers as cwrap

# Set constants related to plotting (for MatPlotLib)
SMALL_SIZE = 10
MEDIUM_SIZE = 12
BIGGER_SIZE = 14

plt.rc('font', size=MEDIUM_SIZE)         # Controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)    # Fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # Fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)   # Fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)   # Fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)   # Legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # Fontsize of the figure title


# -------- USER INPUTS --------

# Provide the filepath to the image that should be imported and segmented.
# Imported images will be converted to grayscale as UINT8 (i.e., max
# grayscale intensities of 255).
file_in_path = "./sem_alpha_beta_ti_6al_4v.tif"

# The resultant segmentation will be saved with the same filename as the 
# input image, but with "_segmented" appended to the name. For example, 
# "my_img.tif" would be segmented and saved as "my_img_segmented.tif".

# Segmentation should result in the grain boundaries being WHITE. If the
# resultant segmentation illustrates black grain boundaries, then the image
# grayscale values should be inverted after it is imported.
invert_grayscales = False

# ---- IMPORTANT ----
# All remaining inputs are provided by each segmentation step below.
# Scroll down and edit these as needed.



# -------- IMPORT IMAGE FILE --------

img1, img1_prop = imex.load_image(file_in_path)

if img1 is None:
    print(f"\nFailed to import images from the directory: \n{file_in_path}")
    print("\nQuitting the script...")

    quit()

# Optionally, extract the (Numpy) properties of image.
img1 = img_as_ubyte(img1)
img1_size = img1_prop[0]  # Total number of pixels
img1_shape = img1_prop[1] # Tuple containing the number of rows and columns
img1_dtype = img1_prop[2] # Returns the image data type (i.e., uint8)

if invert_grayscales:
    img1 = img_as_ubyte(ski_invert(img1))

img2 = img1.copy()


# -------- NON-LOCAL MEANS DENOISING FILTER --------
# ===== START INPUTS ===== 
h_filt = 0.045       # float
patch_size = 5      # int
search_dist = 7     # int
# ===== END INPUTS ===== 

print(f"\nApplying non-local means denoising...")
filt_params = ["nl_means", h_filt, int(patch_size), int(search_dist)]
img2 = sdrv.apply_driver_denoise(img2, filt_params)


# -------- SHARPEN FILTER --------
# ===== START INPUTS ===== 
sharp_radius = 2    # int
sharp_amount = 0.2  # float
# ===== END INPUTS ===== 

print(f"\nApplying sharpening filter...")
filt_params = ["unsharp_mask", int(sharp_radius), sharp_amount] 
img2 = sdrv.apply_driver_sharpen(img2, filt_params)


# -------- ADAPTIVE THRESHOLDING --------
# ===== START INPUTS ===== 
block_sz = 23        # int (odd)
thresh_offset = -5.0   # float (positive or negative)
# ===== END INPUTS ===== 

block_sz = int(block_sz)
if block_sz % 2 == 0:
    block_sz = block_sz + 1

print(f"\nApplying adaptive thresholding...")
img_thresh = sfilt.threshold_local(img2, block_size=block_sz, 
       method='gaussian', offset=thresh_offset)

img2 = img_as_ubyte(img2 > img_thresh)

print(f"\nSuccessfully applied adaptive thresholding:")
print(f"    Block size: {block_sz}")
print(f"    Intensity offset: {thresh_offset}")


# -------- (ALTERNATIVE) HYSTERESIS THRESHOLDING --------
# ===== START INPUTS ===== 
low_val = 64    # int
high_val = 200   # int
# ===== END INPUTS ===== 

# UNCOMMENT BELOW TO USE HYSTERESIS THRESHOLDING
#print(f"\nApplying hysteresis thresholding...")
#filt_params = ["hysteresis_threshold", low_val, high_val]
#img2 = sdrv.apply_driver_thresholding(img2, filt_params)


# -------- MORPHOLOGICAL OPERATIONS --------
# ===== START INPUTS ===== 
# 0: binary_closing
# 1: binary_opening
# 2: binary_dilation
# 3: binary_erosion
op_type = 0     # int

# 0: square 
# 1: disk 
# 2: diamond 
foot_type = 1   # int

# Kernel radius (pixels)
morph_rad = 2   # int
# ===== END INPUTS ===== 

print(f"\nApplying morphological binary operation...")
filt_params = [int(op_type), int(foot_type), int(morph_rad)]
img2 = sdrv.apply_driver_morph(img2, filt_params)


# -------- (CAN DO THIS AGAIN IF NEEDED) MORPHOLOGICAL OPERATIONS --------
# ===== START INPUTS ===== 
# 0: binary_closing
# 1: binary_opening
# 2: binary_dilation
# 3: binary_erosion
op_type = 3     # int

# 0: square 
# 1: disk 
# 2: diamond 
foot_type = 1   # int

# Kernel radius (pixels)
morph_rad = 1   # int
# ===== END INPUTS ===== 

# UNCOMMENT BELOW TO USE AN ADDITIONAL MORPHOLOGICAL OPERATION
#filt_params = [int(op_type), int(foot_type), int(morph_rad)]
#print(f"\nApplying morphological binary operation...")
#img2 = sdrv.apply_driver_morph(img2, filt_params)


# -------- REMOVE PIXEL ISLANDS AND SMALL HOLES --------
# ===== START INPUTS ===== 
max_hole_sz = 25  # int 
min_feat_sz = 9   # int
# ===== END INPUTS ===== 

print(f"\nRemoving small holes and pixel islands...")
img_bool = img_as_bool(img2)
img_temp = morph.remove_small_holes(img_bool, int(max_hole_sz), connectivity=1)
img2 = morph.remove_small_objects(img_temp, int(min_feat_sz), connectivity=1)
img2 = img_as_ubyte(img2)

print(f"\nSuccessfully filled in holes:")
print(f"    Maximum hole (area) size: {max_hole_sz} pixels")

print(f"\nSuccessfully deleted features:")
print(f"    Minimum feature (area) size: {min_feat_sz} pixels")


# -------- CAN ALSO CONSIDER SKELETONIZE --------

print(f"\nComputing the skeleton of the binary image...")
img_bool = img_as_bool(img2)
img2 = morph.skeletonize(img_bool)
img2 = img_as_ubyte(img2)
print(f"  Successfully computed the skeleton of the binary image.")


# -------- CALCULATE RELATIVE DENSITY --------

rel_density = cwrap.calc_rel_density(img2)


# -------- BEFORE AND AFTER PICTURE --------

fig1, ax1 = pwrap.create_2_bw_figs(img1, img2)
fig1.suptitle('Left: Original Image   |   Right: Segmented Image')

plt.tight_layout() 
plt.show()


# -------- SAVE IMAGE --------

filename_no_ext, file_extension = os.path.splitext(file_in_path)
file_out_name = filename_no_ext + "_segmented.tif"

save_flag = imex.save_image(img2, file_out_name)
if not save_flag:
    print(f"\nCould not save {file_out_name}")

plt.close('all') # Close all matplotlib figures
print("\nInteractive script successfully finished!")