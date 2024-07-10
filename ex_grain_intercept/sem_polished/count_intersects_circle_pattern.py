# Import external dependencies
import sys, os
import numpy as np
from skimage.util import img_as_ubyte
from skimage.util import invert as ski_invert
import skimage.draw as draw
from matplotlib import pyplot as plt

# Import local modules
# Ensure this is the correct path to the functions folder
sys.path.insert(1, '../../imppy3d_functions') 
import grain_size_functions as gsz
import cv_processing_wrappers as cwrap
import import_export as imex
import volume_image_processing as vol

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

# Provide the filepath to the image that should be imported. This image should
# be a binary image (containing only black and white pixels), and it should be
# saved as an unsigned 8-bit image. Moreover, the metallographic grain 
# boundaries are assumed to be denoted by white pixels. If the image is not
# square, it will automatically be cropped about the center.
file_in_path = "./sem_alpha_beta_ti_6al_4v_segmented.tif"

# An intersection pattern based on the ASTM E-112 standard will be drawn onto
# the provided, segmented image. This pattern includes lines and circles.
# Intersections between this pattern and the grain boundaries will be found, and
# the lengths of the grain segments between intersections will be recorded (in
# pixels). To provide feedback to the user, these grain segments between the
# intersections will be marked as gray pixels and saved as a new image for
# inspection. These distance values will be saved as an .csv file in the same
# directory as the imported image, and the filename will automatically be
# generated with the appended word, "_distances". For example, if the imported
# image was, "./img_segmented.tif", then the output file would be saved as,
# "./img_segmented_distances.csv".

# Provide a filepath where the marked image can be saved. The image will be
# saved as unsigned integer (8-bit) image. Like the input image, grain
# boundaries are denoted by white pixels. The intersect segments are
# superimposed as gray pixels with intensity 150 (out of 255). Note, the marked
# image will be padded with black pixels arounds its perimeter resulting in
# an image that is two pixels wider and taller than what was cropped (if 
# cropping was needed).
img_out_path = "./intersect_pics/sem_alpha_beta_ti_6al_4v_segmented_E112.tif"

# Set "borders_white" to True if the provided input image contains white pixels
# that correspond to the grain boundaries. If True, then nothing is done. If
# False, then it is assumed that the base color of the image is white with grain
# boundaries denoted by black pixels. In this case, when False, the image will
# be inverted such that grain boundaries are denoted by white pixels.
borders_white = True




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

img2 = img1.copy()

n_row = img2.shape[0]
n_col = img2.shape[1]


# -------- CROP TO SQUARE --------

# Image must be equal aspect ratios (i.e., a square) to avoid biases.
max_img_diameter = np.amin(img2.shape)
if n_row != n_col:
    print(f"\nCropping image to be square...")

    crop_roi = (max_img_diameter, max_img_diameter)
    img2 = cwrap.crop_img(img2, crop_roi)
    n_row2 = img2.shape[0]
    n_col2 = img2.shape[1]


# -------- CALCULATE THE ASTM E112 CIRCLE AND LINE COORDINATES --------

print(f"\nCalculating pixel coordinates of the lines and circles...")

if not borders_white:
    img2 = ski_invert(img2)

img2 = vol.pad_image_boundary(img2, cval_in=0, n_pad_in=1)
n_row2 = img2.shape[0]
n_col2 = img2.shape[1]

margin = int(np.round(0.05*max_img_diameter)) # pixels

# Calculate start and end coordinates of the four lines.
# Using index coordinates, [(row, column), (row, column)]
line_xx = np.array([[n_row2-1 - margin, 0], 
           [n_row2-1 - margin, n_col2-1]]) 

line_yy = np.array([[0, margin],
           [n_row2-1, margin]])

line_xy1 = np.array([[0, 0],
            [n_row2-1, n_col2-1]])

line_xy2 = np.array([[0, n_col2-1],
            [n_row2-1, 0]])

# Calculate center of the image as row and column indices, which will be
# used as the center of the concentric circles
r_center = int(np.floor(n_row2/2.0))
c_center = int(np.floor(n_col2/2.0))

# Calculate the circle radii
max_img_diameter = np.amin(img2.shape)
circ_rad1 = int(np.round(0.7958*max_img_diameter/2.0))
circ_rad2 = int(np.round(0.5305*max_img_diameter/2.0))
circ_rad3 = int(np.round(0.2653*max_img_diameter/2.0))

# Get the line index coordinates, r for rows and c for columns
line_xx_r, line_xx_c = draw.line(line_xx[0,0], line_xx[0,1], 
                                 line_xx[1,0], line_xx[1,1])
line_yy_r, line_yy_c = draw.line(line_yy[0,0], line_yy[0,1], 
                                 line_yy[1,0], line_yy[1,1])
line_xy1_r, line_xy1_c = draw.line(line_xy1[0,0], line_xy1[0,1], 
                                   line_xy1[1,0], line_xy1[1,1])
line_xy2_r, line_xy2_c = draw.line(line_xy2[0,0], line_xy2[0,1], 
                                   line_xy2[1,0], line_xy2[1,1])

# Get the circle index coordinates, r for rows and c for columns
circ_rad1_r, circ_rad1_c = draw.circle_perimeter(r_center, c_center, circ_rad1)
circ_rad2_r, circ_rad2_c = draw.circle_perimeter(r_center, c_center, circ_rad2)
circ_rad3_r, circ_rad3_c = draw.circle_perimeter(r_center, c_center, circ_rad3)

# Pixel coordinates are not guaranteed to be adjacent and continuous.
# Need to go through the coordinates and reorder them to be continuous.
line_xx_rc = np.transpose(np.vstack((line_xx_r, line_xx_c)))
line_xx_rc = gsz.make_continuous_line(line_xx_rc)
line_xx_r = line_xx_rc[:,0]
line_xx_c = line_xx_rc[:,1]

line_yy_rc = np.transpose(np.vstack((line_yy_r, line_yy_c)))
line_yy_rc = gsz.make_continuous_line(line_yy_rc)
line_yy_r = line_yy_rc[:,0]
line_yy_c = line_yy_rc[:,1]

line_xy1_rc = np.transpose(np.vstack((line_xy1_r, line_xy1_c)))
line_xy1_rc = gsz.make_continuous_line(line_xy1_rc)
line_xy1_r = line_xy1_rc[:,0]
line_xy1_c = line_xy1_rc[:,1]

line_xy2_rc = np.transpose(np.vstack((line_xy2_r, line_xy2_c)))
line_xy2_rc = gsz.make_continuous_line(line_xy2_rc)
line_xy2_r = line_xy2_rc[:,0]
line_xy2_c = line_xy2_rc[:,1]

circ_rad1_rc = np.transpose(np.vstack((circ_rad1_r, circ_rad1_c)))
circ_rad1_rc = gsz.make_continuous_circle(circ_rad1_rc)
circ_rad1_rc = gsz.find_new_start_on_circle(circ_rad1_rc, img2)
circ_rad1_r = circ_rad1_rc[:,0]
circ_rad1_c = circ_rad1_rc[:,1]

circ_rad2_rc = np.transpose(np.vstack((circ_rad2_r, circ_rad2_c)))
circ_rad2_rc = gsz.make_continuous_circle(circ_rad2_rc)
circ_rad2_rc = gsz.find_new_start_on_circle(circ_rad2_rc, img2)
circ_rad2_r = circ_rad2_rc[:,0]
circ_rad2_c = circ_rad2_rc[:,1]

circ_rad3_rc = np.transpose(np.vstack((circ_rad3_r, circ_rad3_c)))
circ_rad3_rc = gsz.make_continuous_circle(circ_rad3_rc)
circ_rad3_rc = gsz.find_new_start_on_circle(circ_rad3_rc, img2)
circ_rad3_r = circ_rad3_rc[:,0]
circ_rad3_c = circ_rad3_rc[:,1]


# -------- FIND GRAIN BOUNDARY INTERSECTIONS AND MEASURE LENGTHS --------

print(f"\nSearching for grain boundary intersects and measuring segment lengths...")

# Calculate the line and circumference intersection coordinates
line_xx_arr = img2[line_xx_r, line_xx_c]
line_xx_segs_ii = gsz.find_intersections(line_xx_arr)

line_yy_arr = img2[line_yy_r, line_yy_c]
line_yy_segs_ii = gsz.find_intersections(line_yy_arr)

line_xy1_arr = img2[line_xy1_r, line_xy1_c]
line_xy1_segs_ii = gsz.find_intersections(line_xy1_arr)

line_xy2_arr = img2[line_xy2_r, line_xy2_c]
line_xy2_segs_ii = gsz.find_intersections(line_xy2_arr)

circ_rad1_arr = img2[circ_rad1_r, circ_rad1_c]
circ_rad1_segs_ii = gsz.find_intersections(circ_rad1_arr)

circ_rad2_arr = img2[circ_rad2_r, circ_rad2_c]
circ_rad2_segs_ii = gsz.find_intersections(circ_rad2_arr)

circ_rad3_arr = img2[circ_rad3_r, circ_rad3_c]
circ_rad3_segs_ii = gsz.find_intersections(circ_rad3_arr)

# Calculate linear lengths and arc lengths for each grain segment
line_xx_dist_arr = gsz.measure_line_dist(line_xx_segs_ii, line_xx_rc)
line_yy_dist_arr = gsz.measure_line_dist(line_yy_segs_ii, line_yy_rc)
line_xy1_dist_arr = gsz.measure_line_dist(line_xy1_segs_ii, line_xy1_rc)
line_xy2_dist_arr = gsz.measure_line_dist(line_xy2_segs_ii, line_xy2_rc)
circ_rad1_dist_arr = gsz.measure_circular_dist(circ_rad1_segs_ii, circ_rad1_rc)
circ_rad2_dist_arr = gsz.measure_circular_dist(circ_rad2_segs_ii, circ_rad2_rc)
circ_rad3_dist_arr = gsz.measure_circular_dist(circ_rad3_segs_ii, circ_rad3_rc)

# Mark the grain segments with gray pixels (intensity of 150 out of 255) 
img2 = gsz.mark_segments_on_image(img2, line_xx_segs_ii, line_xx_rc)
img2 = gsz.mark_segments_on_image(img2, line_yy_segs_ii, line_yy_rc)
img2 = gsz.mark_segments_on_image(img2, line_xy1_segs_ii, line_xy1_rc)
img2 = gsz.mark_segments_on_image(img2, line_xy2_segs_ii, line_xy2_rc)
img2 = gsz.mark_segments_on_image(img2, circ_rad1_segs_ii, circ_rad1_rc)
img2 = gsz.mark_segments_on_image(img2, circ_rad2_segs_ii, circ_rad2_rc)
img2 = gsz.mark_segments_on_image(img2, circ_rad3_segs_ii, circ_rad3_rc)


# -------- SAVE MARKED IMAGE --------

save_flag = imex.save_image(img2, img_out_path)
if not save_flag:
    print(f"\nCould not save {img_out_path}")


# -------- SAVE DATA TO CSV FILE --------

filename_no_ext, file_extension = os.path.splitext(file_in_path)
csv_file_out_name = filename_no_ext + "_distances.csv"

header_str = ["Line ID (1: Horizontal line | 2: Vertical line | " + \
              "3: Top-left to bottom-right diagonal line | " + \
              "4: Top-right to bottom-left diagonal line | 5: Largest circle | " + \
              "6: Middle circle | 7: Smallest circle)", 
              "Segment Distance (Pixels)"]

# Need to reformat the data into a 2D array suitable for the .csv file
temp_id_arr = np.ones(line_xx_dist_arr.size)
line_xx_data_out = np.transpose(np.vstack((temp_id_arr, line_xx_dist_arr)))

temp_id_arr = np.ones(line_yy_dist_arr.size)*2
line_yy_data_out = np.transpose(np.vstack((temp_id_arr, line_yy_dist_arr)))

temp_id_arr = np.ones(line_xy1_dist_arr.size)*3
line_xy1_data_out = np.transpose(np.vstack((temp_id_arr, line_xy1_dist_arr)))

temp_id_arr = np.ones(line_xy2_dist_arr.size)*4
line_xy2_data_out = np.transpose(np.vstack((temp_id_arr, line_xy2_dist_arr)))

temp_id_arr = np.ones(circ_rad1_dist_arr.size)*5
circ_rad1_data_out = np.transpose(np.vstack((temp_id_arr, circ_rad1_dist_arr)))

temp_id_arr = np.ones(circ_rad2_dist_arr.size)*6
circ_rad2_data_out = np.transpose(np.vstack((temp_id_arr, circ_rad2_dist_arr)))

temp_id_arr = np.ones(circ_rad3_dist_arr.size)*7
circ_rad3_data_out = np.transpose(np.vstack((temp_id_arr, circ_rad3_dist_arr)))

dist_data_all_out = np.vstack((line_xx_data_out, 
                               line_yy_data_out, 
                               line_xy1_data_out,
                               line_xy2_data_out,
                               circ_rad1_data_out,
                               circ_rad2_data_out,
                               circ_rad3_data_out))

gsz.save_csv(dist_data_all_out, csv_file_out_name, header_str)


# -------- PLOT THE DATA --------

print(f"\nCreating a box-plot for each line and circle...")

# Reshape the data to make it suitable for plotting
pix_dist_2d_plt = [line_xx_data_out[:,1], 
                   line_yy_data_out[:,1], 
                   line_xy1_data_out[:,1],
                   line_xy2_data_out[:,1],
                   circ_rad1_data_out[:,1],
                   circ_rad2_data_out[:,1],
                   circ_rad3_data_out[:,1]]

theta_labels_1D = ["X-X", "Y-Y", "XY1",
                   "XY2", "C-Big", "C-Mid",
                   "C-Tiny"]

fig1, ax1 = plt.subplots()
ax1.boxplot(pix_dist_2d_plt, sym="", labels=theta_labels_1D)

num_boxes = len(pix_dist_2d_plt)
for m in range(num_boxes):
    cur_y_arr = pix_dist_2d_plt[m]
    cur_x_arr = np.random.normal(m+1, 0.05, size=cur_y_arr.size)
    ax1.plot(cur_x_arr, cur_y_arr, '.b', alpha=0.2)

#ax1.set_xlabel('Name of the Line')
ax1.set_ylabel('Segment Length of Grains [pixels]')


# -------- SUMMARIZE THE RESULTS IN THE COMMAND WINDOW --------

print("\n\n ========== RESULTS SUMMARY ========== ")

print(f"\n --- Horizontal Line --- ")
temp_val = (line_xx[0,0] - line_xx[1,0])**2 + (line_xx[0,1] - line_xx[1,1])**2
print(f"  Line Length (Pixels): {np.sqrt(temp_val):.2f}")
print(f"  Number of Grain Segments: {line_xx_dist_arr.size}")
print(f"  Average Grain Size (Pixels): {np.mean(line_xx_dist_arr):.2f}")
print(f"  Median Grain Size (Pixels): {np.median(line_xx_dist_arr):.2f}")
print(f"  Std. Deviation in Grain Size (Pixels): {np.std(line_xx_dist_arr):.2f}")

print(f"\n --- Vertical Line --- ")
temp_val = (line_yy[0,0] - line_yy[1,0])**2 + (line_yy[0,1] - line_yy[1,1])**2
print(f"  Line Length (Pixels): {np.sqrt(temp_val):.2f}")
print(f"  Number of Grain Segments: {line_yy_dist_arr.size}")
print(f"  Average Grain Size (Pixels): {np.mean(line_yy_dist_arr):.2f}")
print(f"  Median Grain Size (Pixels): {np.median(line_yy_dist_arr):.2f}")
print(f"  Std. Deviation in Grain Size (Pixels): {np.std(line_yy_dist_arr):.2f}")

print(f"\n --- Diagonal Line (Top-Left to Bottom-Right) --- ")
temp_val = (line_xy1[0,0] - line_xy1[1,0])**2 + (line_xy1[0,1] - line_xy1[1,1])**2
print(f"  Line Length (Pixels): {np.sqrt(temp_val):.2f}")
print(f"  Number of Grain Segments: {line_xy1_dist_arr.size}")
print(f"  Average Grain Size (Pixels): {np.mean(line_xy1_dist_arr):.2f}")
print(f"  Median Grain Size (Pixels): {np.median(line_xy1_dist_arr):.2f}")
print(f"  Std. Deviation in Grain Size (Pixels): {np.std(line_xy1_dist_arr):.2f}")

print(f"\n --- Diagonal Line (Top-Right to Bottom-Left) --- ")
temp_val = (line_xy2[0,0] - line_xy2[1,0])**2 + (line_xy2[0,1] - line_xy2[1,1])**2
print(f"  Line Length (Pixels): {np.sqrt(temp_val):.2f}")
print(f"  Number of Grain Segments: {line_xy2_dist_arr.size}")
print(f"  Average Grain Size (Pixels): {np.mean(line_xy2_dist_arr):.2f}")
print(f"  Median Grain Size (Pixels): {np.median(line_xy2_dist_arr):.2f}")
print(f"  Std. Deviation in Grain Size (Pixels): {np.std(line_xy2_dist_arr):.2f}")

print(f"\n --- Largest (Outer) Circle --- ")
print(f"  Circumference (Pixels): {2.0*np.pi*circ_rad1:.2f}")
print(f"  Number of Grain Segments: {circ_rad1_dist_arr.size}")
print(f"  Average Grain Size (Pixels): {np.mean(circ_rad1_dist_arr):.2f}")
print(f"  Median Grain Size (Pixels): {np.median(circ_rad1_dist_arr):.2f}")
print(f"  Std. Deviation in Grain Size (Pixels): {np.std(circ_rad1_dist_arr):.2f}")

print(f"\n --- Middle Circle --- ")
print(f"  Circumference (Pixels): {2.0*np.pi*circ_rad2:.2f}")
print(f"  Number of Grain Segments: {circ_rad2_dist_arr.size}")
print(f"  Average Grain Size (Pixels): {np.mean(circ_rad2_dist_arr):.2f}")
print(f"  Median Grain Size (Pixels): {np.median(circ_rad2_dist_arr):.2f}")
print(f"  Std. Deviation in Grain Size (Pixels): {np.std(circ_rad2_dist_arr):.2f}")

print(f"\n --- Smallest (Inner) Circle --- ")
print(f"  Circumference (Pixels): {2.0*np.pi*circ_rad3:.2f}")
print(f"  Number of Grain Segments: {circ_rad3_dist_arr.size}")
print(f"  Average Grain Size (Pixels): {np.mean(circ_rad3_dist_arr):.2f}")
print(f"  Median Grain Size (Pixels): {np.median(circ_rad3_dist_arr):.2f}")
print(f"  Std. Deviation in Grain Size (Pixels): {np.std(circ_rad3_dist_arr):.2f}")

print(f"\n --- All Lines and Circles Grain Size Statistics --- ")
print(f"  Total Number of Grain Segments: {dist_data_all_out.shape[0]}")
print(f"  Average Grain Size (Pixels): {np.mean(dist_data_all_out[:,1]):.2f}")
print(f"  Median Grain Size (Pixels): {np.median(dist_data_all_out[:,1]):.2f}")
print(f"  Std. Deviation in Grain Size (Pixels): {np.std(dist_data_all_out[:,1]):.2f}")

# Actually show the plot after the summary has been reported
plt.show()
print("\n\nScript successfully terminated!")
