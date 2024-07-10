import numpy as np
from scipy.spatial.distance import pdist, squareform
import csv


def find_intersections(pixel_arr_in):
    # INPUT
    #
    # pixel_arr_in: 1D Numpy array of type UINT8 containing only white
    # and black pixels (i.e., intensities 255 or 0). Grain boundaries 
    # are denoted by white pixels. This algorithm will traverse through
    # pixel_arr_in and count the black pixels between successive white
    # pixels.
    #
    #
    # OUTPUT
    #
    # intersect_pairs_ii: A 2D Numpy array containing the start and end
    # indices that mark segments of adjacent black pixels in 
    # pixel_arr_in. The shape of intersect_pairs_ii is N rows by 2 
    # columns, which corresponds to N grain segments that were found.
    # The first column corresponds to start indices, and the second (or
    # last) column corresponds to end indices.


    # Loop through the pixel_arr_in
    pixel_arr_len = pixel_arr_in.size
    intersect_pairs_ii = []

    cur_sect = np.array([-1, -1])
    cur_sect_count = 0
    for cc in np.arange(pixel_arr_len-1): 

        val_cc = pixel_arr_in[cc] # Current pixel grayscale value
        val_cpo = pixel_arr_in[cc+1] # Next pixel grayscale value

        # If on white and found an edge to the right, mark this position as the
        # start of the current grain.
        if val_cpo < val_cc:
            cur_sect[0] = cc + 1

        # If on black and found the left side of white edge, investigate this
        # further. Could be the end of the current grain.
        if val_cpo > val_cc:

            # If there is no marked start, then we never found the start of 
            # grain, so skip this edge. Else, keep it.
            if cur_sect[0] == -1:
                continue

            cur_sect[1] = cc + 1

            # Found a complete grain length segment. Store it for later.
            pix_dist = cur_sect[1] - cur_sect[0]

            # Single pixel distances are not very useful. Skip these segments.
            if pix_dist <= 2:
                cur_sect = np.array([-1, -1])
                continue

            intersect_pairs_ii.append(cur_sect)

            # Reset the start and end positions; get ready to find the next grain.
            cur_sect = np.array([-1, -1])
            cur_sect_count += 1
    
    return np.array(intersect_pairs_ii)


def make_continuous_line(arr_2d_in):
    # INPUT
    #
    # arr_2d_in: A numpy.array of shape (n, 2) of integer type. Each row entry
    # represents the pixel coordinates of a line as [row, column]. This function
    # will return a manipulated order of the 2D array that ensures the line
    # represented by these pixel coordinates is continuous.
    #
    #
    # OUTPUT
    #
    # arr_2d_out: A numpy.array of shape (n, 2) of the same type as arr_2d_in.
    # This array contains the same coordinates as arr_2d_in, but in the case
    # that the input array was not ordered in a continuous fashion, then the
    # returned arr_2d_out will be continuous in order.


    arr_2d = arr_2d_in.copy()

    # Find an end point of the line
    PIXEL_TOL = 0.1
    start_pnt_i = 0

    dist_mat = squareform(pdist(arr_2d, 'euclidean'))
    
    for m, cur_dist_row in enumerate(dist_mat):

        # Sort each row into ascending order
        cur_dist_row_sorted = np.sort(cur_dist_row, axis=-1, kind='quicksort')

        # First entry will be zero since it is the distance with respect
        # to itself. If the next two minimum distances are equal, then this
        # is not an end point.
        if not (np.absolute(cur_dist_row_sorted[1] - cur_dist_row_sorted[2]) < PIXEL_TOL):
            start_pnt_i = m
            break

    start_pnt_dist_row = dist_mat[start_pnt_i, :]
    dist_row_sorted_ii = np.argsort(start_pnt_dist_row, axis=-1, kind='quicksort')

    arr_2d_out = arr_2d[dist_row_sorted_ii]

    return arr_2d_out


def make_continuous_circle(arr_2d_in):
    # INPUT
    #
    # arr_2d_in: A numpy.array of shape (n, 2) of integer type. Each row entry
    # represents the pixel coordinates of a circle as [row, column]. This 
    # function will return a manipulated order of the 2D array that ensures the 
    # circle represented by these pixel coordinates is continuous.
    #
    #
    # OUTPUT
    #
    # arr_2d_out: A numpy.array of shape (n, 2) of the same type as arr_2d_in.
    # This array contains the same coordinates as arr_2d_in, but in the case
    # that the input array was not ordered in a continuous fashion, then the
    # returned arr_2d_out will be continuous in order.


    arr_2d = arr_2d_in.copy()
    circ_center = np.mean(arr_2d, axis=0)
    arr_2d_centered = arr_2d - circ_center

    theta_arr = np.arctan2(arr_2d_centered[:,1], arr_2d_centered[:,0])

    # Find the left-most point as the start, which is theta == -pi radians
    cols_sorted_ii = np.argsort(theta_arr, axis=-1, kind='quicksort')
    arr_2d_out = arr_2d[cols_sorted_ii]

    return arr_2d_out


def find_new_start_on_circle(arr_2d_in, img_in):
    # INPUT
    #
    # arr_2d_in: A 2D numpy.array of shape (n, 2) of integer type. This function
    # works closely with "make_continuous_circle()" defined above; the output
    # array of "make_continuous_circle()" is expected as input for this 
    # function. The array, "arr_2d_in",  represents a sequence of 2D coordinates
    # where each entry corresponds to index (pixel) coordinates of an image
    # as [row, column]. These coordinates should represent the perimeter of a
    # circle and are expected to be continuous. The corresponding image relevant
    # for these coordinates is "img_in". This function will reorder these
    # coordinates such that the first coordinate, or start of the circle, will
    # be on a white pixel cooresponding to a grain boundary. This is necessary
    # in order for the grain-boundary-intersection algorithms to work properly.
    # Moreover, this new, start coordinate will be duplicated and appended to
    # the end of the 2D array, thus making it cyclic/periodic.
    #
    # img_in: A 2D numpy.array of unsigned 8-bit integers representing an image.
    # This 2D matrix of grayscale values should only contain 0 (black) and 255
    # (white) pixels, where white pixels correspond to metal grain boundaries. 
    #
    #
    # OUTPUT
    #
    # arr_2d_out: A 2D numpy.array of shape (n+1, 2) of integer type is
    # returned. Other than the one coordinate that is duplicated, this array
    # is simply a reorder version of the input array, arr_2d_in.


    arr_2d = arr_2d_in.copy()
    img0 = img_in.copy()
    num_pnts = arr_2d.shape[0]

    start_i = 0
    for m, cur_pnt in enumerate(arr_2d):
        cur_r = cur_pnt[0]
        cur_c = cur_pnt[1]
        cur_val = img0[cur_r, cur_c]

        if cur_val > 0:
            start_i = m
            break

    arr_temp_1 = arr_2d[start_i:num_pnts, :]
    arr_temp_2 = arr_2d[0:start_i, :]
    arr_temp_3 = arr_2d[start_i, :]

    arr_2d_out = np.vstack((arr_temp_1, arr_temp_2, arr_temp_3))
    return arr_2d_out


def measure_line_dist(segs_local_arr_in, line_global_arr_in):
    # INPUT
    #
    # segs_local_arr_in: A 2D numpy.array of shape (n, 2) containing integers
    # where each row will be used as start and stop indices for 
    # "line_global_arr_in". Each slice of "line_global_arr_in" represents a
    # continuous line in 2D space, in this case, corresponding to a single
    # segment of pixels between grain boundaries.
    #
    # line_global_arr_in: A 2D numpy.array of shape (m, 2) containing integers
    # where each row represents a single 2D coordinate of a pixel within an 
    # image via indices given as [row, column]. The sequence of coordinates
    # given by "line_global_arr_in" represent a continuous line.
    #
    #
    # OUTPUT
    #
    # seg_dist_arr_out: A 1D numpy.array of length, n (see "segs_local_arr_in").
    # This array contains the pixel distance (as a float) between the start and
    # stop indices given by "segs_local_arr_in" along line "line_global_arr_in"
    # using the Pythagorean theorem.


    seg_dist_arr_out = np.zeros(segs_local_arr_in.shape[0], dtype=np.float64)
    line_global_r = line_global_arr_in[:,0]
    line_global_c = line_global_arr_in[:,1]
    
    for m, cur_seg in enumerate(segs_local_arr_in):
        cur_seg_i_start = cur_seg[0]
        cur_seg_i_end = cur_seg[1]

        pixel_coord_start_r = line_global_r[cur_seg_i_start]
        pixel_coord_start_c = line_global_c[cur_seg_i_start]
        pixel_coord_end_r = line_global_r[cur_seg_i_end]
        pixel_coord_end_c = line_global_c[cur_seg_i_end]

        temp_diff_1 = (pixel_coord_start_r - pixel_coord_end_r)**2
        temp_diff_2 = (pixel_coord_start_c - pixel_coord_end_c)**2
        pixel_global_distance = np.sqrt(temp_diff_1 + temp_diff_2)

        seg_dist_arr_out[m] = pixel_global_distance
    
    return seg_dist_arr_out


def measure_circular_dist(segs_local_arr_in, circ_global_arr_in):
    # INPUT
    #
    # segs_local_arr_in: A 2D numpy.array of shape (n, 2) containing integers
    # where each row will be used as start and stop indices for 
    # "circ_global_arr_in". Each slice of "circ_global_arr_in" represents a
    # continuous arc in 2D space, in this case, corresponding to a single
    # segment of pixels between grain boundaries.
    #
    # circ_global_arr_in: A 2D numpy.array of shape (m, 2) containing integers
    # where each row represents a single 2D coordinate of a pixel within an 
    # image via indices given as [row, column]. The sequence of coordinates
    # given by "circ_global_arr_in" represent a continuous circle.
    #
    #
    # OUTPUT
    #
    # seg_dist_arr_out: A 1D numpy.array of length, n (see "segs_local_arr_in").
    # This array contains the pixel distance (as a float) between the start and
    # stop indices given by "segs_local_arr_in" along circle 
    # "circ_global_arr_in" using an arc-length method based on the radius of the
    # circle and the angle between the start and stop coordinates.


    seg_dist_arr_out = np.zeros(segs_local_arr_in.shape[0], dtype=np.float64)
    circ_global_r = circ_global_arr_in[:,0]
    circ_global_c = circ_global_arr_in[:,1]

    # Calculate the center of the circle
    circ_center = np.mean(circ_global_arr_in, axis=0)
    circ_center_r = circ_center[0]
    circ_center_c = circ_center[1]

    for m, cur_seg in enumerate(segs_local_arr_in):
        cur_seg_i_start = cur_seg[0]
        cur_seg_i_end = cur_seg[1]

        pixel_coord_start_r = circ_global_r[cur_seg_i_start]
        pixel_coord_start_c = circ_global_c[cur_seg_i_start]
        pixel_coord_end_r = circ_global_r[cur_seg_i_end]
        pixel_coord_end_c = circ_global_c[cur_seg_i_end]

        radial_vec_a = np.array([pixel_coord_start_r - circ_center_r, \
                                 pixel_coord_start_c - circ_center_c])
        radial_vec_b = np.array([pixel_coord_end_r - circ_center_r, \
                                 pixel_coord_end_c - circ_center_c])
        
        # It's a circle, so the magnitude of this vector is also the radius,
        # and the radius of vector A (the start) should be approximately the
        # same as the radius of vector B (the end),
        radial_vec_a_mag = np.sqrt((radial_vec_a[0])**2 + (radial_vec_a[1])**2)
        radial_vec_a = radial_vec_a/radial_vec_a_mag # Make unit vector

        radial_vec_b_mag = np.sqrt((radial_vec_b[0])**2 + (radial_vec_b[1])**2)
        radial_vec_b = radial_vec_b/radial_vec_b_mag # Make unit vector

        temp_clipped_dot = np.clip(np.dot(radial_vec_a, radial_vec_b), -1, 1)
        cur_theta = np.absolute(np.arccos(temp_clipped_dot))

        # Even if they are about the same, there will be some round-off errors
        # and subtle differences since discrete, pixel coordinates are being
        # used. So, might as well take the average.
        cur_radius = (radial_vec_a_mag + radial_vec_b_mag)/2.0
        pixel_global_distance = cur_radius*cur_theta

        seg_dist_arr_out[m] = pixel_global_distance
    
    return seg_dist_arr_out


def mark_segments_on_image(img_in, segs_local_arr_in, segs_global_arr_in):
    # INPUT
    #
    # img_in: A 2D numpy.array of shape (m, n) containing unsigned 8-bit 
    # integers (i.e., a gray scale image). This image will be marked in specific
    # locations with gray pixels of intensity 150 (out of 255) based on the 
    # inputs given by "segs_local_arr_in" and "segs_global_arr_in".
    #
    # segs_local_arr_in: A 2D numpy.array of shape (p, 2) containing integers
    # that represent the start and stop indices within the array of coordinates
    # defined in "segs_global_arr_in" which will be marked gray on "img_in".
    # Each row contains the start and end indices that define a slice as
    # [start_index, stop_index]. In this case, the individual slices defined
    # by "segs_local_arr_in" correspond to continuous slices of pixels along 
    # "segs_global_arr_in" that represent pixels between grain boundaries.
    #
    # segs_global_arr_in: A 2D numpy.array of shape (q, 2) containing integers
    # that represent a continuous sequence of coordinates representing either a
    # line or a circle. Each row in "segs_global_arr_in" represents one 2D 
    # coordinate corresponding to the row-index and column-index of "img_in",
    # given as [row_index, column_index]. 
    #
    #
    # OUTPUT
    #
    # img_out: A 2D numpy.array with the same type and shape as "img_in". 
    # Specific pixels will have been marked gray with pixel intensity 150 (out
    # of 255) based on "segs_global_arr_in" and "segs_local_arr_in". 


    img_out = img_in.copy()

    segs_global_r = segs_global_arr_in[:,0]
    segs_global_c = segs_global_arr_in[:,1]
    
    for cur_seg in segs_local_arr_in:
        cur_seg_i_start = cur_seg[0]
        cur_seg_i_end = cur_seg[1]

        pixel_arr_coord_r = segs_global_r[cur_seg_i_start:cur_seg_i_end]
        pixel_arr_coord_c = segs_global_c[cur_seg_i_start:cur_seg_i_end]

        img_out[pixel_arr_coord_r, pixel_arr_coord_c] = 150
    
    return img_out


def convert_2d_list_to_str(list_in):
    # INPUT
    #
    # list_in: 2D python list as input containing values that can be
    # converted to strings.
    #
    #
    # OUTPUT
    #
    # list_out: A 2D python list with all values converted to strings


    list_out = []
    for cur_row in list_in:
        cur_row_str = []

        for cur_val in cur_row:
            cur_row_str.append(str(cur_val))

        list_out.append(cur_row_str)

    return list_out


def save_csv(list_in, path_in, header_in):
    # INPUT
    #
    # list_in: A 2D Numpy array containing strings and/or numbers.
    #
    # path_in: File path as a string
    #
    # header_in: If list_in is of size (m, n), then header_in
    # should be a 1D Python list of size (n) containing strings. 
    # This list will be written first (i.e., this is the header line)


    list_in_str = convert_2d_list_to_str(list_in.tolist())

    print(f"\nWriting segment distances to: {path_in}")

    with open(path_in, 'w', newline='') as file_obj:
        csv_writer = csv.writer(file_obj, delimiter=',')

        csv_writer.writerow(header_in)

        for cur_row in list_in_str:
             csv_writer.writerow(cur_row)