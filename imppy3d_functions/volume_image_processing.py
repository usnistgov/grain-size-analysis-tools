import numpy as np


def pad_image_boundary(img_arr_in, cval_in=0, n_pad_in=1, quiet_in=False):
    """
    Adds enlarges the image array in all directions by one extra
    row of pixels, and then pads these locations with values equal
    to cval_in. So, for a 2D image of size equal to (rows, cols), 
    the output image with extended boundaries in all directions will
    be of size (rows + 2, cols + 2). In 3D, which is a sequence of
    images, (num_imgs, rows, cols), the output will be a similarly
    extended image array, (num_imgs + 2, rows + 2, cols + 2). Note,
    the SciKit-Image library has a more powerful function if it is
    needed, called skimage.util.pad().

    ---- INPUT ARGUMENTS ---- 
    [[img_arr_in]]: A 2D or 3D Numpy array representing the image 
        sequence. If 3D, is important that this is a Numpy array and not
        a Python list of Numpy matrices. If 3D, the shape of img_arr_in
        is expected to be as (num_images, num_pixel_rows,
        num_pixel_cols). If 2D, the shape of img_arr_in is expected to
        be as (num_pixel_rows, num_pixel_cols). It is expected that the
        image(s) are single-channel (i.e., grayscale), and the data
        type of the values are np.uint8.

    cval_in: Constant padding value to be used in the extended regions
        of the image array. This should be an integer between 0 and 255,
        inclusive. 

    n_pad_in: An integer value to determine how much to pad by. For 
        example, if img_arr_in is of shape (5, 10, 20), and n_pad is
        equal to 3, then three extra rows/columns will be created on
        all sides. The resulting size would be (5+3*2, 10+3*2, 20+3*2).

    quiet_in: A boolean that determines if this function should print
        any statements to standard output. If False (default), outputs  
        are written. Conversely, if True, outputs are suppressed. This
        is particularly useful in the event of batch processing.

    ---- RETURNED ----
    [[img_arr_out]]: The image array with extended boundaries in all
        dimensions. The input image array will be in the center of this
        output array, and the data type will be uint8. 

    ---- SIDE EFFECTS ---- 
    The input array may be affected since a deep copy is not made in 
    order to be more memory efficient. Strings are printed to standard
    output. Nothing is written to the hard drive.
    """

    # ---- Start Local Copies ----
    img_arr = img_arr_in # Makes a new view -- NOT a deep copy
    cval = cval_in
    n_pad1 = np.around(n_pad_in).astype(np.int32)
    n_pad2 = (2*n_pad1).astype(np.int32)
    quiet = quiet_in
    # ---- End Start Local Copies ----

    if not quiet_in:
        print(f"\nExtending the boundaries of the image data...\n"\
            f"    Values in the padded regions will be set to: {cval}")

    img_shape = img_arr_in.shape
    if len(img_shape) == 3:
        num_imgs = img_shape[0]
        num_rows = img_shape[1]
        num_cols = img_shape[2]

        img_arr_out = np.ones((num_imgs+n_pad2, \
            num_rows+n_pad2, num_cols+n_pad2), dtype=np.uint8)*cval

        img_arr_out[n_pad1:num_imgs+n_pad1, n_pad1:num_rows+n_pad1, \
            n_pad1:num_cols+n_pad1] = img_arr

    elif len(img_shape) == 2:
        num_rows = img_shape[0]
        num_cols = img_shape[1]

        img_arr_out = np.ones((num_rows+n_pad2, num_cols+n_pad2), \
            dtype=np.uint8)*cval

        img_arr_out[n_pad1:num_rows+n_pad1, n_pad1:num_cols+n_pad1] = img_arr

    else:
        if not quiet_in:
            print(f"\nERROR: Can only pad the boundary of an image array "\
                f"in 2D or 3D.\nCurrent image shape is: {img_shape}")

    if not quiet_in:
        print(f"\nSuccesfully padded the image boundaries!")

    return img_arr_out
