import numpy as np
from scipy import ndimage


def multispectral_image_zoom(img, zoom_factor, **kwargs):
    '''
    Parameters
    
    img: Numpy array: 3D array
    zoom_factor: float : Factor of zoom. If equal to 1.0 dont make nothing. Factor applied in the 2 first dimensions.
    kwargs: Others parameters of scipy.ndimage.zoom() function.
    
    Return
    
    out: Numpy array: 3D array with the same shape of img.
    
    '''
    h, w = img.shape[:2]

    # For multichannel images we don't want to apply the zoom factor to the RGB
    # dimension, so instead we create a tuple of zoom factors, one per array
    # dimension, with 1's for any trailing dimensions after the width and height.
    zoom_tuple = (zoom_factor,) * 2 + (1,) * (img.ndim - 2)

    # Zooming out
    if zoom_factor < 1:

        # Bounding box of the zoomed-out image within the output array
        zh = int(np.round(h * zoom_factor))
        zw = int(np.round(w * zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2

        # Zero-padding
        out = np.zeros_like(img)
        out[top:top+zh, left:left+zw] = ndimage.zoom(img, zoom_tuple, **kwargs)

    # Zooming in
    elif zoom_factor > 1:

        # Bounding box of the zoomed-in region within the input array
        zh = int(np.round(h / zoom_factor))
        zw = int(np.round(w / zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2

        out = ndimage.zoom(img[top:top+zh, left:left+zw], zoom_tuple, **kwargs)

        # `out` might still be slightly larger than `img` due to rounding, so
        # trim off any extra pixels at the edges
        trim_top = ((out.shape[0] - h) // 2)
        trim_left = ((out.shape[1] - w) // 2)
        out = out[trim_top:trim_top+h, trim_left:trim_left+w]

    # If zoom_factor == 1, just return the input array
    else:
        out = img
    return out
    
def batch_multispectral_image_zoom(batch_dat, zoom_factor, **kwargs):
    '''
    Parameters
    
    batch_dat: Numpy array: 4D array. First dimension is sample counter.
    zoom_factor: float : Factor of zoom. If equal to 1.0 dont make nothing. Factor applied in the second and third dimension.
    kwargs: Others parameters of scipy.ndimage.zoom() function.
    
    Return
    
    new_batch: Numpy array: 4D array with the same shape of batch_dat.
    
    '''
    new_batch=batch_dat.copy();
    L=batch_dat.shape[0];
    
    for l in range(L):
        new_batch[l]=multispectral_image_zoom(new_batch[l], zoom_factor, **kwargs);
    
    return new_batch;
    
def batch_multispectral_image_horizontal_flip(X):
    '''
    Flip horizontal of X 4D array. #0:nbatch,1:first_dim,2:second_dim,3:ch_dim
    
    Parameters
    
    X: Numpy array: 4D array
    
    Return
    
    out: Numpy array: 4D array with the same shape of X.
    
    '''
    return np.flip(X,2);
    
def batch_multispectral_image_vertical_flip(X):
    '''
    Flip vertical of X 4D array. #0:nbatch,1:first_dim,2:second_dim,3:ch_dim
    
    Parameters
    
    X: Numpy array: 4D array
    
    Return
    
    out: Numpy array: 4D array with the same shape of X.
    
    '''
    return np.flip(X,1);
    
def batch_multispectral_image_rotate(X,angle):
    '''
    Rotate X 4D array. #0:nbatch,1:first_dim,2:second_dim,3:ch_dim
    
    Parameters
    
    X: Numpy array: 4D array
    angle: float : angle in degree.
    
    Return
    
    out: Numpy array: 4D array with the same shape of X.
    
    '''
    return ndimage.rotate(X, angle,axes=(1,2), reshape=False);
    
