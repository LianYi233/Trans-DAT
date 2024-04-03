import jpeg4py
import cv2 as cv
from PIL import Image
import numpy as np
import os

davis_palette = np.repeat(np.expand_dims(np.arange(0, 256), 1), 3, 1).astype(np.uint8)
davis_palette[:22, :] = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                         [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                         [64, 0, 0], [191, 0, 0], [64, 128, 0], [191, 128, 0],
                         [64, 0, 128], [191, 0, 128], [64, 128, 128], [191, 128, 128],
                         [0, 64, 0], [128, 64, 0], [0, 191, 0], [128, 191, 0],
                         [0, 64, 128], [128, 64, 128]]


def default_image_loader(path):
    """The default image loader, reads the image from the given path. It first tries to use the jpeg4py_loader,
    but reverts to the opencv_loader if the former is not available."""
    # print(path)
    if default_image_loader.use_jpeg4py is None:
        # Try using jpeg4py
        # im = jpeg4py_loader(path)
        im = opencv_loader(path)
        if im is None:
            default_image_loader.use_jpeg4py = False
            print('Using opencv_loader instead.')
        else:
            default_image_loader.use_jpeg4py = True
            return im
    if default_image_loader.use_jpeg4py:
        return jpeg4py_loader(path)
    return opencv_loader(path)

default_image_loader.use_jpeg4py = None


def X2Cube_3bands(img, bandNum):   # TODO with band selection.
    # input NxMx1   output selected nxmx3
    M, N = img.shape
    col_extent = N - bandNum + 1
    row_extent = M - bandNum + 1
    start_idx = np.arange(bandNum)[:, None] * N + np.arange(bandNum)
    didx = M * N * np.arange(1)
    start_idx = (didx[:, None] + start_idx.ravel()).reshape((-1, bandNum, bandNum))
    offset_idx = np.arange(row_extent)[:, None] * N + np.arange(col_extent)
    out = np.take(img, start_idx.ravel()[:, None] + offset_idx[::bandNum, ::bandNum].ravel())
    out = np.transpose(out)
    DataCube = out.reshape(M//bandNum, N//bandNum, bandNum*bandNum)

    # select 3-band with max sum
    band_sum = [0]*bandNum*bandNum
    for ii in range(bandNum*bandNum):
        band_sum[ii] = np.sum(DataCube[:, :, ii]**2)        # a simple band selection by pixel-wise sum square
    max3bands = sorted(band_sum, reverse=True)[0:3]
    selected = np.stack((DataCube[:, :, band_sum.index(max3bands[0])], DataCube[:, :, band_sum.index(max3bands[1])], DataCube[:, :, band_sum.index(max3bands[2])]))
    selected = np.transpose(selected, (1, 2, 0))

    # return DataCube
    return selected

def jpeg4py_loader(img_path, use_hsi=True):
    """ Image reading using jpeg4py https://github.com/ajkxyz/jpeg4py"""
    try:
        img_fc = cv.imread(img_path)     # default: false-color only
        if use_hsi:
            hsi_path = img_path.replace('-FalseColor', '')
            hsi_path = hsi_path.replace('jpg', 'png')
            img_hsi = cv.imread(hsi_path)
            img_hsi = X2Cube_3bands(img_hsi[:, :, 0], bandNum=int(img_hsi.shape[0] / img_fc.shape[0]))
            img_fc = np.concatenate([img_fc, img_hsi], axis=2)  # channel-wise concat: (n, m, c1+c2)
        return img_fc
        # return jpeg4py.JPEG(path).decode()
    except Exception as e:
        print('ERROR: Could not read image "{}"'.format(img_path))
        print(e)
        return None


def opencv_loader(path):
    """ Read image using opencv's imread function and returns it in rgb format"""
    try:
        im = cv.imread(path, cv.IMREAD_COLOR)

        # convert to rgb and return
        return cv.cvtColor(im, cv.COLOR_BGR2RGB)
    except Exception as e:
        print('ERROR: Could not read image "{}"'.format(path))
        print(e)
        return None


def jpeg4py_loader_w_failsafe(path):
    """ Image reading using jpeg4py https://github.com/ajkxyz/jpeg4py"""
    try:
        return jpeg4py.JPEG(path).decode()
    except:
        try:
            im = cv.imread(path, cv.IMREAD_COLOR)

            # convert to rgb and return
            return cv.cvtColor(im, cv.COLOR_BGR2RGB)
        except Exception as e:
            print('ERROR: Could not read image "{}"'.format(path))
            print(e)
            return None


def opencv_seg_loader(path):
    """ Read segmentation annotation using opencv's imread function"""
    try:
        return cv.imread(path)
    except Exception as e:
        print('ERROR: Could not read image "{}"'.format(path))
        print(e)
        return None


def imread_indexed(filename):
    """ Load indexed image with given filename. Used to read segmentation annotations."""

    im = Image.open(filename)

    annotation = np.atleast_3d(im)[...,0]
    return annotation


def imwrite_indexed(filename, array, color_palette=None):
    """ Save indexed image as png. Used to save segmentation annotation."""

    if color_palette is None:
        color_palette = davis_palette

    if np.atleast_3d(array).shape[2] != 1:
        raise Exception("Saving indexed PNGs requires 2D array.")

    im = Image.fromarray(array)
    im.putpalette(color_palette.ravel())
    im.save(filename, format='PNG')