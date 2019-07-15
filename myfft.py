import numpy as np
import cv2
import matplotlib.pyplot as plt
import math

def power_spectre(src):
    fimg = np.fft.fft2(src)
    mag = 20*np.log(np.abs(fimg))
    return mag

def highpass_filter(src, a):
    fsrc = np.fft.fft2(src)
    h, w = src.shape
    cy, cx = int(h/2), int(w/2)
    rh, rw = int(a*cy), int(a*cx)
    fsrc = 20*np.log(fsrc)
    shift_fsrc = np.fft.fftshift(fsrc)
    shift_fdst = shift_fsrc.copy()
    shift_fdst[cy-rh:cy+rh, cx-rw:cx+rw] = 0
    fdst = np.fft.ifftshift(shift_fdst)
    fdst = pow(math.e,fdst/20)
    dst = np.fft.ifft2(fdst)
    return np.uint8(dst.real)

def lowpass_filter(src, a):
    plt.subplot(131)
    plt.imshow(src, cmap = 'gray')
    fsrc = np.fft.fft2(src)
    h, w = src.shape
    cy, cx = int(h/2), int(w/2)
    rh, rw = int(a*cy), int(a*cx)
    fsrc = 20*np.log(fsrc)
    plt.subplot(132)
    plt.imshow(np.uint8(fsrc.real), cmap = 'gray')
    fdst = fsrc
    fdst[cy-rh:cy+rh, cx-rw:cx+rw] = 0
    plt.subplot(133)
    plt.imshow(np.uint8(fdst.real), cmap = 'gray')
    plt.show()
    fdst = pow(math.e,fdst/20)
    dst = np.fft.ifft2(fdst)
    return np.uint8(dst.real)    

def main(path):
    src = cv2.imread(path)
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    mag = power_spectre(gray)
    low_img = lowpass_filter(gray,0.9)
    high_img = highpass_filter(gray,0.001)
    plt.subplot(221)
    plt.imshow(gray, cmap = 'gray')
    plt.subplot(222)
    plt.imshow(mag, cmap = 'gray')
    plt.subplot(223)
    plt.imshow(low_img, cmap = 'gray')
    plt.subplot(224)
    plt.imshow(high_img, cmap = 'gray')
    plt.show()
    cv2.imwrite("out.png", low_img)


# while True:
#     input = input('file name? >')
#     if input!='':break
main('lena_gray.png')
