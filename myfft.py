import numpy as np
import cv2
import matplotlib.pyplot as plt
import math

def fourier(src):
    fimg = np.fft.fft2(src)
    mag = 20*np.log(np.abs(fimg))
    return mag

def highpass_filter(src, a):
    src = np.fft.fft2(src)
    h, w = src.shape
    cy, cx = int(h/2), int(w/2)
    rh, rw = int(a*cy), int(a*cx)
    src = 20*np.log(src)
    fsrc = np.fft.fftshift(src)
    fdst = fsrc.copy()
    fdst[cy-rh:cy+rh, cx-rw:cx+rw] = 0
    fdst = np.fft.fftshift(fdst)
    fdst = pow(math.e,fdst/20)
    dst = np.fft.ifft2(fdst)
    return np.uint8(dst.real)

def lowpass_filter(src, a):
    plt.subplot(131)
    plt.imshow(src, cmap = 'gray')
    src = np.fft.fft2(src)
    h, w = src.shape
    cy, cx = int(h/2), int(w/2)
    rh, rw = int(a*cy), int(a*cx)
    fsrc = 20*np.log(src)
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
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mag = fourier(gray)
    limg = lowpass_filter(gray,0.8)
    himg = highpass_filter(gray,0.1)
    plt.subplot(221)
    plt.imshow(gray, cmap = 'gray')
    plt.subplot(222)
    plt.imshow(mag, cmap = 'gray')
    plt.subplot(223)
    plt.imshow(limg, cmap = 'gray')
    plt.subplot(224)
    plt.imshow(himg, cmap = 'gray')
    plt.show()
    cv2.imwrite("out.png", limg)


# while True:
#     inp = input('file name? >')
#     if inp!='':break
main('lena_gray.png')
