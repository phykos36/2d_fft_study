import numpy as np
import cv2
import matplotlib.pyplot as plt

def fourier(src):
    fimg = np.fft.fft2(src)
    fimg = np.fft.fftshift(fimg)
    mag = 20*np.log(np.abs(fimg))
    return mag

def highpass_filter(src, a):
    src = np.fft.fft2(src)
    h, w = src.shape
    cy, cx = int(h/2), int(w/2)
    rh, rw = int(a*cy), int(a*cx)
    fsrc = np.fft.fftshift(src)
    fdst = fsrc.copy()
    fdst[cy-rh:cy+rh, cx-rw:cx+rw] = 0
    fdst = np.fft.fftshift(fdst)
    dst = np.fft.ifft2(fdst)
    return np.uint8(dst.real)

def lowpass_filter(src, a):
    src = np.fft.fft2(src)
    h, w = src.shape
    cy, cx = int(h/2), int(w/2)
    rh, rw = int(a*cy), int(a*cx)
    fsrc = np.fft.fftshift(src)
    plt.subplot(121)
    plt.imshow(np.uint8(fsrc.real), cmap = 'gray')
    fdst = np.zeros(src.shape, dtype=complex)
    fdst[cy-rh:cy+rh, cx-rw:cx+rw] = fsrc[cy-rh:cy+rh, cx-rw:cx+rw]
    # plt.subplot(122)
    # plt.imshow(np.uint8(fdst.real), cmap = 'gray')
    # plt.show()
    fdst = np.fft.fftshift(fdst)
    dst = np.fft.ifft2(fdst)
    return np.uint8(dst.real)    

def main(path):
    img = cv2.imread(path)
    # compare = 60/img.shape
    # img = cv2.resize(img,(60,60)) # この前にサイズ比からlowpassの倍率を決める
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mag = fourier(gray)
    limg = lowpass_filter(gray,0.9)
    himg = highpass_filter(gray,0.1)
    # bler = lowpass_filter(gray,compare)
    plt.subplot(221)
    plt.imshow(gray, cmap = 'gray')
    plt.subplot(222)
    plt.imshow(mag, cmap = 'gray')
    plt.subplot(223)
    plt.imshow(limg, cmap = 'gray')
    plt.subplot(224)
    plt.imshow(himg, cmap = 'gray')
    plt.show()


while True:
    inp = input('file name? >')
    if inp!='':break
main(inp)
