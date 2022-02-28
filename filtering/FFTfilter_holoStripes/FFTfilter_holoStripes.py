import os
import numpy as np
import cv2
import math


def filter_stripes(img):
    # read input as grayscale
    # img = cv2.imread('pattern_lines.png', 0)
    hh, ww = img.shape

    # get min and max and mean values of img
    img_min = np.amin(img)
    img_max = np.amax(img)
    img_mean = int(np.mean(img))

    # pad the image to dimension a power of 2
    hhh = math.ceil(math.log2(hh))
    hhh = int(math.pow(2,hhh))
    www = math.ceil(math.log2(ww))
    www = int(math.pow(2,www))
    imgp = np.full((hhh,www), img_mean, dtype=np.uint8)
    imgp[0:hh, 0:ww] = img

    # convert image to floats and do dft saving as complex output
    dft = cv2.dft(np.float32(imgp), flags = cv2.DFT_COMPLEX_OUTPUT)

    # apply shift of origin from upper left corner to center of image
    dft_shift = np.fft.fftshift(dft)

    # extract magnitude and phase images
    mag, phase = cv2.cartToPolar(dft_shift[:,:,0], dft_shift[:,:,1])

    # get spectrum
    spec = np.log(mag) / 20
    min, max = np.amin(spec, (0,1)), np.amax(spec, (0,1))

    # threshold the spectrum to find bright spots
    thresh = (255*spec).astype(np.uint8)
    thresh = cv2.threshold(thresh, 155, 255, cv2.THRESH_BINARY)[1]

    # cover the center columns of thresh with black
    xc = www // 2
    cv2.line(thresh, (xc,0), (xc,hhh-1), 0, 5)

    # get the x coordinates of the bright spots
    points = np.column_stack(np.nonzero(thresh))
    # print(points)

    # create mask from spectrum drawing vertical lines at bright spots
    mask = thresh.copy()
    for p in points:
        x = p[0]
        cv2.line(mask, (x,0), (x,hhh-1), 255, 5)

    # apply mask to magnitude such that magnitude is made black where mask is white
    mag[mask!=0] = 0

    # convert new magnitude and old phase into cartesian real and imaginary components
    real, imag = cv2.polarToCart(mag, phase)

    # combine cartesian components into one complex image
    back = cv2.merge([real, imag])

    # shift origin from center to upper left corner
    back_ishift = np.fft.ifftshift(back)

    # do idft saving as complex output
    img_back = cv2.idft(back_ishift)

    # combine complex components into original image again
    img_back = cv2.magnitude(img_back[:,:,0], img_back[:,:,1])

    # crop to original size
    img_back = img_back[0:hh, 0:ww]

    # re-normalize to 8-bits in range of original
    min, max = np.amin(img_back, (0,1)), np.amax(img_back, (0,1))
    notched = cv2.normalize(img_back, None, alpha=img_min, beta=img_max, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)


    # cv2.imshow("ORIGINAL", img)
    # # cv2.imshow("PADDED", imgp)
    # # cv2.imshow("MAG", mag)
    # # cv2.imshow("PHASE", phase)
    # cv2.imshow("SPECTRUM", spec)
    # cv2.imshow("THRESH", thresh)
    # cv2.imshow("MASK", mask)
    # cv2.imshow("NOTCHED", notched)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return notched
    # # write result to disk
    # cv2.imwrite("pattern_lines_spectrum.png", (255*spec).clip(0,255).astype(np.uint8))
    # cv2.imwrite("pattern_lines_thresh.png", thresh)
    # cv2.imwrite("pattern_lines_mask.png", mask)
    # cv2.imwrite("pattern_lines_notched.png", notched)


# # Press the green button in the gutter to run the script.
# if __name__ == '__main__':
dirPath = '.\images'
files = os.listdir(dirPath)
for file in files:
    imgPath = os.path.join(dirPath, file)
    print(imgPath)
    img = cv2.imread(imgPath)
    b, g, r = cv2.split(img)
    b = filter_stripes(b)
    g = filter_stripes(g)
    r = filter_stripes(r)
    imresult = cv2.merge([b, g, r])
    cv2.imwrite(f'./filtered_images/filtered_{file}', imresult)

    # cv2.imshow("before", img)
    # cv2.imshow("after", imresult)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
