import sys, os
import numpy as np
import cv2 as cv


def getContours():
    for i in range(0, 63):
        """ load an image """
        img = cv.imread('Binary/%d.jpg' % i)
        """ convert the source to greyscale """
        grey = 255 - cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], np.uint8)
        # grey = cv.erode(cv.dilate(grey, kernel, iterations=3), kernel, iterations=2)
        """ find contours """
        binary, contours, hierarchy = cv.findContours(grey, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        """ crop digit(s) """
        if not os.path.isdir('Segmentation/%d' % i):
            os.mkdir('Segmentation/%d' % i)
        contours = [contour for contour in contours if cv.contourArea(contour) >= 96]
        contours.sort(key=lambda contour: cv.minEnclosingCircle(contour)[0][1])
        """ number of group(s) """
        group = 0
        while group*37 < contours.__len__():
            """ student id """
            if not os.path.isdir('Segmentation/%d/%d' % (i, group*3)):
                os.mkdir('Segmentation/%d/%d' % (i, group*3))
            n = 0
            for contour in sorted(contours[group*37:group*37+8], key=lambda contour: cv.minEnclosingCircle(contour)[0][0]):
                [x, y, w, h] = cv.boundingRect(contour)
                cv.imwrite('Segmentation/%d/%d/%d.jpg' % (i, group*3, n), img[y:y+h, x:x+w])
                n += 1
            """ tel number """
            if not os.path.isdir('Segmentation/%d/%d' % (i, group*3+1)):
                os.mkdir('Segmentation/%d/%d' % (i, group*3+1))
            n = 0
            for contour in sorted(contours[group*37+8:group*37+19], key=lambda contour: cv.minEnclosingCircle(contour)[0][0]):
                [x, y, w, h] = cv.boundingRect(contour)
                cv.imwrite('Segmentation/%d/%d/%d.jpg' % (i, group*3+1, n), img[y:y+h, x:x+w])
                n += 1
            """ id """
            if not os.path.isdir('Segmentation/%d/%d' % (i, group*3+2)):
                os.mkdir('Segmentation/%d/%d' % (i, group*3+2))
            n = 0
            for contour in sorted(contours[group*37+19:group*37+37], key=lambda contour: cv.minEnclosingCircle(contour)[0][0]):
                [x, y, w, h] = cv.boundingRect(contour)
                cv.imwrite('Segmentation/%d/%d/%d.jpg' % (i, group*3+2, n), img[y:y+h, x:x+w])
                n += 1
            group += 1
        for contour in contours:
            [x, y, w, h] = cv.boundingRect(contour)
            cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        """ save """
        cv.imwrite('Segmentation/%d.jpg' % i, img)


if __name__ == '__main__':
    getContours()