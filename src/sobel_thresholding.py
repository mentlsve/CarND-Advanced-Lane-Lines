class SobelThresholding:

    def __init__(self):
        pass

    def sobel_thresh(self, gray, orient='x', sobel_kernel=3, sobel_thresh=(0, 255)):

        if orient == 'x':
            sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
        if orient == 'y':
            sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1)

        abs_sobel = np.absolute(sobel)

        scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))

        sbinary = np.zeros_like(scaled_sobel)
        filter = (scaled_sobel >= sobel_thresh[0]) & (scaled_sobel <= sobel_thresh[1])
        sbinary[filter] = 1

        return sbinary

    def mag_thresh(self, gray, sobel_kernel=3, mag_thresh=(0, 255)):

        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)

        abs_sobelxy = np.sqrt(np.square(sobelx) + np.square(sobely))

        scaled_sobel = np.uint8(255 * abs_sobelxy / np.max(abs_sobelxy))

        sbinary = np.zeros_like(scaled_sobel)
        filter = (scaled_sobel > mag_thresh[0]) & (scaled_sobel < mag_thresh[1])
        sbinary[filter] = 1

        return sbinary

    def dir_thresh(self, gray, sobel_kernel=3, dir_thresh=(0.7, np.pi/2)):

        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

        absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))

        sbinary =  np.zeros_like(absgraddir)
        filter = (absgraddir >= dir_thresh[0]) & (absgraddir <= dir_thresh[1])
        sbinary[filter] = 1

        return sbinary
