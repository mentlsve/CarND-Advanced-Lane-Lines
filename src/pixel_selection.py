class PixelSelection:

    def __init__(self):
        self.ct = ColorThresholding()
        self.st = SobelThresholding()

    def detectLanePixels(self, image_BGR):
        colors =self.ct.extractWhiteAndYellow(image_BGR)
        gray = cv2.cvtColor(image_BGR, cv2.COLOR_BGR2GRAY)
        sobel_x = self.st.sobel_thresh(gray, orient='x', sobel_thresh=(15, 120))
        sobel_y = self.st.sobel_thresh(gray, orient='y', sobel_thresh=(10, 120))
        magnitude = self.st.mag_thresh(gray, mag_thresh=(20, 120))

        sobel = np.zeros_like(sobel_x)
        combined = np.zeros_like(sobel_x)

        sobel[((magnitude == 1) | (sobel_x == 1))] = 1
        sobel[(sobel_y == 1)] = 0
        combined[((colors == 1) | (sobel == 1))] = 1
        combined = cv2.blur(combined,(5,5))

        return combined
