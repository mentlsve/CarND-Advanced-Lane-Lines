class ColorThresholding:

    def __init__(self):
        pass

    def extractYellow(self, image_BGR, min_thresh=(28,28,50), max_thresh=(58,100,100)):
        return self._thresholdColor(image_BGR, min_thresh, max_thresh)


    def extractWhite(self, image_BGR, min_thresh=(20,0,75), max_thresh=(360,10,100)):
        return self._thresholdColor(image_BGR, min_thresh, max_thresh)


    def extractWhiteAndYellow(self, image_BGR):
        yellow = self.extractYellow(image_BGR)
        white = self.extractWhite(image_BGR)
        yellow_and_white = np.zeros_like(yellow)
        filter = (yellow > 0) | (white > 0)
        yellow_and_white[filter] = 1
        return yellow_and_white


    def _thresholdColor(self, image_BGR, min_thresh, max_thresh):

        min_thresh_normalized = (min_thresh[0] * 0.7, min_thresh[1] * 2.55, min_thresh[2] * 2.55)
        max_thresh_normalized = (max_thresh[0] * 0.7, max_thresh[1] * 2.55, max_thresh[2] * 2.55)

        image_HSV = cv2.cvtColor(image_BGR, cv2.COLOR_BGR2HSV).astype(np.float)
        hsv_filtered = cv2.inRange(image_HSV, min_thresh_normalized, max_thresh_normalized)

        return hsv_filtered
