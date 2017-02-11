# %load transformations.py
class PerspectiveTransform:

    def __init__(self):

        top_left = (588 , 454)
        top_right = (696 , 454)
        bottom_right = (1049, 684)
        bottom_left = (269, 684)

        self.src_points = np.float32([
                (588 , 454),    # top_left
                (696 , 454),    # top_right
                (1049, 684),    # bottom_right
                (269, 684)])    # bottom_left

        self.dst_points = np.float32([
                (320, 0),       # top_left
                (960, 0),       # top_right
                (960, 720),     # bottom_right
                (320, 720)])    # bottom_left

        self.M = cv2.getPerspectiveTransform(self.src_points, self.dst_points)
        self.invM = cv2.getPerspectiveTransform(self.dst_points, self.src_points)

    def toBirdsEyeView(self, img):
        return cv2.warpPerspective(img, self.M, (img.shape[1], img.shape[0]))

    def toCarCameraView(self, img):
        return cv2.warpPerspective(img, self.invM, (img.shape[1], img.shape[0]))

    def getSrcPoints(self):
        return self.src_points

    def getDstPoints(self):
        return self.dst_points
