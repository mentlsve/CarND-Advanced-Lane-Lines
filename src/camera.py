class Camera:

    def __init__(self):
        pass

    def calibrateCamera(self, p_fileNameImages, p_nx, p_ny, p_imgSize):

        width = p_imgSize[0]
        height = p_imgSize[1]

        # objpoints will hold the 3D points in real world space.
        # since we know the concrete chessboard image, we also know the coordinates
        # z will be always zero, since the chessboard is on a plane
        # objpoints is the SAME for all images
        objPointsAllImages = []

        # imgpoints holds the 2D points in image plane (the distorted image)
        imgPointsAllImages = []

        # prepare object points
        objPointsSingleImage = np.zeros((p_nx * p_ny, 3), np.float32)
        objPointsSingleImage[:, :2] = np.mgrid[0:p_nx, 0:p_ny].T.reshape(-1, 2)

        for fname in p_fileNameImages:
            imgPointsSingleImage = self.getImagePoints(fname, p_nx, p_ny, p_imgSize);
            if len(imgPointsSingleImage) > 0:
                objPointsAllImages.append(objPointsSingleImage)
                imgPointsAllImages.append(imgPointsSingleImage[0])

        imgPointsAllImages = np.array(imgPointsAllImages)
        objPointsAllImages = np.array(objPointsAllImages)

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objPointsAllImages, imgPointsAllImages, (height, width), None, None)

        self.mtx = mtx
        self.dist = dist

    def getImagePoints(self, p_fname, p_nx, p_ny, p_imgSize):

        img = cv2.imread(p_fname)

        # imgpoints holds the 2D points in image plane (the distorted image)
        imgpoints = []

        resizedImage = cv2.resize(img, p_imgSize)

        # find the chessboard corners
        gray = cv2.cvtColor(resizedImage, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (p_nx, p_ny), None)

        # if corners are found, add object points, image points
        if ret == True:
            imgpoints.append(corners)
        else:
            print('No corners detected in: ', p_fname)


        return imgpoints

    def undistort(self, image_BGR):
        undistorted = cv2.undistort(image_BGR, self.mtx, self.dist, None, self.mtx )
        return undistorted
