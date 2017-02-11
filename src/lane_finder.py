class Window:

    def __init__(self, top_left, bottom_right, nonzerox, nonzeroy):
        self.top_left = top_left
        self.bottom_right = bottom_right
        self.nonzerox = nonzerox
        self.nonzeroy = nonzeroy

    def getTopLeft(self):
        return self.top_left;

    def getBottomRight(self):
        return self.bottom_right;

    def getNonZeroPixels(self):
        self.good_inds = (
            (self.nonzeroy >= self.top_left[1]) &
            (self.nonzeroy < self.bottom_right[1]) &
            (self.nonzerox >= self.top_left[0]) &
            (self.nonzerox < self.bottom_right[0])
        ).nonzero()[0]

        return self.good_inds

    def getMeanXofNonZeroPixels(self):
        mean = np.int(np.mean(self.nonzerox[self.good_inds]))
        return mean

class WindowFactory:

    def __init__(self, nwindows, binary_warped):
        self.nwindows = nwindows
        self.binary_warped = binary_warped

        self.window_height = np.int(binary_warped.shape[0]/nwindows) # Height of one vertical slice

        # Set the width of the windows +/- margin
        self.margin = 100

        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        self.nonzeroy = np.array(nonzero[0])
        self.nonzerox = np.array(nonzero[1])

    def getWindowLeft(self, number, leftx_current):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = self.binary_warped.shape[0] - (number+1) * self.window_height
        win_y_high = self.binary_warped.shape[0] - number * self.window_height
        win_xleft_low = leftx_current - self.margin
        win_xleft_high = leftx_current + self.margin

        window_left = Window((win_xleft_low, win_y_low), (win_xleft_high, win_y_high), self.nonzerox, self.nonzeroy)

        return window_left

    def getWindowRight(self, number, rightx_current):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = self.binary_warped.shape[0] - (number+1)* self.window_height
        win_y_high = self.binary_warped.shape[0] - number * self.window_height

        win_xright_low = rightx_current - self.margin
        win_xright_high = rightx_current + self.margin

        window_right = Window((win_xright_low, win_y_low), (win_xright_high, win_y_high), self.nonzerox, self.nonzeroy)

        return window_right

class LaneFinder:

    def __init__(self, binary_warped, windows, lastLeftLanes, lastRightLanes,  interactiveMode = False):

        self.binary_warped = binary_warped
        self.interactiveMode = interactiveMode
        self.nwindows = windows

        self.lastLeftLanes = lastLeftLanes
        self.lastRightLanes = lastRightLanes

        # Create empty lists to receive left and right lane pixel indices
        self.left_lane_inds = []
        self.right_lane_inds = []
        self.leftx_current = 0
        self.rightx_current = 0

        self.minpix = 50

        self.ym_per_pix = 30/720 # meters per pixel in y dimension
        self.xm_per_pix = 3.7/700

        # Take a histogram of the bottom half of the image
        height = int(binary_warped.shape[0]/3)
        histogram = np.sum(binary_warped[720-height:,:], axis=0)
        #print(binary_warped.shape)

        # binary_warped.shape = (720, 1280)
        # histogram.shape = (1280,)

        # Create an output image to draw on and  visualize the result
        if interactiveMode:
            self.out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
            cv2.rectangle(self.out_img, (0, 720-height),(640, 720),(255,124,0), thickness=10)

        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]/2)
        # x coordinate where the histogram has the highest value in the left half of the image
        leftx_base = np.argmax(histogram[:midpoint])
        # x coordinate where the histogram has the highest value in the right half of the image
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
        #print("rightx_base", rightx_base)

        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        self.nonzeroy = np.array(nonzero[0])
        self.nonzerox = np.array(nonzero[1])

        # Current positions to be updated for each window
        self.leftx_current = leftx_base
        self.rightx_current = rightx_base

    def detectLanes(self):

        wf = WindowFactory(self.nwindows, self.binary_warped)

        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(0, self.nwindows):

            window_left = wf.getWindowLeft(window, self.leftx_current)
            window_right = wf.getWindowRight(window, self.rightx_current)

            # Draw the windows on the visualization image
            if self.interactiveMode:

                cv2.rectangle(self.out_img,window_left.getTopLeft(),window_left.getBottomRight(),(0,255,0), 2)
                cv2.rectangle(self.out_img,window_right.getTopLeft(),window_right.getBottomRight(),(0,255,0), 2)

            # Identify the nonzero pixels in x and y within the window
            good_left_inds = window_left.getNonZeroPixels()
            good_right_inds = window_right.getNonZeroPixels()

            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > self.minpix:
                self.leftx_current = window_left.getMeanXofNonZeroPixels()
            if len(good_right_inds) > self.minpix:
                self.rightx_current = window_right.getMeanXofNonZeroPixels()


            self.leftx_current


            # Append these indices to the lists
            self.left_lane_inds.append(good_left_inds)
            self.right_lane_inds.append(good_right_inds)


        # Concatenate the arrays of indices
        self.left_lane_inds = np.concatenate(self.left_lane_inds)
        self.right_lane_inds = np.concatenate(self.right_lane_inds)

        x_coords_left_lane = self.nonzerox[self.left_lane_inds]
        y_coords_left_lane = self.nonzeroy[self.left_lane_inds]
        x_coords_right_lane = self.nonzerox[self.right_lane_inds]
        y_coords_right_lane = self.nonzeroy[self.right_lane_inds]

        #print("x_coords_left_lane.shape", x_coords_left_lane.shape)
        for lane in self.lastLeftLanes:
            #print("left lane.getXCoords.shape", lane.getXCoords().shape)
            x_coords_left_lane = np.hstack((x_coords_left_lane, lane.getXCoords()))
            y_coords_left_lane = np.hstack((y_coords_left_lane, lane.getYCoords()))
        #print("x_coords_left_lane.shape", x_coords_left_lane.shape)

        for lane in self.lastRightLanes:
            x_coords_right_lane = np.hstack((x_coords_right_lane, lane.getXCoords()))
            y_coords_right_lane = np.hstack((y_coords_right_lane, lane.getYCoords()))

        #print(x_coords_right_lane.shape)

        self.left_lane = Lane(x_coords_left_lane, y_coords_left_lane, self.binary_warped.shape[0])
        self.right_lane = Lane(x_coords_right_lane, y_coords_right_lane, self.binary_warped.shape[0])

        return (self.left_lane, self.right_lane)

    def getLeftLane(self):
        return self.left_lane

    def getRightLane(self):
        return self.right_lane

    def getOutImage(self):
        return self.out_img
