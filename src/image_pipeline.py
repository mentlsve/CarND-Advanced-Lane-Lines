class ImagePipeline:

    def __init__(self, camera):
        self.camera = camera
        self.lastLeftLanes = deque()
        self.lastRightLanes = deque()

    def run(self, image_RGB):

        # moviepy reads in the image as RGB, but the pipeline is based on BGR
        image_BGR = cv2.cvtColor(image_RGB, cv2.COLOR_RGB2BGR)

        # 1. undistort the image
        undistorted = self.camera.undistort(image_BGR)

        # 2. perspective transformation
        perspective = PerspectiveTransform()
        warped = perspective.toBirdsEyeView(undistorted)

        # 3. find potential lane pixels
        ps = PixelSelection()
        binary_warped = ps.detectLanePixels(warped)

        # 4. derive the lanes
        laneFinder = LaneFinder(binary_warped,
                                15,
                                self.lastLeftLanes,
                                self.lastRightLanes,
                                interactiveMode = False)
        left_lane, right_lane = laneFinder.detectLanes()


        self.lastLeftLanes.append(left_lane)
        self.lastRightLanes.append(right_lane)
        if(len(self.lastLeftLanes) > 5):
            self.lastLeftLanes.popleft()
            self.lastRightLanes.popleft()

        # 5. create the overlay
        oh = OverlayHelper(left_lane, right_lane)
        street_overlay_birds_eye_view = oh.createOverlay(warped)
        street_overlay_car_camera_view = pt.toCarCameraView(street_overlay_birds_eye_view)

        # 6. combine the overlay
        combined = oh.addOverlay(street_overlay_car_camera_view, undistorted)

        # 7. write the metadata on the image
        oh.addMetadata(combined, XM_PER_PIX)

        output = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)

        return output
