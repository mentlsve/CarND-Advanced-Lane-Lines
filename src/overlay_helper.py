class OverlayHelper:
    
    def __init__(self, left_lane, right_lane):
        self.left_lane = left_lane
        self.right_lane = right_lane
    
    def addOverlay(self, overlay, image_BGR_car_camera_view):
        image = image_BGR_car_camera_view
        combined = np.zeros_like(image)
        cv2.addWeighted(overlay, 1, image, 0.8, 0, combined)
        return combined;
        
    def createOverlay(self, image_BGR_birds_eye_view):
        image = image_BGR_birds_eye_view
        
        overlay = np.zeros_like(image)
        pts_left_from_top_to_bottom = np.transpose(np.vstack([self.left_lane.getXCoords(), self.left_lane.getYCoords()]))
        pts_right_from_top_to_bottom = np.transpose(np.vstack([self.right_lane.getXCoords(), self.right_lane.getYCoords()]))
        
        street_polygon = self.__createPolygon(pts_left_from_top_to_bottom, pts_right_from_top_to_bottom)
        left_lane_polygon = self.__createLanePolygon(pts_left_from_top_to_bottom) 
        right_lane_polygon = self.__createLanePolygon(pts_right_from_top_to_bottom)  
        
        cv2.fillPoly(overlay, [np.int32(street_polygon)], (0,255,255))
        cv2.fillPoly(overlay, [np.int32(right_lane_polygon)], (255,0,255))
        cv2.fillPoly(overlay, [np.int32(left_lane_polygon)], (255,0,255))
                     
        return overlay
    
    def addMetadata(self, img, xm_per_pix):
        center_distance = self.__getPositionRelativeToCenter()
        center_distance = round(center_distance * xm_per_pix, 2)

        if center_distance < 0:
            text = "Car is " + str(center_distance) + " meters left of center"
        elif center_distance > 0:
            text = "Car is " + str(center_distance) + " meters left of center"
        else:
            text = "Car is " + str(center_distance) + " meters in the center"

        cv2.putText(img, text, (100,100), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),3, cv2.LINE_AA)

        text = "Radius of left lane curvature is " + str(round(left_lane.getCurvatureWorldSpace(720), 0))
        cv2.putText(img, text, (100,150), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),3, cv2.LINE_AA)

        text = "Radius of right lane curvature is " + str(round(right_lane.getCurvatureWorldSpace(720), 0))
        cv2.putText(img, text, (100,200), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),3, cv2.LINE_AA)  
    
    def __getPositionRelativeToCenter(self, height = 720, width = 1280):
        left_lane_bottom_x = self.left_lane.getXCoord(height)
        right_lane_bottom_x = self.right_lane.getXCoord(height)
        center_point = (left_lane_bottom_x + right_lane_bottom_x)/2
        center_distance = (width/2 - center_point)
        return center_distance

    def __createPolygon(self, left_pts, right_pts):
        polygon = np.concatenate((left_pts, np.flipud(right_pts)), axis=0)                                                
        return polygon 
  
    def __createLanePolygon(self, pts, thickness = 15):
        
        # polygon left_lane
        y_coords = np.hsplit(pts, 2)[1]
        x_coords = np.hsplit(pts, 2)[0]
        x_coords_minus = x_coords - thickness
        x_coords_plus = x_coords + thickness
        pts_left = np.dstack((x_coords_minus, y_coords))
        pts_right = np.dstack((x_coords_plus, y_coords))
        
        return self.__createPolygon(pts_left, pts_right)