class Lane:
    
    # x = A * y ** 2 + B * y + C
    def __init__(self, xCoords, yCoords, height):
        
        # PIXEL SPACE
        coefficients = np.polyfit(yCoords, xCoords, 2)
        self.A_px = coefficients[0]
        self.B_px = coefficients[1]
        self.C_px = coefficients[2]
        
        # yCoords which has been passed in may have gaps
        self.yCoords = np.linspace(0, height-1, height)
        self.yCoords = self.yCoords.astype(int)
        
        # recalculate xCoords
        self.xCoords = (self.A_px * self.yCoords ** 2) + (self.B_px * self.yCoords) + self.C_px
        self.xCoords = self.xCoords.astype(int)
        
        # WOLD SPACE
        coefficients = np.polyfit(self.yCoords * YM_PER_PIX, self.xCoords * XM_PER_PIX, 2)
        self.A_rw = coefficients[0]
        self.B_rw = coefficients[1]
        self.C_rw = coefficients[2]
        
    def getCoeffs(self):
        return (self.A_px, self.B_px, self.C_px)
        
    def getXCoord(self, yCoord):
        val =  self.A_px * yCoord ** 2 + self.B_px * yCoord + self.C_px
        return val
    
    def getCurvatureWorldSpace(self, yCoord):
        dividend = (1 + (self.A_rw * yCoord + self.B_rw) ** 2) ** 1.5
        divisor = np.absolute(2 * self.A_rw)
        curverad = dividend / divisor
        return curverad
    
    def getCurvature(self, yCoord):
        dividend = (1 + (self.A_px * yCoord + self.B_px) ** 2) ** 1.5
        divisor = np.absolute(2 * self.A_px)
        curverad = dividend / divisor
        return curverad
    
    def getXCoords(self):
        return self.xCoords
    
    def getYCoords(self):
        return self.yCoords