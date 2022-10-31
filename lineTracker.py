#   IMPORTS 

import os
import time
import keyboard
import numpy as np
import cv2
from scipy.ndimage.filters import gaussian_filter


#   GLOBAL VARIABLES / DEFINIES

FILE_NAME = "example.mp4" # 
BASE_PATH = os.path.dirname(os.path.realpath(__file__))
FILE_PATH = os.path.join(BASE_PATH, FILE_NAME)

#size of the image coming from the cam
CAM_W = 800
CAM_H = 450

#size of the birds eye view image
BEV_W = 256
BEV_H = 512

FPS = 29.97

LANE_START_W = 0.85
LANE_START_H = 0.25
LANE_START_Y = 0.75

LANE_SEARCH_MAX_LANES = 6
LANE_SEARCH_LANE_SEGMENTS = 32
LANE_SEARCH_WIDTH = 24
LANE_SEARCH_HEIGHT = 12



yStart = int(BEV_H * LANE_START_Y)
yEnd = int(BEV_H * (LANE_START_Y + LANE_START_H))

xCenter = BEV_W * 0.5
xWidth = BEV_W * LANE_START_W
xStart = int(xCenter - (xWidth * 0.5))
xEnd = int(xCenter + (xWidth * 0.5))


running = True

#   GLOBAL FUNCTIONS
def onkeypress(event):
    global stop
    if event.name == 'esc':
        global running
        running = False

def loadImage(filepath):
    return np.asarray_chkfinite(cv2.imread(BASE_PATH + "/" + FILE_NAME, cv2.IMREAD_COLOR), "uint8")

def show(img, windowName):
    cv2.imshow(windowName, img)

def waitForKeyPress(sec):
    cv2.waitKey(sec * 1000)
      
def adaptiveThesholdLightness(value):
    return cv2.adaptiveThreshold(value, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 55, -25)
    
def thresholdLightness(value):
    return cv2.inRange(value, 170, 255) 

def thresholdNegSat(sat):
    return cv2.inRange(sat, 0, 30) 
    
def whiteLineThresholding(hls):
    lightnessLocal = adaptiveThesholdLightness(hls[:,:,1])
    lightnessGlobal = thresholdLightness(hls[:,:,1])
      
    negativeSat = thresholdNegSat(hls[:,:,2])
    negativeSat = cv2.GaussianBlur(negativeSat, (55,55), 0)
    negativeSat = cv2.inRange(negativeSat, 240, 255)
    
    return cv2.bitwise_and(cv2.bitwise_and(lightnessLocal, lightnessGlobal), negativeSat)

def yellowLineThresholding(hls):
    hlsMin = (25, 100, 25)   
    hlsMax = (75, 255, 255)    
    yellow =  cv2.inRange(hls, hlsMin, hlsMax) 
    return yellow
    
def getLineStartRoi(laneMarkings):
    return laneMarkings[yStart:yEnd, xStart:xEnd]
    
def samplePolynomial2Deg(coefficients, start, end):   

    
    pointsY = []
    for y in range (start, end):
        pointsY.append(y)
        

    pointsX = np.asarray_chkfinite(np.polyval(coefficients, pointsY), "uint8")
    pointsY = np.asarray_chkfinite(pointsY, "uint8")
    points = np.transpose(np.vstack((pointsY, pointsX)))
    #print(type(points), np.shape(points))
    return points
        

    
    
def findLines(lineMask, laneStartPositions):
    
    starts = np.asarray_chkfinite(laneStartPositions, "uint8")
    
    fixedStarts = (starts + xStart)[0]
    
    rgbMask = cv2.cvtColor(lineMask, cv2.COLOR_GRAY2RGB)
    
    halfWidth = LANE_SEARCH_WIDTH * 0.5
    
    for startX in fixedStarts:
        cv2.circle(rgbMask, (startX, BEV_H), 4, (0, 0, 255), -1, lineType = cv2.LINE_AA)  
    
    
    lineCount = len(fixedStarts)
    
    lines = []
    
    listOfLinesFinal = []
    for l in range(0, min(lineCount, LANE_SEARCH_MAX_LANES)):
        listOfLinesFinal.append([])
        
    lineSignals = []
    for l in range(0, min(lineCount, LANE_SEARCH_MAX_LANES)):
        lineSignals.append(0.0)    
    
    for i in range(0, LANE_SEARCH_LANE_SEGMENTS):
        trackedPoseY = int(BEV_H - (i * LANE_SEARCH_HEIGHT))
        boxCenterY = trackedPoseY - (LANE_SEARCH_HEIGHT * 0.5)
        
        segments = []         
        
        
        for l in range(0, min(lineCount, LANE_SEARCH_MAX_LANES)):
            
            if i == 0:     
                trackedPoseX = fixedStarts[l]
            else:
                trackedPoseX = lines[i - 1][l]
       
            left = int(trackedPoseX - halfWidth)
            right = left + LANE_SEARCH_WIDTH
            
            top = trackedPoseY - LANE_SEARCH_HEIGHT
            bottom = trackedPoseY
            
            cv2.rectangle(rgbMask, (left, top), (right, bottom), (255, 0, 0), 1)
            
            rectRoi = lineMask[top:bottom, left:right]
            (y, x) = np.where(rectRoi > 0)     
            pointCount = len(x)
            
            signal = 0.0
            xMean = halfWidth
                   
            if pointCount > 0:       
                signal += pointCount / (LANE_SEARCH_WIDTH * LANE_SEARCH_HEIGHT)        
                xMean = np.mean(x)
            
            finalX = left + xMean
            
            #print(signal, xMean)
            
        
            segments.append(finalX)
            
            if (signal > 0.01):
                listOfLinesFinal[l].append(np.asarray_chkfinite((finalX, boxCenterY), "float64"))
                lineSignals[l] += 1.0
        
        lines.append(segments)
        
        
        
        

    # fitting polynomial
        
    listOfLineCoeffs = []       
    for line in listOfLinesFinal:
        #line = listOfLinesFinal[0]
        #print(line)
        
        if len(line) > 3:

            xes = []
            yes = []
            for point in line:
                cv2.circle(rgbMask, (int(point[0]), int(point[1])), 2, (255, 0, 255), -1, lineType = cv2.LINE_AA)  
                xes.append(point[0])
                yes.append(point[1])
            
            xes = np.asarray_chkfinite(xes, "float64")
            yes = np.asarray_chkfinite(yes, "float64")
            
            turnedXes = BEV_H - yes
            turnedYes = BEV_W - xes
            
            lineCoeffs = np.polyfit(turnedXes, turnedYes, 2)
            #lineCoeffs[2] = 0
            listOfLineCoeffs.append(lineCoeffs)
            #print(lineCoeffs)
                 
        
    # averaging polynomial
    avgA = 0.0
    avgB = 0.0   
    
    summedWeight = 0.0   
    #print(lineSignals)
    
    for l in range(0, len(listOfLineCoeffs)):
        line = listOfLineCoeffs[l]
        a = line[0]
        b = line[1]
        
        weight = lineSignals[l]
        
        avgA += a * weight
        avgB += b * weight
        summedWeight += weight

    
    # correct low signal lines based on average
    if summedWeight > 1.0:    
        avgA /= summedWeight
        avgB /= summedWeight
        
        for l in range(0, len(listOfLineCoeffs)):
            line = listOfLineCoeffs[l] 
            weight = (lineSignals[l] / summedWeight)
            # #print(weightB)     
            
            blendedA = (line[0] * weight) + (avgA * (1.0 - weight))
            blendedB = (line[1] * weight) + (avgB * (1.0 - weight))
            
            
            line[0] = blendedA
            line[1] = blendedB
            
            
            
    yValues = np.arange(0, BEV_H, 4, "float64")        
    for line in listOfLineCoeffs:       

        xValues = np.polyval(line, yValues)      
        for i in range(0, len(yValues)):
            x = BEV_W - xValues[i]
            y = BEV_H - yValues[i]
            cv2.circle(rgbMask, (int(x), int(y)), 1, (0, 255, 0), -1, lineType = cv2.LINE_AA)  
        
            

     
     
           # fixedYs = line
            #fixedXs = line[:][0]   
            #print(np.shape(fixedYs))


            #lineCoeffs = np.polyfit(fixedYs, fixedXs, 2)
            #yValues = np.arange(0, BEV_H, 4, "uint8")
            
            
            #np.asarray_chkfinite(np.polyval(lineCoeffs, yValues), "uint8")
            
            
            
            
            
    # lanesAsCoeff = []                  
    # for line in listOfLinesFinal:
    #     xList = []
    #     yList = []
    #     for (x, y) in line:
    #         xList.append(x) 
    #         yList.append(y)   
    #         cv2.circle(rgbMask, (int(x), int(y)), 2, (255, 0, 255), -1, lineType = cv2.LINE_AA)  
           
    #     if len(xList) and len(yList) and (len(xList) == len(yList)) and len(xList) > 3:
    #         lanesAsCoeff.append(np.polyfit(yList, xList, 2))
            
            
        
    # lanesAsHighResPoints = []
    # for line in lanesAsCoeff:
    #     lanePoints = samplePolynomial2Deg(line, 0, BEV_H)
    #     #print(np.shape(lanePoints))
    #     lanesAsHighResPoints.append(lanePoints)
        
    # for line in lanesAsHighResPoints:
    #     #print(type(line), np.shape(line))
        
    #     length = np.shape(line)[0]
        
    #     for i in range(0, length):
    #         point = line[i]   
    #         cv2.circle(rgbMask, (int(point[1]), int(point[0])), 1, (0, 255, 0), -1)
            
        
    
    show(rgbMask, "rgbmaskktest")
    
  
    
#   CLASSES


# the BirdsEyeView class allows us to warp the image from the cam to an image from a top down perspective
class BirdsEyeView:
    def __init__(self, camSize, bevSize, interp):
        self.sizeSource = camSize
        self.sizeTarget = bevSize
        self.interp = interp
        centerX = self.sizeSource[1] * 0.5
        
        fWidth = 0.5
        
        
        top = self.sizeSource[0] * 0.63
        bottom = self.sizeSource[0] * 0.9
        
        halfWidthTop = self.sizeSource[1] * 0.2 * fWidth
        halfWidthBottom = self.sizeSource[1] * 2.2 * fWidth
        
        
        targetTop = 0.0
        targetBottom = 1.1
        
        self.rectSource = np.float32([[centerX - halfWidthTop, top], [centerX + halfWidthTop, top], 
                        [centerX - halfWidthBottom, bottom], [centerX + halfWidthBottom, bottom]])

        self.rectTarget = np.float32([[0, self.sizeTarget[1] * targetTop],[self.sizeTarget[0], self.sizeTarget[1] * targetTop],
                                      [0, self.sizeTarget[1] * targetBottom], [self.sizeTarget[0], self.sizeTarget[1] * targetBottom]])

        self.matrixScreenToBird = cv2.getPerspectiveTransform(self.rectSource, self.rectTarget)
        self.matrixBirdToScreen = cv2.getPerspectiveTransform(self.rectTarget, self.rectSource)


    def getBirdFromCam(self, screen):
        return cv2.warpPerspective(screen, self.matrixScreenToBird, (self.sizeTarget[0], self.sizeTarget[1]), flags = self.interp, borderMode = cv2.BORDER_DEFAULT)

    def getCamFrombird(self, screen):
        return cv2.warpPerspective(screen, self.matrixBirdToScreen, (self.sizeSource[1], self.sizeSource[0]), flags = self.interp, borderMode = cv2.BORDER_DEFAULT)   





#   SCRIPT

keyboard.on_press(onkeypress)

while running:

    video = cv2.VideoCapture(FILE_PATH)
    delayAfterFrame = int(1000 / FPS)

    
    expHistoryOfLaneStarts = None


    ret, img = video.read()
    while running and ret == True:
        
        # save current time
        start = time.time()
        
        # resize image to a defined size
        img = cv2.resize(img, (CAM_W, CAM_H), interpolation = cv2.INTER_LANCZOS4) # INTER_NEAREST, INTER_LINEAR, INTER_CUBIC, INTER_LANCZOS4
                
        # show rgb iamge from cam
        show(img, "0 - original")

        # the BirdsEyeView class allows us to warp the image from the camera view to an image from a top down perspective
        bevTransformer = BirdsEyeView((CAM_H, CAM_W), (BEV_W, BEV_H), cv2.INTER_CUBIC)  # INTER_NEAREST, INTER_LINEAR, INTER_CUBIC, INTER_LANCZOS4

        # transform the image
        bevImg = bevTransformer.getBirdFromCam(img)
        # blur it
        bevImg = cv2.GaussianBlur(bevImg, (1,3), 0)
        # show the image
        show(bevImg, "1 - bevImg")

        # convert the top down rgb image to hls color space and show it
        hls = cv2.cvtColor(bevImg, cv2.COLOR_BGR2HLS_FULL)
        
        # show that
        show(hls, "2 - hls")
        #show(hls[:,:,0], "2a - Hue")
        show(hls[:,:,1], "2b - Lightness")
        show(hls[:,:,2], "2c - Saturation")
        
        # get white lines via thresholding
        whiteLine = whiteLineThresholding(hls)
        #show(whiteLine, "3a - whiteLine")
        
        # get white yellow via thresholding
        yellowLine = yellowLineThresholding(hls)
        #show(yellowLine, "3b - yellowLine")
        
        # combine both masks
        bothLines = cv2.bitwise_or(whiteLine, yellowLine)
        show(bothLines, "3 - bothLines")
        
        # select a region at the bottom
        laneStartRoi = getLineStartRoi(bothLines)
        #show(laneStartRoi, "4a - laneStartRoi")
        
        # take average of each colum and blur it
        laneStartRoiReduced = np.average(np.asarray_chkfinite(laneStartRoi, "float32" ) / 255.0, 0) 
        laneStartRoiBlur = gaussian_filter(laneStartRoiReduced, sigma = 9)
        
        # show that
        laneStartRoiReducedStacked = np.tile(laneStartRoiBlur, (100, 1))
        #show(laneStartRoiReducedStacked, "4b - laneStartRoiReduced")
        
        #Temporal blending of the blured lane start region
        # temporal blend factor between 0.1 and 0.2. these are magic numbers that work well
        weight = (laneStartRoiBlur * 0.1) + 0.05
        
        # on first time set the history to the current detection 
        if type(expHistoryOfLaneStarts) is not np.ndarray:
            expHistoryOfLaneStarts = laneStartRoiBlur
        # after that, the current detection is blended with the final result of the last one
        else:
            expHistoryOfLaneStarts = ((1.0 - weight) * expHistoryOfLaneStarts) + (weight * laneStartRoiBlur)   
            
        # show that
        expHistoryOfLaneStartsStacked = np.tile(expHistoryOfLaneStarts, (100, 1))
        #show(expHistoryOfLaneStartsStacked, "4b - expHistoryOfLaneStarts")
        
        # get mask of local maxima on the temporarily blended lane start detection
        peaks = np.diff(np.sign(np.diff(-expHistoryOfLaneStarts)))
        
        # show that
        peaksStacked = np.tile(peaks, (100, 1))
        show(peaksStacked, "4 - peaks")
        
        result = np.where(peaks > 0)
        #print(result[0], sep='\n')
        
        lanePolys = findLines(bothLines, result)
        
         
        # sobely = sobelPlusYellow(hls[:,:,2])
        # sobely = cv2.convertScaleAbs(sobely)
        # show(sobely, "3b - sobel + yellow")

        # # # apply thresholding to get pixel mask that only shows pixels that might be part of a white line and show that maskk
        
        # whiteLineThresholded = whiteLineThresholding(hls)      
        # show(whiteLineThresholded, "3c - whiteLineThresholded")


        # # whiteLines = cv2.bitwise_and(whiteLineThresholded, valueThreshed)
        # # show(whiteLines, "4a - whiteLines")
        
    
        # how long it took to process this frame
        delta = int((time.time() - start) * 1000)
        
        # calculate time left to sleep before we wanna process the next frame
        sleepTime = max(1, delayAfterFrame - delta)  
        
        # sleep and than show
        cv2.waitKey(sleepTime)
        
        # Read next frame
        ret, img = video.read()
           
        
    del video
    cv2.waitKey(1)
    
exit(0)


# waitForKeyPress(25)





