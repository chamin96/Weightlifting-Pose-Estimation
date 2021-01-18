import numpy as np
import math
import cv2
import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd

# initialize an empty dictionary to store absolute angles of body parts
absolute_angles = {}
relative_angles = {}

def padRightDownCorner(img, stride, padValue):
    h = img.shape[0]
    w = img.shape[1]

    pad = 4 * [None]
    pad[0] = 0 # up
    pad[1] = 0 # left
    pad[2] = 0 if (h % stride == 0) else stride - (h % stride) # down
    pad[3] = 0 if (w % stride == 0) else stride - (w % stride) # right

    img_padded = img
    pad_up = np.tile(img_padded[0:1, :, :]*0 + padValue, (pad[0], 1, 1))
    img_padded = np.concatenate((pad_up, img_padded), axis=0)
    pad_left = np.tile(img_padded[:, 0:1, :]*0 + padValue, (1, pad[1], 1))
    img_padded = np.concatenate((pad_left, img_padded), axis=1)
    pad_down = np.tile(img_padded[-2:-1, :, :]*0 + padValue, (pad[2], 1, 1))
    img_padded = np.concatenate((img_padded, pad_down), axis=0)
    pad_right = np.tile(img_padded[:, -2:-1, :]*0 + padValue, (1, pad[3], 1))
    img_padded = np.concatenate((img_padded, pad_right), axis=1)

    return img_padded, pad

# transfer caffe model to pytorch which will match the layer name
def transfer(model, model_weights):
    transfered_model_weights = {}
    for weights_name in model.state_dict().keys():
        transfered_model_weights[weights_name] = model_weights['.'.join(weights_name.split('.')[1:])]
    return transfered_model_weights

# draw the body keypoint and lims
def draw_bodypose(canvas, candidate, subset):
    stickwidth = 4
    limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], \
               [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], \
               [1, 16], [16, 18], [3, 17], [6, 18]]

    bodyParts = ['lShoulder', 'rShoulder', 'lArm', 'lForearm', 'rArm', 'rForearm', 'lTrunk', 'lThigh', \
                 'lLeg', 'rTrunk', 'rThigh', 'rLeg', 'neck', 'lCheeck', 'lEye', \
                 'rCheeck', 'rEye', '', '']

    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

    for i in range(18):
        for n in range(len(subset)):
            index = int(subset[n][i])
            if index == -1:
                continue
            x, y = candidate[index][0:2]
            cv2.circle(canvas, (int(x), int(y)), 4, colors[i], thickness=-1)
    for i in range(17):
        for n in range(len(subset)):
            index = subset[n][np.array(limbSeq[i]) - 1]
            if -1 in index:
                continue
            cur_canvas = canvas.copy()
            Y = candidate[index.astype(int), 0]
            X = candidate[index.astype(int), 1]
            mX = np.mean(X)
            mY = np.mean(Y)
            # print body parts lables on the image
            cv2.putText(cur_canvas, bodyParts[i], (int(mY),int(mX)), cv2.FONT_HERSHEY_TRIPLEX, 1.0, colors[i], thickness=2)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            # append absolute angle and relevant body part to the dictionary
            absolute_angles.update({bodyParts[i]: angle})

            # calculate and save relative angle
            # TODO: need to optimize the conditions
            if angle >= 0 and angle <= 90:
                relative_angles.update({bodyParts[i]: (180.0 - angle)})
            elif angle >= 90 and angle <= 180:
                relative_angles.update({bodyParts[i]: (180.0 - angle)})
            elif angle >= -90 and angle <= 0:
                relative_angles.update({bodyParts[i]: (180.0 + abs(angle))})
            elif angle >= -180 and angle <= -90:
                relative_angles.update({bodyParts[i]: (180.0 + abs(angle))})
            
            polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(cur_canvas, polygon, colors[i])
            canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)
    print(absolute_angles)
    print(relative_angles)
    print("Right Knee Angle = ", abs(abs(360.0 - relative_angles['rThigh']) - abs(180.0 - relative_angles['rLeg'])))
    print("Left Knee Angle = ", abs(abs(180.0 - relative_angles['lThigh']) - abs(360.0 - relative_angles['lLeg'])))
    # plt.imsave("preview.jpg", canvas[:, :, [2, 1, 0]])
    # plt.imshow(canvas[:, :, [2, 1, 0]])
    return canvas

def draw_handpose(canvas, all_hand_peaks, show_number=False):
    edges = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10], \
             [10, 11], [11, 12], [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]]
    fig = Figure(figsize=plt.figaspect(canvas))

    fig.subplots_adjust(0, 0, 1, 1)
    fig.subplots_adjust(bottom=0, top=1, left=0, right=1)
    bg = FigureCanvas(fig)
    ax = fig.subplots()
    ax.axis('off')
    ax.imshow(canvas)

    width, height = ax.figure.get_size_inches() * ax.figure.get_dpi()

    for peaks in all_hand_peaks:
        for ie, e in enumerate(edges):
            if np.sum(np.all(peaks[e], axis=1)==0)==0:
                x1, y1 = peaks[e[0]]
                x2, y2 = peaks[e[1]]
                ax.plot([x1, x2], [y1, y2], color=matplotlib.colors.hsv_to_rgb([ie/float(len(edges)), 1.0, 1.0]))

        for i, keyponit in enumerate(peaks):
            x, y = keyponit
            ax.plot(x, y, 'r.')
            if show_number:
                ax.text(x, y, str(i))
    bg.draw()
    canvas = np.fromstring(bg.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
    return canvas

# image drawed by opencv is not good.
def draw_handpose_by_opencv(canvas, peaks, show_number=False):
    edges = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10], \
             [10, 11], [11, 12], [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]]
    # cv2.rectangle(canvas, (x, y), (x+w, y+w), (0, 255, 0), 2, lineType=cv2.LINE_AA)
    # cv2.putText(canvas, 'left' if is_left else 'right', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    for ie, e in enumerate(edges):
        if np.sum(np.all(peaks[e], axis=1)==0)==0:
            x1, y1 = peaks[e[0]]
            x2, y2 = peaks[e[1]]
            cv2.line(canvas, (x1, y1), (x2, y2), matplotlib.colors.hsv_to_rgb([ie/float(len(edges)), 1.0, 1.0])*255, thickness=2)

    for i, keyponit in enumerate(peaks):
        x, y = keyponit
        cv2.circle(canvas, (x, y), 4, (0, 0, 255), thickness=-1)
        if show_number:
            cv2.putText(canvas, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), lineType=cv2.LINE_AA)
    return canvas

# detect hand according to body pose keypoints
# please refer to https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/src/openpose/hand/handDetector.cpp
def handDetect(candidate, subset, oriImg):
    # right hand: wrist 4, elbow 3, shoulder 2
    # left hand: wrist 7, elbow 6, shoulder 5
    ratioWristElbow = 0.33
    detect_result = []
    image_height, image_width = oriImg.shape[0:2]
    for person in subset.astype(int):
        # if any of three not detected
        has_left = np.sum(person[[5, 6, 7]] == -1) == 0
        has_right = np.sum(person[[2, 3, 4]] == -1) == 0
        if not (has_left or has_right):
            continue
        hands = []
        #left hand
        if has_left:
            left_shoulder_index, left_elbow_index, left_wrist_index = person[[5, 6, 7]]
            x1, y1 = candidate[left_shoulder_index][:2]
            x2, y2 = candidate[left_elbow_index][:2]
            x3, y3 = candidate[left_wrist_index][:2]
            hands.append([x1, y1, x2, y2, x3, y3, True])
        # right hand
        if has_right:
            right_shoulder_index, right_elbow_index, right_wrist_index = person[[2, 3, 4]]
            x1, y1 = candidate[right_shoulder_index][:2]
            x2, y2 = candidate[right_elbow_index][:2]
            x3, y3 = candidate[right_wrist_index][:2]
            hands.append([x1, y1, x2, y2, x3, y3, False])

        for x1, y1, x2, y2, x3, y3, is_left in hands:
            # pos_hand = pos_wrist + ratio * (pos_wrist - pos_elbox) = (1 + ratio) * pos_wrist - ratio * pos_elbox
            # handRectangle.x = posePtr[wrist*3] + ratioWristElbow * (posePtr[wrist*3] - posePtr[elbow*3]);
            # handRectangle.y = posePtr[wrist*3+1] + ratioWristElbow * (posePtr[wrist*3+1] - posePtr[elbow*3+1]);
            # const auto distanceWristElbow = getDistance(poseKeypoints, person, wrist, elbow);
            # const auto distanceElbowShoulder = getDistance(poseKeypoints, person, elbow, shoulder);
            # handRectangle.width = 1.5f * fastMax(distanceWristElbow, 0.9f * distanceElbowShoulder);
            x = x3 + ratioWristElbow * (x3 - x2)
            y = y3 + ratioWristElbow * (y3 - y2)
            distanceWristElbow = math.sqrt((x3 - x2) ** 2 + (y3 - y2) ** 2)
            distanceElbowShoulder = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            width = 1.5 * max(distanceWristElbow, 0.9 * distanceElbowShoulder)
            # x-y refers to the center --> offset to topLeft point
            # handRectangle.x -= handRectangle.width / 2.f;
            # handRectangle.y -= handRectangle.height / 2.f;
            x -= width / 2
            y -= width / 2  # width = height
            # overflow the image
            if x < 0: x = 0
            if y < 0: y = 0
            width1 = width
            width2 = width
            if x + width > image_width: width1 = image_width - x
            if y + width > image_height: width2 = image_height - y
            width = min(width1, width2)
            # the max hand box value is 20 pixels
            if width >= 20:
                detect_result.append([int(x), int(y), int(width), is_left])

    '''
    return value: [[x, y, w, True if left hand else False]].
    width=height since the network require squared input.
    x, y is the coordinate of top left 
    '''
    return detect_result

# get max index of 2d array
def npmax(array):
    arrayindex = array.argmax(1)
    arrayvalue = array.max(1)
    i = arrayvalue.argmax()
    j = arrayindex[i]
    return i, j

# column headers for dataframe
columns = ["x0", "y0", "x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4", "x5", "y5",
			 "x6", "y6", "x7", "y7", "x8", "y8", "x9", "y9", "x10", "y10", "x11", "y11",
			 "x12", "y12", "x13", "y13", "x14", "y14", "x15", "y15", "x16", "y16", "x17", "y17"]


# temporary list to store x,y coordinate values
temp_list = []

def toTempList(array):
    #slice the array and save only x,y coordinates(0,1 columns)
    if len(array[:,[0,1]]) == 18:
        temp_list.append(array[:,[0,1]].flatten().tolist())
    else:
        temp_list.append([0]*36)

def toDataframe():
    # initialize an empty dataframe with column headers and pass temp_list
    df = pd.DataFrame(data=temp_list, columns=columns)
    df.to_csv('output/joints.csv')
    print(df)

# returns the initial knee angle
# considers the joints 11, 12, and 13
# needs a side view
def calcInitialKneeAngle (x11, y11, x12, y12, x13, y13):
    angle = math.degrees (math.atan2 ((y13 - y12), (x13 - x12)) - math.atan2 ((y11 - y12), (x11 - x12)))
    return angle + 360 if angle < 0 else angle

# returns the angle of the weightlifting bar with the x axis
# considers the joints 4 and 7
# needs a front view
def calcBarAngle (x4, y4, x7, y7): 
    return math.atan2 ((y7 - y4), (x7 - x4))

# returns the thigh angle with x axis in split jerk stance
# considers the joints 11 and 12
# needs a side view
# should be 20-40 degrees
def calcSplitJerkThighAngle(x11, y11, x12, y12):
    return math.atan2((y11-y12), (x11-x12))

# returns weightlifting bar position
def calcBarPosition(x4, y4, x7, y7):
    return (x4+x7)/2, (y4+y7)/2

# returns the velocity of the bar in x and y directions
# takes bar positions of 2 adjacent frames and fps value as inputs 
def calcBarVelocity (x1, y1, x2, y2, fps):
    dx = (x2 - x1) / fps
    dy = (y2 - y1) / fps
    # return math.sqrt (dx^2 + dy^2)
    return dx, dy