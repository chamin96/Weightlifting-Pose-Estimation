import cv2
import matplotlib.pyplot as plt
import copy
import numpy as np
import csv

from src import model
from src import util
from src.body import Body
from scoring_model import scoring

import time

body_estimation = Body('model/body_pose_model.pth')

input_filename = 'men85kg_0012'
capture = cv2.VideoCapture("85kg_men_test/{}.mp4".format(input_filename))

# Default resolutions of the frame are obtained.The default resolutions are system dependent.
# We convert the resolutions from float to integer.
frame_width = int(capture.get(3))
frame_height = int(capture.get(4))

output_filename = "{}-output".format(input_filename)

# Define the codec and create VideoWriter object.
out = cv2.VideoWriter("results/{}.avi".format(output_filename), cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))

# Creating a dictionary from the output.csv file
reader = csv.reader(open('85kg_men_results/{}/output.csv'.format(input_filename), 'r'))
dict_csv = {}
next(reader) # skip header
for row in reader:
  k, v = row
  dict_csv[int(k)] = v

frame_count = 0
start = time.time()

kneeAngles = []
barPositions = []
barAngles = []

while True:
    isTrue, frame = capture.read()

    if isTrue == True:
        
        print('Processing %dth frame...' %frame_count)

        candidate, subset = body_estimation(frame)
        canvas = copy.deepcopy(frame)
        canvas = util.draw_bodypose(canvas, candidate, subset)

        # util.toTempList(candidate)
        coordinates = np.delete(candidate, (2,3), axis=1)
        
        # angle calculation
        action = dict_csv.get(frame_count)
        print('Action:', action)
        # calculate bar position of the frame
        if len(coordinates) > 15:
          barPositions.append(util.calcBarPosition(coordinates[4][0], coordinates[4][1], coordinates[7][0], coordinates[7][1]))

          # calculate bar angle of the frame
          barAngles.append(util.calcBarAngle(coordinates[4][0], coordinates[4][1], coordinates[7][0], coordinates[7][1]))

          # calculate initial knee angle
          if action == 'setupsnatch' or action == 'setupclean':
            kneeAngles.append(util.calcInitialKneeAngle(coordinates[11], coordinates[12], coordinates[13]))

        # Write the frame into the file 'output.avi'
        out.write(canvas)

        # Display the resulting frame    
        cv2.imshow('Preview',canvas)

        # Press Q on keyboard to stop recording
        if cv2.waitKey(1) & 0xFF == ord('d'):
          break

        frame_count += 1

    # Break the loop
    else:
      break

# calculate average initial knee angle
initialKneeAngle = sum(kneeAngles) / len(kneeAngles)

# calculate average of bar angles
avgBarAngle = sum(barAngles) / len(barAngles)

print('Initial Knee Angle(Avg):', initialKneeAngle)
# print('Bar Position', barPositions)
print('Average Bar Angle:', avgBarAngle)

print('------ SCORE REPORT ------')
knee_score = scoring.kneeAngleScore(initialKneeAngle)
bar_score = scoring.barAngleScore(avgBarAngle)
overall_score = scoring.overallScore(knee_score, bar_score)

print('Knee Angle Score:', knee_score)
print('Bar Angle Score:', bar_score)
print('Overall Score:', overall_score)

capture.release()
out.release()

cv2.destroyAllWindows()

end = time.time()
print("Elapsed time: {} seconds".format(end - start))

# util.toDataframe()
print('Done')
