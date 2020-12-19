import cv2
import matplotlib.pyplot as plt
import copy
import numpy as np

from src import model
from src import util
from src.body import Body

body_estimation = Body('model/body_pose_model.pth')

capture = cv2.VideoCapture('videos/men85kg_0017.mp4')

# Default resolutions of the frame are obtained.The default resolutions are system dependent.
# We convert the resolutions from float to integer.
frame_width = int(capture.get(3))
frame_height = int(capture.get(4))

# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
out = cv2.VideoWriter('results/output.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))

while True:
    isTrue, frame = capture.read()

    if isTrue == True:
        print('Processing...')
        candidate, subset = body_estimation(frame)
        canvas = copy.deepcopy(frame)
        canvas = util.draw_bodypose(canvas, candidate, subset)

        # Write the frame into the file 'output.avi'
        out.write(canvas)

        # Display the resulting frame    
        cv2.imshow('Preview',canvas)

        # Press Q on keyboard to stop recording
        if cv2.waitKey(1) & 0xFF == ord('d'):
          break

    # Break the loop
    else:
      break

capture.release()
out.release()

cv2.destroyAllWindows()

print('Done')