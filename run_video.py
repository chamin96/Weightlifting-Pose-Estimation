"""

Run score model for a single video.

"""
import time
import csv
import copy
import cv2
import numpy as np

from src import util
from src.body import Body
from scoring_model import scoring

body_estimation = Body("model/body_pose_model.pth")

INPUT_FILENAME = "men105kg_0001"
capture = cv2.VideoCapture("olympic-wl/{}.mp4".format(INPUT_FILENAME))

# Default resolutions of the frame are obtained.The default resolutions are system dependent.
# We convert the resolutions from float to integer.
frame_width = int(capture.get(3))
frame_height = int(capture.get(4))
OUTPUT_FILENAME = "{}-output".format(INPUT_FILENAME)

# Define the codec and create VideoWriter object.
out = cv2.VideoWriter(
    "results/{}.avi".format(OUTPUT_FILENAME),
    cv2.VideoWriter_fourcc("M", "J", "P", "G"),
    10,
    (frame_width + int(frame_width / 4), frame_height),
)

# Creating a dictionary from the output.csv file
reader = csv.reader(
    open("olympic-wl-results/{}/output.csv".format(INPUT_FILENAME), "r")
)
dict_csv = {}
next(reader)  # skip header
for row in reader:
    k, v = row
    dict_csv[int(k)] = v

frame_count = 0
start = time.time()

kneeAngles = []
larmAngles = []
rarmAngles = []
lkneeAngles = []
rkneeAngles = []
barPositions = []
barAngles = []

while True:
    isTrue, frame = capture.read()

    if isTrue:

        print("Processing %dth frame..." % frame_count)

        candidate, subset = body_estimation(frame)
        canvas = copy.deepcopy(frame)
        canvas = util.draw_bodypose(canvas, candidate, subset)

        if len(candidate) > 0:
            coordinates = np.delete(candidate, (2, 3), axis=1)

            # angle calculation
            action = dict_csv.get(frame_count)
            print("Action:", action)

            # draw action text
            canvas = util.draw_action(canvas, action)

            # calculate bar position of the frame
            if len(coordinates) > 15:
                barPositions.append(
                    util.calcBarPosition(
                        coordinates[4][0],
                        coordinates[4][1],
                        coordinates[7][0],
                        coordinates[7][1],
                    )
                )

                # calculate bar angle of the frame
                bar_angle = util.calcBarAngle(
                    coordinates[4][0],
                    coordinates[4][1],
                    coordinates[7][0],
                    coordinates[7][1],
                )
                barAngles.append(bar_angle if bar_angle > 0 else 0)

                # draw bar angle
                canvas = util.draw_bar_angle(canvas, bar_angle)

                # calculate initial knee angle
                if action == "setupsnatch" or action == "setupclean":
                    knee_angle = util.calcInitialKneeAngle(
                        coordinates[11], coordinates[12], coordinates[13]
                    )
                    kneeAngles.append(knee_angle if knee_angle > 0 else 0)
                    # draw knee angle
                    canvas = util.draw_knee_angle(canvas, knee_angle)

                # calculate arm/knee angles at the end
                if action == "standsnatch" or action == "recoveryjerk":
                    l_arm_angle = util.calcArmAngle(
                        coordinates[5], coordinates[6], coordinates[7]
                    )
                    larmAngles.append(l_arm_angle if l_arm_angle > 0 else 0)

                    r_arm_angle = util.calcArmAngle(
                        coordinates[2], coordinates[3], coordinates[4]
                    )
                    rarmAngles.append(r_arm_angle if r_arm_angle > 0 else 0)

                    # draw arm angle
                    canvas = util.draw_arm_angle(canvas, l_arm_angle, r_arm_angle)

                    l_knee_angle = util.calcInitialKneeAngle(
                        coordinates[11], coordinates[12], coordinates[13]
                    )
                    lkneeAngles.append(l_knee_angle if l_knee_angle > 0 else 0)

                    r_knee_angle = util.calcInitialKneeAngle(
                        coordinates[8], coordinates[9], coordinates[10]
                    )
                    rkneeAngles.append(r_knee_angle if r_knee_angle > 0 else 0)

                    # draw knee angle
                    canvas = util.draw_knee_angle_end(
                        canvas, l_knee_angle, r_knee_angle
                    )

        # Write the frame into the file 'output.avi'
        out.write(canvas)

        # Display the resulting frame
        cv2.imshow("Preview", canvas)

        # Press Q on keyboard to stop recording
        if cv2.waitKey(1) & 0xFF == ord("d"):
            break

        frame_count += 1

    # Break the loop
    else:
        break

# calculate average initial knee angle
initialKneeAngle = sum(kneeAngles) / len(kneeAngles)

# calculate average of bar angles
avgBarAngle = sum(barAngles) / len(barAngles)

# calculate average of l arm angles
avglArmAngle = sum(larmAngles) / len(larmAngles)

# calculate average of r arm angles
avgrArmAngle = sum(rarmAngles) / len(rarmAngles)

# calculate average of l legs angles
avglLegAngle = sum(lkneeAngles) / len(lkneeAngles)

# calculate average of r legs angles
avgrLegAngle = sum(rkneeAngles) / len(rkneeAngles)

print("Initial Knee Angle(Avg):", initialKneeAngle)
print("Average Bar Angle:", avgBarAngle)
print("Average Left Arm Angle:", avglArmAngle)
print("Average Right Arm Angle:", avgrArmAngle)
print("Average Left Leg Angle:", avglLegAngle)
print("Average Right Leg Angle:", avgrLegAngle)

print("------ SCORE REPORT ------")
knee_score = scoring.kneeAngleScore(initialKneeAngle)
bar_score = scoring.barAngleScore(avgBarAngle)
arms_score = scoring.armsAngleScore(avglArmAngle, avgrArmAngle)
legs_score = scoring.legsAngleScore(avglLegAngle, avgrLegAngle)
overall_score = scoring.overallScore(knee_score, bar_score, legs_score, arms_score)

print("Knee Angle Score:", knee_score)
print("Bar Angle Score:", bar_score)
print("Legs Angle Score:", legs_score)
print("Arms Angle Score:", arms_score)
print("Overall Score:", overall_score)

canvas = util.draw_score_report(
    canvas,
    initialKneeAngle,
    avgBarAngle,
    avglArmAngle,
    avgrArmAngle,
    avglLegAngle,
    avgrLegAngle,
    knee_score,
    bar_score,
    legs_score,
    arms_score,
    overall_score,
)

# Write the frame into the file 'output.avi'
for i in range(100):
    out.write(canvas)
    # Display the resulting frame
    cv2.imshow("Preview", canvas)
    # Press Q on keyboard to stop recording
    if cv2.waitKey(1) & 0xFF == ord("d"):
        break

score_list = []
field_names = [
    "file",
    "kAngle",
    "bAngle",
    "larmAngle",
    "rarmAngle",
    "llegAngle",
    "rlegAngle",
    "legScore",
    "armScore",
    "kScore",
    "bScore",
    "oScore",
]
temp_scores = {
    "file": INPUT_FILENAME,
    "kAngle": initialKneeAngle,
    "bAngle": avgBarAngle,
    "larmAngle": avglArmAngle,
    "rarmAngle": avgrArmAngle,
    "llegAngle": avglLegAngle,
    "rlegAngle": avgrLegAngle,
    "legScore": legs_score,
    "armScore": arms_score,
    "kScore": knee_score,
    "bScore": bar_score,
    "oScore": overall_score,
}
score_list.append(temp_scores.copy())

with open("./score_report.csv", "w") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=field_names)
    writer.writeheader()
    writer.writerows(score_list)

capture.release()
out.release()

cv2.destroyAllWindows()

end = time.time()
print("Elapsed time: {} seconds".format(end - start))

# util.toDataframe()
print("Done")
