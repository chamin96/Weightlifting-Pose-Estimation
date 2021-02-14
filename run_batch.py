"""

Run multiple videos in directory at once.

"""

from os import walk
import copy
import csv
import time
import numpy as np
import cv2

from src import util
from src.body import Body
from scoring_model import scoring

SRC_DIR = "olympic-wl"
CSV_DIR = "./olympic-wl-results"
DST_DIR = "./result_score/{}".format(SRC_DIR)
body_estimation = Body("model/body_pose_model.pth")
score_list = []
field_names = ["file", "kAngle", "bAngle", "kScore", "bScore", "oScore"]

_, _, video_files = next(walk(SRC_DIR))

for file in video_files:
    print("Processing {}".format(file))
    capture = cv2.VideoCapture("{}/{}".format(SRC_DIR, file))
    frame_width = int(capture.get(3))
    frame_height = int(capture.get(4))

    OUTPUT_FILENAME = "{}-output".format(file)

    # Define the codec and create VideoWriter object.
    out = cv2.VideoWriter(
        "results/{}/{}.avi".format(SRC_DIR, OUTPUT_FILENAME),
        cv2.VideoWriter_fourcc("M", "J", "P", "G"),
        10,
        (frame_width, frame_height),
    )

    # Creating a dictionary from the output.csv file
    sub_dir = file.replace(".mp4", "")
    reader = csv.reader(open("{}/{}/output.csv".format(CSV_DIR, sub_dir), "r"))
    dict_csv = {}
    next(reader)  # skip header

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

        if isTrue:
            candidate, subset = body_estimation(frame)
            canvas = copy.deepcopy(frame)
            canvas = util.draw_bodypose(canvas, candidate, subset)

            if len(candidate) > 0:
                coordinates = np.delete(candidate, (2, 3), axis=1)

                # util.toTempList(candidate)
                coordinates = np.delete(candidate, (2, 3), axis=1)

                # angle calculation
                action = dict_csv.get(frame_count)

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
                    barAngles.append(
                        util.calcBarAngle(
                            coordinates[4][0],
                            coordinates[4][1],
                            coordinates[7][0],
                            coordinates[7][1],
                        )
                    )

                    # calculate initial knee angle
                    if action == "setupsnatch" or action == "setupclean":
                        kneeAngles.append(
                            util.calcInitialKneeAngle(
                                coordinates[11], coordinates[12], coordinates[13]
                            )
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
    if len(kneeAngles) > 0:
        avg_knee_angle = sum(kneeAngles) / len(kneeAngles)
    else:
        avg_knee_angle = -1

    # calculate average of bar angles
    if len(barAngles) > 0:
        avg_bar_angle = sum(barAngles) / len(barAngles)
    else:
        avg_bar_angle = -1

    if avg_knee_angle > 0 and avg_bar_angle > 0:
        print("Initial Knee Angle(Avg):", avg_knee_angle)
        print("Average Bar Angle:", avg_bar_angle)

        print("------ SCORE REPORT ------")
        knee_score = scoring.kneeAngleScore(avg_knee_angle)
        bar_score = scoring.barAngleScore(avg_bar_angle)
        overall_score = scoring.overallScore(knee_score, bar_score)

        print("Knee Angle Score:", knee_score)
        print("Bar Angle Score:", bar_score)
        print("Overall Score:", overall_score)

        temp_scores = {
            "file": file,
            "kAngle": avg_knee_angle,
            "bAngle": avg_bar_angle,
            "kScore": knee_score,
            "bScore": bar_score,
            "oScore": overall_score,
        }
        score_list.append(temp_scores.copy())

    else:
        print("Angles were not detected!")

    capture.release()
    out.release()

    cv2.destroyAllWindows()

    end = time.time()
    print("Elapsed time: {} seconds".format(end - start))
    print("Finished processing {}".format(file))

with open("{}/score_report.csv".format(DST_DIR), "w") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=field_names)
    writer.writeheader()
    writer.writerows(score_list)
print("Done")
