import cv2
import numpy as np
import os

def check_match_status(path):

    if not os.path.exists(path):
        print(f"File not found: {path}")
        return "error_no_frame"

    frame_array = cv2.imread(str(path))

    if frame_array is None:
        print(f"OpenCV could not read: {path}")
        return "error_no_frame"

    y1_loss, y2_loss = 95,  120
    y1_win,  y2_win  = 373, 398
    x1, x2 = 200, 328

    roi_winner = frame_array[y1_win:y2_win,   x1:x2]
    roi_loser  = frame_array[y1_loss:y2_loss, x1:x2]

    cv2.imwrite("debug_roi_winner.png", roi_winner)
    cv2.imwrite("debug_roi_loser.png",  roi_loser)



    target_winner = np.array([102, 255, 255], dtype=np.float32)   # BGR
    target_loser  = np.array([255, 204, 255], dtype=np.float32)   # BGR


    avg_color_winner = np.mean(roi_winner.reshape(-1, 3), axis=0)
    avg_color_loser  = np.mean(roi_loser.reshape(-1, 3),  axis=0)


    dist_winner = np.linalg.norm(avg_color_winner - target_winner)
    dist_loser  = np.linalg.norm(avg_color_loser  - target_loser)

    color_diff_threshold = 20

    print(f" dist_winner={dist_winner:.1f}  dist_loser={dist_loser:.1f}")


    if dist_winner < dist_loser - color_diff_threshold:
        return "win"
    elif dist_loser < dist_winner - color_diff_threshold:
        return "loss"

    return "ongoing"