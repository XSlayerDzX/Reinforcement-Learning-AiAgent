import cv2
import numpy as np
import os

WIN_HSV_LOWER  = np.array([ 80, 100, 150])
WIN_HSV_UPPER  = np.array([105, 255, 255])

LOSS_HSV_LOWER = np.array([140,  20, 180])
LOSS_HSV_UPPER = np.array([165,  90, 255])

# Minimum fraction of ROI pixels that must land inside the colour range.
COVERAGE_THRESHOLD = 0.45


WIN_ROI  = (373, 398, 200, 328)   # y1, y2, x1, x2
LOSS_ROI = ( 95, 120, 200, 328)


def _coverage(roi_bgr: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """Return the percentage of pixels in *roi_bgr* whose HSV value falls within [lower, upper]."""
    if roi_bgr.size == 0:
        return 0.0
    roi_hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    mask    = cv2.inRange(roi_hsv, lower, upper)
    # mask is a binary image where pixels in range are 255 and others are 0;
    # count nonzero and divide by total pixels to get coverage fraction.
    return float(np.count_nonzero(mask)) / mask.size


def check_match_status(path: str) -> str:
    """
    Instead of comparing *average* colours ,
    we measure what *percentage* of ROI pixels fall inside
    the target HSV colour range.
    """
    if not os.path.exists(path):
        print(f"[check_match_status] File not found: {path}")
        return "error_no_frame"

    frame = cv2.imread(str(path))
    if frame is None:
        print(f"[check_match_status] OpenCV could not read: {path}")
        return "error_no_frame"

    y1w, y2w, x1w, x2w = WIN_ROI
    y1l, y2l, x1l, x2l = LOSS_ROI

    roi_win  = frame[y1w:y2w, x1w:x2w]
    roi_loss = frame[y1l:y2l, x1l:x2l]

    # debug output save with imwrite for visual verification of ROIs and HSV thresholds;
    os.makedirs("image_process", exist_ok=True)
    cv2.imwrite("image_process/debug_roi_winner.png", roi_win)
    cv2.imwrite("image_process/debug_roi_loser.png",  roi_loss)

    win_cov  = _coverage(roi_win,  WIN_HSV_LOWER,  WIN_HSV_UPPER)
    loss_cov = _coverage(roi_loss, LOSS_HSV_LOWER, LOSS_HSV_UPPER)

    print(f"[check_match_status] win_coverage={win_cov:.1%}  loss_coverage={loss_cov:.1%}  "
          f"threshold={COVERAGE_THRESHOLD:.0%}")

    # We do NOT use a relative comparison (win vs loss) because both could be above threshold simultaneously
    if win_cov >= COVERAGE_THRESHOLD > loss_cov:
        return "win"
    if loss_cov >= COVERAGE_THRESHOLD > win_cov:
        return "loss"

    # Edge case: both above threshold → something unexpected; treat as ongoing
    if win_cov >= COVERAGE_THRESHOLD and loss_cov >= COVERAGE_THRESHOLD:
        print("[check_match_status] WARNING: both ROIs matched — treating as ongoing")

    return "ongoing"


# calibrate() for verification on a known frame: print mean and std HSV values in each ROI to help adjust the HSV thresholds if needed.
def calibrate(path: str) -> None:

    frame = cv2.imread(str(path))
    if frame is None:
        print("Cannot open file for calibration.")
        return

    y1w, y2w, x1w, x2w = WIN_ROI
    y1l, y2l, x1l, x2l = LOSS_ROI

    for label, roi_bgr in [("WIN ROI", frame[y1w:y2w, x1w:x2w]),
                            ("LOSS ROI", frame[y1l:y2l, x1l:x2l])]:
        hsv   = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        mean  = hsv.reshape(-1, 3).mean(axis=0)
        std   = hsv.reshape(-1, 3).std(axis=0)
        print(f"[calibrate] {label}  mean HSV=({mean[0]:.1f}, {mean[1]:.1f}, {mean[2]:.1f})  "
              f"std=({std[0]:.1f}, {std[1]:.1f}, {std[2]:.1f})")


# ─────────────────────────────────────────────────────────────────────────────
# if __name__ == "__main__":
#     BASE = "C:/Users/SK-TECH/PycharmProjects/clash-royale-rl-agent/Ai/image_process"
#     #print(check_match_status(f"{BASE}/temp_screens/capture_0.png"))
#     print(check_match_status(f"{BASE}/test_win.png"))


    #calibrate(f"{BASE}/test_win.png")