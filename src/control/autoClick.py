import cv2
import time
import win32api, win32con
import logging
from pathlib import Path
import argparse
from typing import Tuple
from src.vision.Stream_Frame import flux_capture_temporaire


def load_template(path: Path):
    """Load an image template and return it.

    Raises FileNotFoundError if the image cannot be loaded.
    """
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Template not found or unreadable: {path}")
    return img


def match_template(image, template) -> Tuple[float, Tuple[int, int], Tuple[int, int]]:
    """Match a template in an image using TM_CCORR_NORMED.

    Returns (max_val, max_loc, (w, h)).
    """
    res = cv2.matchTemplate(image, template, cv2.TM_CCORR_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    # Ensure returned coordinates and sizes are plain ints (type checker-friendly)
    max_val = float(max_val)
    max_loc = (int(max_loc[0]), int(max_loc[1]))
    h, w = template.shape[:2]
    w, h = int(w), int(h)
    return max_val, max_loc, (w, h)


def click_at(x: int, y: int) -> None:
    """Click at absolute screen coordinates using the Windows API.

    This uses win32api to be consistent with the original implementation.
    """
    win32api.SetCursorPos((x, y))
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0)
    time.sleep(0.05)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0)


def find_play_button(img_yellow_path: Path,
                     img_blue_path: Path,
                     thresh: float = 0.99,
                     poll_interval: float = 1.0,
                     debug: bool = False) -> None:
    """Continuously scan frames produced by flux_capture_temporaire and click matched buttons.

    - img_yellow_path / img_blue_path: paths to templates (play, ok)
    - thresh: matching threshold
    - poll_interval: seconds to wait between iterations
    - debug: if True, show visualization windows
    """
    logging.info("Loading templates")
    img_y = load_template(img_yellow_path)
    img_b = load_template(img_blue_path)

    for path_img, zone in flux_capture_temporaire():
        try:
            img2 = cv2.imread(path_img)
            if img2 is None:
                logging.warning("Could not read captured frame: %s", path_img)
                time.sleep(poll_interval)
                continue

            max_val_y, max_loc_y, (w1, h1) = match_template(img2, img_y)
            max_val_b, max_loc_b, (w2, h2) = match_template(img2, img_b)

            # Choose the stronger match
            if max_val_y >= max_val_b:
                best_val = max_val_y
                best_loc = max_loc_y
                tw, th = w1, h1
                button_name = "play"
            else:
                best_val = max_val_b
                best_loc = max_loc_b
                tw, th = w2, h2
                button_name = "ok"

            if best_val >= thresh:
                # center within the captured frame
                cx = int(best_loc[0] + tw // 2)
                cy = int(best_loc[1] + th // 2)

                # convert to global screen coordinates using zone offsets
                global_x = zone.get("left", 0) + cx
                global_y = zone.get("top", 0) + cy

                logging.info("Detected %s button with confidence %.3f at %s", button_name, best_val, (global_x, global_y))

                click_at(global_x, global_y)

                # small pause after click to allow UI to change
                time.sleep(1.0)

                if debug:
                    # draw rectangle and show
                    top_left = best_loc
                    bottom_right = (best_loc[0] + tw, best_loc[1] + th)
                    vis = img2.copy()
                    cv2.rectangle(vis, top_left, bottom_right, (0, 255, 0), 2)
                    cv2.putText(vis, f"{button_name} {best_val:.2f}", (top_left[0], top_left[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.imshow("match", vis)
                    cv2.waitKey(500)
                    cv2.destroyWindow("match")

            # normal poll interval between processed frames
            time.sleep(poll_interval)

        except KeyboardInterrupt:
            logging.info("Interrupted by user, exiting")
            break
        except Exception as e:
            logging.exception("Unexpected error while processing frame: %s", e)
            # don't break the whole loop on a single frame error; wait and continue
            time.sleep(poll_interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find and click Play/OK buttons from captured frames.")
    parser.add_argument("--play", type=Path, default=Path(__file__).parent / "images" / "play_button.png",
                        help="Path to the play button template")
    parser.add_argument("--ok", type=Path, default=Path(__file__).parent / "images" / "ok_button.png",
                        help="Path to the ok button template")
    parser.add_argument("--thresh", type=float, default=0.99, help="Template matching threshold (0..1)")
    parser.add_argument("--interval", type=float, default=1.0, help="Seconds between processing iterations")
    parser.add_argument("--debug", action="store_true", help="Show debug visualization windows")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    #example of out : 2026-02-24 09:37:29,123 INFO: User logged in

    try:
        find_play_button(args.play, args.ok, thresh=args.thresh, poll_interval=args.interval, debug=args.debug)
    finally:
        # ensure all OpenCV windows are closed on exit
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
