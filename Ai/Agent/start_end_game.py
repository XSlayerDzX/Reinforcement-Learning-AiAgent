import cv2
import time
import win32api
import win32con
import logging
from pathlib import Path
from typing import Tuple, Optional, Dict, Union



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

    max_val = float(max_val)
    max_loc = (int(max_loc[0]), int(max_loc[1]))
    h, w = template.shape[:2]
    w, h = int(w), int(h)
    return max_val, max_loc, (w, h)


def click_at(x: int, y: int) -> None:
    """Click at absolute screen coordinates using the Windows API."""
    win32api.SetCursorPos((x, y))
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0)
    time.sleep(0.05)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0)


def auto_play(
    frame_path: Union[str, Path],
    zone: Dict[str, int],
    thresh: float = 0.99,
    debug: bool = False
) -> Optional[str]:
    """
    Process one captured frame and click a matched button if found.

    Parameters:
        frame_path: path to the screenshot image already captured elsewhere
        zone: dict containing at least 'left' and 'top' offsets
        img_yellow_path: template path for yellow/play button
        img_blue_path: template path for blue/ok button
        img_red_path: template path for red button
        thresh: template matching threshold
        debug: if True, show a debug window with the match

    Returns:
        The detected button name ('play', 'ok', or 'red') if matched,
        otherwise None.
    """
    img_yellow_path = r"C:\Users\SlayerDz\PycharmProjects\clash-royale-rl-agent\Ai\Agent\yellow.jpg"
    img_blue_path = r"C:\Users\SlayerDz\PycharmProjects\clash-royale-rl-agent\Ai\Agent\blue.jpg"
    img_red_path = r"C:\Users\SlayerDz\PycharmProjects\clash-royale-rl-agent\Ai\Agent\red.jpg"

    logging.info("Loading templates")
    img_y = load_template(img_yellow_path)
    img_b = load_template(img_blue_path)
    img_r = load_template(img_red_path)

    frame_path = str(frame_path)
    img2 = cv2.imread(frame_path)
    if img2 is None:
        logging.warning("Could not read captured frame: %s", frame_path)
        return None

    max_val_y, max_loc_y, (w1, h1) = match_template(img2, img_y)
    max_val_b, max_loc_b, (w2, h2) = match_template(img2, img_b)
    max_val_r, max_loc_r, (w3, h3) = match_template(img2, img_r)

    # Choose the strongest match
    best_val = max_val_y
    best_loc = max_loc_y
    tw, th = w1, h1
    button_name = "play"

    if max_val_r > best_val:
        best_val = max_val_r
        best_loc = max_loc_r
        tw, th = w3, h3
        button_name = "red"

    if max_val_b > best_val:
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

        logging.info(
            "Detected %s button with confidence %.3f at %s",
            button_name,
            best_val,
            (global_x, global_y)
        )

        click_at(global_x, global_y)

        if debug:
            top_left = best_loc
            bottom_right = (best_loc[0] + tw, best_loc[1] + th)
            vis = img2.copy()
            cv2.rectangle(vis, top_left, bottom_right, (0, 255, 0), 2)
            cv2.putText(
                vis,
                f"{button_name} {best_val:.2f}",
                (top_left[0], top_left[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )
            cv2.imshow("match", vis)
            cv2.waitKey(500)
            cv2.destroyWindow("match")

        return button_name

    return None
