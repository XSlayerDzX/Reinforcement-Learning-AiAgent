import cv2
import time
import win32api
import win32con
import logging
from pathlib import Path
from typing import Tuple, Optional, Dict, Union

from Ai.Stream_to_frame import Frame_Handler


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
    templates: Dict,               # <-- preloaded templates passed in
    thresh: float = 0.92,          # lowered from 0.99 — was rejecting all matches
    debug: bool = False
) -> Optional[str]:
    """
    Process one captured frame and click a matched button if found.

    Parameters:
        frame_path: path to the screenshot image already captured elsewhere
        zone: dict containing at least 'left' and 'top' offsets
        templates: preloaded template images dict (loaded once outside the loop)
        thresh: template matching threshold (default 0.92)
        debug: if True, show a debug window with the match

    Returns:
        The detected button name if matched, otherwise None.
        Priority order: 'menu' -> 'training_camp' -> 'ok_training' -> 'ok'
    """

    img2 = cv2.imread(str(frame_path))
    if img2 is None:
        logging.warning("Could not read captured frame: %s", frame_path)
        return None

    # --- Run all template matches ---
    max_val_b,    max_loc_b,    (w1, h1) = match_template(img2, templates["ok"])
    max_val_menu, max_loc_menu, (w2, h2) = match_template(img2, templates["menu"])
    max_val_tc,   max_loc_tc,   (w3, h3) = match_template(img2, templates["training_camp"])
    max_val_ok_t, max_loc_ok_t, (w4, h4) = match_template(img2, templates["ok_training"])

    # Always print confidence scores — visible in NAV loop output
    print(
        f"[AUTO_PLAY] menu={max_val_menu:.4f}  training_camp={max_val_tc:.4f}"
        f"  ok_training={max_val_ok_t:.4f}  ok={max_val_b:.4f}  (thresh={thresh:.2f})"
    )

    # --- Candidate table (name, val, loc, w, h) ---
    candidates = [
        ("menu",          max_val_menu, max_loc_menu, w2, h2),
        ("training_camp", max_val_tc,   max_loc_tc,   w3, h3),
        ("ok_training",   max_val_ok_t, max_loc_ok_t, w4, h4),
        ("ok",            max_val_b,    max_loc_b,    w1, h1),
    ]

    best = max(
        (c for c in candidates if c[1] >= thresh),
        key=lambda c: c[1],
        default=None,
    )

    if best is None:
        return None

    button_name, best_val, best_loc, tw, th = best

    cx = int(best_loc[0] + tw // 2)
    cy = int(best_loc[1] + th // 2)

    global_x = zone.get("left", 0) + cx
    global_y = zone.get("top", 0) + cy

    print(f"[AUTO_PLAY] >>> clicking '{button_name}' (conf={best_val:.4f}) at ({global_x}, {global_y})")
    click_at(global_x, global_y)

    if debug:
        top_left     = best_loc
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
            2,
        )
        cv2.imshow("match", vis)
        cv2.waitKey(500)
        cv2.destroyWindow("match")

    return button_name
