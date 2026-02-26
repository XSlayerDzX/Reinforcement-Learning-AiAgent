import os
import time
import mss
import mss.tools
import win32gui

def Frame_Handler(count=0, temp_folder="temp_screens"):
    """
    Capture single frame from BlueStacks. Returns filename or None.
    """
    # Create the directory if it doesn't exist
    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder)

    with mss.mss() as sct:
        # Find BlueStacks window handle by title
        hwnd = win32gui.FindWindow(None, "BlueStacks App Player 1")

        if not hwnd:
            # Fallback: partial search if the name changes
            def callback(h, handles):
                if "BlueStacks" in win32gui.GetWindowText(h):
                    handles.append(h)
            handles = []
            win32gui.EnumWindows(callback, handles)
            if handles:
                hwnd = handles[0]
            else:
                print("BlueStacks window not found")
                return None

        # Skip if minimized
        if win32gui.IsIconic(hwnd):
            print("Window minimised, skipping capture")
            return None

        # Get dynamic coordinates
        rect = win32gui.GetWindowRect(hwnd)
        x = rect[0]
        y = rect[1] + 40  # Title bar offset
        w = rect[2] - x - 40
        h = rect[3] - y

        # Fix negative borders
        if x < 0 and x > -20:
            x = 0
        if y < 0 and y > -20:
            y = 0

        # Invalid dimensions check
        if w <= 0 or h <= 0:
            print("Invalid window dimensions, skipping")
            return None

        monitor = {"top": y, "left": x, "width": w, "height": h}

        # Capture
        try:
            filename = os.path.join(temp_folder, f"capture_{count}.png")
            sct_img = sct.grab(monitor)
            mss.tools.to_png(sct_img.rgb, sct_img.size, output=filename)
            print(f"Captured: {filename}")
            return filename

        except mss.exception.ScreenShotError as e:
            print(f"Capture error (off-screen?): {e}")
            return None
