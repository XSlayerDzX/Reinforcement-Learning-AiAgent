import mss
import mss.tools
import time
import os
import win32gui


def flux_capture_temporaire(dossier_temp="temp_screens", intervalle=1.0):
    # os.makedirs(dossier_temp, exist_ok=True) used to create the directory if it doesn't exist
    if not os.path.exists(dossier_temp):
        os.makedirs(dossier_temp)

    # --- INITIALISATION of MSS ---
    with (mss.mss() as sct):

        # 2. Find BlueStacks window handle by title

        hwnd = win32gui.FindWindow(None, "BlueStacks App Player 2")

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
                raise RuntimeError("BlueStacks window not found")
        count = 0
        while True:
            # Verification: If the window is minimised, we wait.
            if win32gui.IsIconic(hwnd):
                print("Window minimised, waiting...")
                time.sleep(1)
                continue

            # 3. UPDATING DYNAMIC COORDINATES
            # We recalculate each turn in case you move the window.
            rect = win32gui.GetWindowRect(hwnd)
            x = rect[0]
            y = rect[1] + 54  # Offset for the title bar
            w = rect[2] - x - 40
            h = rect[3] - y

            # Correction des bordures invisibles de Windows 10/11 (souvent -7 pixels)
            # if x is negatif but close to 0, we set it to 0 to avoid crash
            if x < 0 and x > -20: x = 0
            if y < 0 and y > -20: y = 0

            # Security check to prevent GetDIBits crash
            if w <= 0 or h <= 0:
                print("Invalid window dimensions, waiting...")
                time.sleep(1)
                continue

            monitor = {"top": y, "left": x, "width": w, "height": h}

            # 4. CAPTURE
            try:
                #timestamp used to avoid collisions by generating unique filenames
                count +=1
                name_fichier = os.path.join(dossier_temp, f"capture_{count}.png")

                sct_img = sct.grab(monitor)
                mss.tools.to_png(sct_img.rgb, sct_img.size, output=name_fichier)

                # Send the path to the main programme
                yield name_fichier

            except mss.exception.ScreenShotError as e:
                print(f"Capture error (the window may be off-screen) : {e}")

            # 5. NETTOYAGE
            time.sleep(intervalle)

# --- EXECUTION ---
print("Starting monitoring...")
try:
    for path_img in flux_capture_temporaire(intervalle=1.0):
        print(f"Traitement : {path_img}")
except KeyboardInterrupt:
    print("END")
except Exception as e:
    print(f"Fatal error : {e}")