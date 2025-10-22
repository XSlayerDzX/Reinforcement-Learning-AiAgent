import subprocess
from Tools.scripts.md5sum import bufsize


adb_path = r"C:\Program Files\BlueStacks_nxt\HD-Adb.exe" # adb path for bluestack
device_id = "emulator-5554"
Mouse_Click_Event = "/dev/input/event4" # event4 is the event responsible for touch input in bluestacks
Keyboard_Click_Event = "/dev/input/event3"


def Capture_Click():

    process = subprocess.Popen(
        [adb_path, "-s", device_id, "shell", "getevent", "-lt", Mouse_Click_Event],
        stdout=subprocess.PIPE,
        text=True,
        bufsize=1
)
    current_x = None
    current_y = None
    print("Listening for clicks... Ctrl+C to stop")

    try:
        for line in process.stdout:
            print(line)
            line = line.strip()
            if "0035" in line:
                current_x = int(line.split()[-1], 16)
            elif "0036" in line:
                current_y = int(line.split()[-1], 16)
            elif "BTN_TOUCH" in line and "UP" in line:
                if current_x is not None and current_y is not None:
                    print(f"TAP: X={current_x}, Y={current_y}")
                    current_x = None
                    current_y = None
    except KeyboardInterrupt:
        print("hey")


#subprocess.run([adb_path, "connect", "127.0.0.1:5555"])
#subprocess.run([adb_path, "disconnect", "127.0.0.1:5555"])
def list_connected_devices():
    result = subprocess.run([adb_path, "devices"], capture_output=True, text=True)
    print("Connected devices:")
    print(result.stdout)

list_connected_devices()



