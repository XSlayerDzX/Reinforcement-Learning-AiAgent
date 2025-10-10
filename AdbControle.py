import subprocess
from Tools.scripts.md5sum import bufsize

device_id = "emulator-5554"
event_device = "/dev/input/event4" # event4 is the event responsible for touch input in bluestacks
# another event for keyboard tracking
adb_path = r"C:\Program Files\BlueStacks_nxt\HD-Adb.exe" # adb path for bluestack

def Capture_Click():

    process = subprocess.Popen(
        [adb_path, "-s", device_id, "shell", "getevent", "-lt", event_device],
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












#subprocess.run([adb_path, "connect", "127.0.0.1:5555"])
subprocess.run([adb_path, "disconnect", "127.0.0.1:5555"])



