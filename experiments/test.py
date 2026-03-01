import subprocess
adb_path = r"C:\Program Files\BlueStacks_nxt\HD-Adb.exe"
device_id = "emulator-5554"
Mouse_Click_Event = "/dev/input/event5" # event4 is the event responsible for touch input in bluestacks
Keyboard_Click_Event = "/dev/input/event2"


#subprocess.run([adb_path,"-s",device_id,"shell","getevent","-p"]) #See the Events commands
subprocess.run([adb_path,"-s",device_id,"shell","getevent","-lt",Mouse_Click_Event])

