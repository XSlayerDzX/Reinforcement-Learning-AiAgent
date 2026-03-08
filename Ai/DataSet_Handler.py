from os import waitpid
from time import sleep
import pandas as pd
from Ai.State_Tracker import interrupt
from Event_listners import *
from Ai.Stream_to_frame import *
from State_Tracker import *

id = 0
match_id = 1
mouse_listener , keyboard_listener = Start_Listeners()

try:
    while True:

        current_frame = Frame_Handler(count=id)

        if current_frame:
            State_Tracker.Current_img = current_frame
            row_dict = Create_Dataset_Row(current_frame, id, match_id)
            State_Tracker.Current_Id = id
            if row_dict:
                match_dict_input["data"].append(row_dict)
                print(f"Row {id} added to dataset.")
            else:
                print(f"Failed to create dataset row for frame {id}. Skipping.")
                os.remove(current_frame)
                sleep(1)
                continue
        else:
            print("No frame captured. Skipping dataset row creation.")
            continue
        id +=1
        sleep(2)

except KeyboardInterrupt:
    print("\nKeyboardInterrupt received. Saving data before exiting...")

finally:
    mouse_listener.stop()
    keyboard_listener.stop()

    mouse_listener.join(timeout=1)
    keyboard_listener.join(timeout=1)
    print(f"input: {match_dict_input}")
    print(f"output {match_dict_output}")
    df_input = pd.DataFrame(match_dict_input["data"])
    df_output = pd.DataFrame(match_dict_output["data"])
    df_input.to_csv(F"match_input_{match_id}.csv", index=False)
    df_output.to_csv(F"match_output_{match_id}.csv", index=False)
    print("Data saved successfully.")












