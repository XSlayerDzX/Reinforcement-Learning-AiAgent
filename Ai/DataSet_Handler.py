from os import waitpid
from time import sleep
import pandas as pd
import queue as q

from Ai.State_Tracker import interrupt
from Event_listners import *
from Ai.Stream_to_frame import *
import State_Tracker

match_dict_input = {
    "data" : [],
}
match_dict_output = {
    "data" : [],
}

play = True
id = 0
match_id = 1
mouse_listener , keyboard_listener = Start_Listeners()

while play and id < 3:

    current_frame = Frame_Handler(count=id)

    if current_frame:
        State_Tracker.Current_img = current_frame
        row_dict = Create_Dataset_Row(current_frame, id, match_id)
        if row_dict:
            if State_Tracker.interrupt:
                State_Tracker.interrupt = False
                print("Interrupt received, creating dataset row with current card and position.")
                output = Output_Dataset_Schema(State_Tracker.CurrentCard, State_Tracker.pos_x, State_Tracker.pos_y, id)
                State_Tracker.CurrentCard = None
            else:
                output = Output_Dataset_Schema("wait", None, None, id)
            match_dict_input["data"].append(row_dict)
            match_dict_output["data"].append(output)
            print(f"Row {id} added to dataset.")
            id += 1
        else:
            print(f"Failed to create dataset row for frame {id}. Skipping.")
            os.remove(current_frame)
    else:
        print("No frame captured. Skipping dataset row creation.")
    id +=1
    sleep(10)

mouse_listener.stop()
keyboard_listener.stop()

mouse_listener.join(timeout=1)
keyboard_listener.join(timeout=1)
print(match_dict_input)
print(match_dict_output)
df = pd.DataFrame(match_dict_input["data"])
df = pd.concat([df, pd.DataFrame(match_dict_output["data"])], axis=1)
df.to_csv("dataset.csv", index=False)












