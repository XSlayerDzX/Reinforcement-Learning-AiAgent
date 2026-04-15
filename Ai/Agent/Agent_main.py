#script to handle the main loop for the agent that will play
# this script is supposed to do the following:
# step 1 : use the streamtoframe function to take a frame from the window of the game
# step 2 : input that frame to our cv model to get the current state of the game
# step 3 : clean and transform the output of the cv model to be ready for the lstm model
# step 4 : have a buffer that stores frames and padd with zeros if the buffer is not full
# step 5 : input that buffer window into the lstm to get a prediction of action,pos x and y
# step 6 : here the agent is going to start doing its work : ------> {
    # if action predicted is "wait" then the agent wont do anything and skips
    # if action is anything else then "wait" then the agent will first get the slot of that action card
    # then the agent will translate the grid of x and y into pure pixel coordinates
    # then excutes pyautogui to press the slot number in the keyboard and move the mouse to the mouse coordinates
    # and click at that position ( masking for position grids is not done yet )
import pandas as pd
# Note : each frame information is saved into a csv for logging to be used later for training and improving the model
# Note : we will implement the auto start match in this script as well

#importing
#import pyauto gui
import pyautogui as pya
from marshmallow.fields import Boolean
from pymsgbox import password

from Ai.Create_DataSet import Create_Dataset_Row
from Ai.Data_Cleaning import final_clean
from Ai.Stream_to_frame import Frame_Handler

Agent_State = {
    "current_frame" : "",
    "current_card" : "",
    "current_elixir" : "",
    "current_id" : "",
    "current_match_id" : "",
    "current_towers" : {},
    "current_slots": {},
}

Agent_global_state = {
    "current_match_id": 0,
}


def react_agent(action,pos_x,pos_y):
    if action == "wait": return
    else:
        #get the slot where that card is currently in so we can press it
        # press the keyboard button
        # translate the pos x and pos y from grid to pixel coordinates
        # move the mouse to the pixel coordinates and click
        pass





    pass

def Agent(state = True):
    #check if bluestack window is open
    if pya.getWindowsWithTitle("BlueStacks App Player 1")[0].isActive == False:
        print("Please open the BlueStacks window and start the game.")
        return

    current_frame = None
    current_card = None
    current_elixir = None
    current_id = None
    current_match_id = Agent_global_state["current_match_id"]
    current_towers = None
    current_slots = None
    current_id = 0

    #start the match if the start pixels are found ( there should be a made function for this already )
    # else keep looping untill its found

    # if match is started we call the function that takes a frame from the game
    # here there should be a loop that keeps taking frames
    while state:

        current_frame = Frame_Handler()
        row_dict = Create_Dataset_Row(current_frame, current_id, current_match_id)

        if row_dict:
            current_slots = {
                "slot_1" : row_dict["slot_1"],
                "slot_2" : row_dict["slot_2"],
                "slot_3" : row_dict["slot_3"],
                "slot_4" : row_dict["slot_4"],
            }
            current_elixir = row_dict["Elixir"]
            Agent_State["current_frame"] = current_frame
            Agent_State["current_id"] = current_id
            Agent_State["current_elixir"] = current_elixir

            row_csv = row_dict.to_csv()
            cleaned_row_csv = final_clean(row_csv,match_id=current_match_id)
            # now this input should be added to a buffer function to create a window if there is not enough frames
            # the function should padd the window with zeroes and return a list of frames
            #window = window_handler()

            #since a window is ready
            # we can input it to the lstm model to get a prediction of action, pos x and pos y
            #action, pos_x, pos_y = lstm_model.predict(window)

            #now we can call the react function to execute the action (wait do nothing else execute the action )
            #react_agent(action,pos_x,pos_y)

            current_id += 1



    Agent_global_state["current_match_id"] = current_match_id + 1



