# python
from pynput import keyboard, mouse  # Importing keyboard and mouse listeners from the pynput library
import ClashRoyalData  # Importing a custom module for Clash Royale data handling
from Ai.Roboflow.CardPredictor import ExtractSlot  # Importing the ExtractSlots function from the CardPredictor module
import win32gui  # Importing win32gui for interacting with Windows GUI elements
import pygetwindow as gw
from Create_DataSet import *
from State_Tracker import *
img_path = r'C:\Users\SlayerDz\Desktop\Screenshot_2025.09.14_21.27.07.354.png'
import ctypes

def make_dpi_aware():
    try:
        # Windows 10+ recommended API
        # Load the Windows DLL user32.dll and grant me access to its functions
        user32 = ctypes.windll.user32
        # Try SetProcessDpiAwarenessContext (Windows 10)
        # hasattr(object, "attribute_name") checks whether an object has an attribute or function.
        if hasattr(user32, "SetProcessDpiAwarenessContext"):
            # DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 = -4
            user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
        else:
            # Fallback to SetProcessDPIAware (older)
            user32.SetProcessDPIAware()
    except Exception:
        pass

def convert_to_bluestacks_coords(global_x, global_y, bluestacks_resolution=(540, 960)):
    make_dpi_aware()

    hwnd = win32gui.FindWindow(None, "BlueStacks App Player 4")
    if not hwnd:
        raise RuntimeError("BlueStacks window not found")

    left2, top2, right2, bottom2 = win32gui.GetClientRect(hwnd)
    window_width = right2 - left2
    window_height = bottom2 - top2

    # Validate window dimensions before proceeding
    if window_width <= 0 or window_height <= 0:
        raise ValueError(f"Invalid window dimensions: width={window_width}, height={window_height}")

    origin_x, origin_y = win32gui.ClientToScreen(hwnd, (0, 0))
    rel_x = global_x - origin_x
    rel_y = global_y - origin_y

    rel_x = max(0, min(rel_x, window_width))
    rel_y = max(0, min(rel_y, window_height))

    virtual_w, virtual_h = bluestacks_resolution
    px_brd = 36.4

    bs_x = round((rel_x * virtual_w / window_width), 2)
    bs_y = round((rel_y - px_brd) * virtual_h / (window_height - px_brd), 2)
    bs_y = max(0, bs_y)

    return bs_x, bs_y

def Click_Validation(x,y):
    CurrentCard = State_Tracker.CurrentCard
    if State_Tracker.CurrentCard:
        try:
            card_elix = ClashRoyalData.ElixirCost[CurrentCard]
        except:
            card_elix = 4
        current_elix = State_Tracker.CurrentElixir + 1 # added 1 incase of delay in predection
        if current_elix >= card_elix:
            print(f"Click at BlueStacks coordinates: ({x}, {y}) is valid for card: {State_Tracker.CurrentCard}")
            State_Tracker.CurrentElixir = abs(current_elix - card_elix)
            return True
        else:
            print(f"Not enough elixir for card: {State_Tracker.CurrentCard}. Current elixir: {current_elix}, required: {card_elix}. Click ignored.")
            return False
    else:
        return False


def on_click(x, y, button, pressed):
    if pressed:
        try:
            windows = gw.getWindowsWithTitle("BlueStacks App Player 4")
            if not windows:
                print("BlueStacks window not found. Click ignored.")
                return

            new_x, new_y = convert_to_bluestacks_coords(x, y, bluestacks_resolution=(540, 960))
            State_Tracker.interrupt = True

            Validated = Click_Validation(new_x, new_y)
            if Validated and State_Tracker.interrupt:
                id = State_Tracker.Current_Id
                print("click validated, interrupting dataset creation and updating state tracker...")
                output = Output_Dataset_Schema(State_Tracker.CurrentCard, new_x, new_y, id)
                match_dict_output["data"].append(output)
                State_Tracker.interrupt = False
                State_Tracker.pos_x = new_x
                State_Tracker.pos_y = new_y
            else:
                print("Click was not valid, no action taken.")
        except (ValueError, RuntimeError) as e:
            print(f"Error processing click: {e}. Click ignored.")
        except Exception as e:
            print(f"Unexpected error in on_click: {type(e).__name__}: {e}")

def CurrentCard(keypressed,img):
    """
    Updates the current card based on the key pressed.

    :param keypressed: The key pressed by the user, used to identify the card slot.
    """
    if img is None:
        print("No image available to extract card information.")
        return
    key = str(keypressed)
    try:
        slots = ExtractSlot(img)
        current = slots.get(f"slot_{keypressed}")  #3 Get the card in the corresponding slot
        if current:
            #print(f"this is the selected card: {current}")
            State_Tracker.CurrentCard = current  # Update the current card in ClashRoyalData
            State_Tracker.output_action_cards[State_Tracker.Current_Id] = current  # Store the selected card for the current dataset row
        else:
            print(f"No card found in slot {key}")
    except Exception as e:
        print(f"Error in CurrentCard: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()


def on_press(key):
    """
    Handles keyboard key press events and updates the current card if a specific key is pressed.

    :param key: The key that was pressed.
    """
    try:
        # Check if the key has a 'char' attribute (alphanumeric keys)
        if hasattr(key, 'char') and key.char:
            print(f"Key {key.char} pressed")
            # Check if the key is one of the predefined keys and update the current card
            if key.char in ['1', '2', '3', '4']:
                CurrentCard(key.char, State_Tracker.Current_img)
    except AttributeError:
        # This is for special keys like Escape, Shift, etc.
        print(f"Special key {key} pressed")
    except Exception as e:
        print(f"Error in on_press: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()


def Start_Listeners():
    """
    Starts the mouse and keyboard listeners to capture events.
    """
    # Create mouse and keyboard listeners
    mouse_listener = mouse.Listener(on_click=on_click)
    keyboard_listener = keyboard.Listener(on_press=on_press)

    print("Starting mouse and keyboard listeners...")
    mouse_listener.start()
    keyboard_listener.start()

    print("Listening... Press Escape to exit the keyboard.")

    return mouse_listener, keyboard_listener