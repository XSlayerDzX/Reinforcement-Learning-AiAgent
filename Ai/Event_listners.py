# python
from pynput import keyboard, mouse  # Importing keyboard and mouse listeners from the pynput library
import ClashRoyalData  # Importing a custom module for Clash Royale data handling
from CardPredictor import ExtractSlot  # Importing the ExtractSlots function from the CardPredictor module
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

    # Ensure DPI awareness is configured before any coordinate calculations
    make_dpi_aware()

    # Find BlueStacks window handle by title
    hwnd = win32gui.FindWindow(None, "BlueStacks App Player 1")
    if not hwnd:
        raise RuntimeError("BlueStacks window not found")
    else:
        print(f"Found BlueStacks window handle: {hwnd}")

    # Get BlueStacks window position and size
    left2, top2, right2, bottom2 = win32gui.GetClientRect(hwnd)

    window_width = right2 - left2
    window_height = bottom2 - top2

    # Convert the client origin (0,0) to screen coordinates
    origin_x, origin_y = win32gui.ClientToScreen(hwnd, (0, 0))

    # Calculate relative position inside the window
    rel_x = global_x - origin_x
    rel_y = global_y - origin_y

    # what do rel_x and rel_y represent here
    # Clamp relative position to window bounds
    # rel_x and rel_y should be between 0 and window_width/window_height
    rel_x = max(0, min(rel_x, window_width))
    rel_y = max(0, min(rel_y, window_height))

    virtual_w, virtual_h = bluestacks_resolution

    px_brd = 36.4 # pixels border to ignore the title bar and borders

    bs_x = round((rel_x * virtual_w / window_width), 2)
    bs_y = round((rel_y - px_brd) * virtual_h / (window_height - px_brd), 2)
    bs_y = max(0, bs_y)  # ignore title bar area

    return bs_x, bs_y

def Click_Validation(x,y):
    CurrentCard = State_Tracker.CurrentCard
    if State_Tracker.CurrentCard:
        card_elix = ClashRoyalData.ElixirCost[CurrentCard]
        current_elix = State_Tracker.CurrentElixir
        if current_elix >= card_elix:
            print(f"Click at BlueStacks coordinates: ({x}, {y}) is valid for card: {State_Tracker.CurrentCard}")
            State_Tracker.CurrentElixir = abs(current_elix - card_elix)
            return True
        else:
            print(f"Not enough elixir for card: {State_Tracker.CurrentCard}. Current elixir: {current_elix}, required: {card_elix}. Click ignored.")
            return False
    else:
        print("No current card selected. Click ignored.")


def on_click(x,y,pressed):
    """
    Handles mouse click events and converts the coordinates to BlueStacks coordinates.
    :param x: Global X coordinate of the mouse click.
    :param y: Global Y coordinate of the mouse click.
    :param button: The mouse button that was clicked.
    :param pressed: Boolean indicating whether the button was pressed.
    """
    if pressed:
        windows = gw.getWindowsWithTitle("BlueStacks App Player 1")
        if not windows:
            raise RuntimeError("BlueStacks window not found.")
        window = windows[0]
        new_x, new_y = convert_to_bluestacks_coords(x, y, bluestacks_resolution=(540, 960))
        #print(f"bluestacks_x: {new_x}, bluestacks_y: {new_y}")
        Validated = True#Click_Validation(new_x, new_y)
        if Validated:
            id = State_Tracker.Current_Id
            State_Tracker.interrupt = True
            print("click validated, interrupting dataset creation and updating state tracker...")
            output = Output_Dataset_Schema(State_Tracker.CurrentCard, new_x, new_y, id)
            match_dict_output["data"].append(output)
            State_Tracker.pos_x = new_x
            State_Tracker.pos_y = new_y
            print(f"x: {State_Tracker.pos_x}, y: {State_Tracker.pos_y}")
        else:
            print("Click was not valid, no action taken.")

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
            print(f"this is the selected card: {current}")
            State_Tracker.CurrentCard = current  # Update the current card in ClashRoyalData2
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