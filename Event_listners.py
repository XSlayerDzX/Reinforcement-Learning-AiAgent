from numpy import integer  # Importing the integer type from the numpy library
from pynput import keyboard, mouse  # Importing keyboard and mouse listeners from the pynput library
from threading import Thread  # Importing Thread for multithreading
import StatePredictor  # Importing a custom module for state prediction
import ClashRoyalData  # Importing a custom module for Clash Royale data handling
from CardPredictor import ExtractSlots  # Importing the ExtractSlots function from the CardPredictor module
import win32gui  # Importing win32gui for interacting with Windows GUI elements

def CurrentCard(keypressed):
        """
        Updates the current card based on the key pressed.

        :param keypressed: The key pressed by the user, used to identify the card slot.
        """
        current = None
        Slots = ExtractSlots()  # Extract available card slots
        current = Slots.get(f"slot_{keypressed}")  # Get the card in the corresponding slot
        if current:
            print(f"Current card selected: {current}")
            ClashRoyalData.CurrentCard = current  # Update the current card in ClashRoyalData
        else:
            print(f"No card found in slot {keypressed}")

def convert_to_bluestacks_coords(global_x, global_y, bluestacks_resolution=(960, 540)):
        """
        Converts global mouse coordinates to BlueStacks window coordinates.

        :param global_x: Mouse X coordinate on the Windows desktop.
        :param global_y: Mouse Y coordinate on the Windows desktop.
        :param bluestacks_resolution: Tuple representing the internal resolution of BlueStacks (width, height).
        :return: Tuple (x, y) representing the coordinates inside the BlueStacks screen.
        """
        # Find BlueStacks window handle by title
        hwnd = win32gui.FindWindow(None, "BlueStacks App Player")
        if not hwnd:
            raise RuntimeError("BlueStacks window not found")

        # Get BlueStacks window position and size
        left, top, right, bottom = win32gui.GetWindowRect(hwnd)
        window_width = right - left
        window_height = bottom - top

        # Calculate relative position inside the window
        rel_x = global_x - left
        rel_y = global_y - top

        # Clamp values inside window bounds
        rel_x = max(0, min(rel_x, window_width))
        rel_y = max(0, min(rel_y, window_height))

        # Scale relative position to BlueStacks resolution
        scale_x = bluestacks_resolution[0] / window_width
        scale_y = bluestacks_resolution[1] / window_height

        bluestacks_x = int(rel_x * scale_x)
        bluestacks_y = int(rel_y * scale_y)

        return bluestacks_x, bluestacks_y

def on_click(x, y, button, pressed):
        """
        Handles mouse click events and converts the coordinates to BlueStacks coordinates.

        :param x: Global X coordinate of the mouse click.
        :param y: Global Y coordinate of the mouse click.
        :param button: The mouse button that was clicked.
        :param pressed: Boolean indicating whether the button was pressed.
        """
        if pressed:
            new_x, new_y = convert_to_bluestacks_coords(x, y)
            print(f"Mouse new click at ({new_x}, {new_y}) with {button}")

def on_key(key):
        """
        Handles keyboard key press events and updates the current card if a specific key is pressed.

        :param key: The key that was pressed.
        """
        try:
            print(f"Key {key.char} pressed")
            # Check if the key is one of the predefined keys and update the current card
            if key == keyboard.Key.f1 or keyboard.Key.f2 or keyboard.Key.f3 or keyboard.Key.f4:
                CurrentCard(key.char)
        except AttributeError:
            print(f"Special key {key} pressed")

    # Create mouse and keyboard listeners
mouse_listener = mouse.Listener(on_click=on_click)
keyboard_listener = keyboard.Listener(on_press=on_key)

def start_mouse():
        """
        Starts the mouse listener in the current thread.
        """
        mouse_listener.start()
        mouse_listener.join()

def start_keyboard():
        """
        Starts the keyboard listener in the current thread.
        """
        keyboard_listener.start()
        keyboard_listener.join()

    # Run listeners in separate threads
mouse_thread = Thread(target=start_mouse)
keyboard_thread = Thread(target=start_keyboard)

mouse_thread.start()
keyboard_thread.start()

try:
        # Wait for both threads to complete
        mouse_thread.join()
        keyboard_thread.join()
except KeyboardInterrupt:
        # Stop listeners on keyboard interrupt
        mouse_listener.stop()
        keyboard_listener.stop()
        print("Listeners stopped.")









