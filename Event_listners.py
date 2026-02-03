from pynput import keyboard, mouse  # Importing keyboard and mouse listeners from the pynput library
import StatePredictor  # Importing a custom module for state prediction
import ClashRoyalData  # Importing a custom module for Clash Royale data handling
from CardPredictor import ExtractSlots  # Importing the ExtractSlots function from the CardPredictor module
import win32gui  # Importing win32gui for interacting with Windows GUI elements
import pygetwindow as gw
img_path = r"C:\Users\SK-TECH\Downloads\photo_2026-02-02_17-35-32.jpg"
def CurrentCard(keypressed):
        """
        Updates the current card based on the key pressed.

        :param keypressed: The key pressed by the user, used to identify the card slot.
        """
        current = None
        Slots = ExtractSlots(img_path)  # Extract available card slots
        current = Slots.get(f"slot_{keypressed}")  # Get the card in the corresponding slot
        if current:
           ClashRoyalData.CurrentCard = current  # Update the current card in ClashRoyalData
           print(f"Current card set to: {ClashRoyalData.CurrentCard}")
        else:
            print(f"No card found in slot {keypressed}")


# remove scaling issues on high-DPI displays
# transform logical coordinates to physical coordinates
import ctypes
def make_dpi_aware():
    try:
        # Windows 10+ recommended API
        # Charge la DLL Windows user32.dll et donne-moi accès à ses fonctions
        user32 = ctypes.windll.user32
        # Try SetProcessDpiAwarenessContext (Windows 10)
        # hasattr(objet, "nom_attribut") vérifie si un objet possède un attribut ou une fonction.
        if hasattr(user32, "SetProcessDpiAwarenessContext"):
            # DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 = -4
            user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
        else:
            # Fallback to SetProcessDPIAware (older)
            user32.SetProcessDPIAware()
    except Exception:
        pass

make_dpi_aware()

def convert_to_bluestacks_coords(global_x, global_y, bluestacks_resolution=(540, 960)):



    # Find BlueStacks window handle by title
    hwnd = win32gui.FindWindow(None, "BlueStacks App Player 2")
    if not hwnd:
        raise RuntimeError("BlueStacks window not found")

    # Get BlueStacks window position and size
    left2 , top2, right2, bottom2 = win32gui.GetClientRect(hwnd)

    window_largeur = right2 - left2
    window_hauteur = bottom2 - top2



    # Convertir l'origine client (0,0) en coordonnées écran
    origin_x ,  origin_y = win32gui.ClientToScreen(hwnd, (0, 0))

    # Calculate relative position inside the window
    rel_x = global_x - origin_x
    rel_y = global_y - origin_y

    # what do rel_x and rel_y represent here
    # Clamp relative position to window bounds
    # rel_x and rel_y should be between 0 and window_largeur/window_hauteur
    rel_x = max(0, min(rel_x, window_largeur))
    rel_y = max(0, min(rel_y, window_hauteur))


    virtual_w, virtual_h = bluestacks_resolution

    px_brd = 36.4 # pixels border to ignore the title bar and borders

    bs_x = round((rel_x * virtual_w / window_largeur), 2)
    bs_y = round( (rel_y - px_brd)* virtual_h / (window_hauteur - px_brd),2)
    bs_y = max(0,bs_y)  # ignore title bar area

    return bs_x , bs_y

def on_click(x, y, button, pressed):
        """
        Handles mouse click events and converts the coordinates to BlueStacks coordinates.
        :param x: Global X coordinate of the mouse click.
        :param y: Global Y coordinate of the mouse click.
        :param button: The mouse button that was clicked.
        :param pressed: Boolean indicating whether the button was pressed.
        """
        if pressed:
            windows = gw.getWindowsWithTitle("BlueStacks App Player 2")
            if not windows:
                raise RuntimeError("BlueStacks window not found.")
            window = windows[0]
            new_x, new_y = convert_to_bluestacks_coords(x, y, bluestacks_resolution=(540, 960))
            print(f"bluestacks_x: {new_x}, bluestacks_y: {new_y}")


def on_press(key):
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
keyboard_listener = keyboard.Listener(on_press=on_press)

print("Démarrage des écouteurs de souris et de clavier...")
mouse_listener.start()
keyboard_listener.start()

print("Écoute en cours... Appuyez sur Echap pour quitter le clavier.")

# --- 3. Maintien du programme en vie ---
try:
    # On demande au programme principal d'attendre que les listeners finissent.
    # Si on_press retourne False (touche Echap), k_listener s'arrête.
    keyboard_listener.join()
    mouse_listener.join()
except KeyboardInterrupt:
    # Pour gérer le Ctrl+C dans le terminal proprement
    print("\nStop the listner.")
    keyboard_listener.stop()
    mouse_listener.stop()









