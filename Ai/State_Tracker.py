Current_img = None
Current_Id = None
CurrentCard = None
pos_x = None
pos_y = None
CurrentElixir = 0
CurrentTowers = {}
CurrentTroops_ally = {}
CurrentTroops_enemy = {}
CurrentSlots = {}
interrupt = False

match_dict_input = {
    "data" : [],
}
match_dict_output = {
    "data" : [],
}



# import win32gui
#
# def enum_callback(hwnd, windows):
#     if win32gui.IsWindowVisible(hwnd):
#         title = win32gui.GetWindowText(hwnd)
#         if title:  # Skip empty titles
#             windows.append((hwnd, title))
#     return True
#
# windows = []
# win32gui.EnumWindows(enum_callback, windows)
# for hwnd, title in windows:
#     print(f"HWND: {hex(hwnd)}, Title: '{title}'")

