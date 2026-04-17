import ctypes
import win32gui

ARENA_W = 540
ARENA_H = 960
GRID_W = 9
GRID_H = 18
PX_BORDER = 36.4


def make_dpi_aware():
    """Enable DPI awareness so coordinate transforms match real screen pixels."""
    try:
        user32 = ctypes.windll.user32
        if hasattr(user32, "SetProcessDpiAwarenessContext"):
            user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
        else:
            user32.SetProcessDPIAware()
    except Exception:
        pass


def grid_to_pixel(gx, gy, arena_w=ARENA_W, arena_h=ARENA_H, grid_w=GRID_W, grid_h=GRID_H):
    """Convert grid cell indices to BlueStacks virtual pixel coordinates (cell center)."""
    gx_i = int(round(float(gx)))
    gy_i = int(round(float(gy)))

    gx_i = max(0, min(gx_i, grid_w - 1))
    gy_i = max(0, min(gy_i, grid_h - 1))

    x_px = (gx_i + 0.5) / grid_w * arena_w
    y_px = (gy_i + 0.5) / grid_h * arena_h
    return x_px, y_px


def bluestacks_to_global_coords(bs_x, bs_y, bluestacks_resolution=(ARENA_W, ARENA_H), window_title="BlueStacks App Player 1"):
    """Invert global->BlueStacks mapping to recover screen coordinates for clicking."""
    make_dpi_aware()

    hwnd = win32gui.FindWindow(None, window_title)
    if not hwnd:
        raise RuntimeError(f"{window_title} window not found")

    left, top, right, bottom = win32gui.GetClientRect(hwnd)
    window_width = right - left
    window_height = bottom - top
    if window_width <= 0 or window_height <= 0:
        raise ValueError(f"Invalid window dimensions: width={window_width}, height={window_height}")

    origin_x, origin_y = win32gui.ClientToScreen(hwnd, (0, 0))

    virtual_w, virtual_h = bluestacks_resolution
    rel_x = float(bs_x) * window_width / float(virtual_w)
    rel_y = float(bs_y) * (window_height - PX_BORDER) / float(virtual_h) + PX_BORDER

    global_x = origin_x + rel_x
    global_y = origin_y + rel_y
    return int(round(global_x)), int(round(global_y))

