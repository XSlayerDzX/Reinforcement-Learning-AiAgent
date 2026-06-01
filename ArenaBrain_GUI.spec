# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_dynamic_libs

VENV = r'C:\Users\SlayerDz\PycharmProjects\clash-royale-rl-agent\.venv\Lib\site-packages'
PROJECT = r'C:\Users\SlayerDz\PycharmProjects\clash-royale-rl-agent'

datas = [
    (PROJECT + r'\Ai', 'Ai'),
    (PROJECT + r'\arena_web_integration', 'arena_web_integration'),
]
binaries = []
hiddenimports = [
    # GUI
    'customtkinter',
    'PIL._tkinter_finder',
    'PIL.Image',
    'PIL.ImageTk',
    # ML
    'torch',
    'torch.nn',
    'torch.nn.functional',
    'torch.optim',
    'torch.utils',
    'torch.utils.data',
    # Data
    'numpy',
    'pandas',
    # Computer vision
    'cv2',
    'mss',
    # Clipping
    'pyclipper',
    # Input / window control
    'pyautogui',
    'pygetwindow',
    'pynput',
    'pynput.keyboard',
    'pynput.mouse',
    # Windows API
    'win32api',
    'win32con',
    'win32gui',
    # Network / env
    'requests',
    'dotenv',
    'python_dotenv',
    'inference_sdk',
    # Misc
    'debugpy',
]

# Collect full packages (data files + binaries + hidden imports)
for pkg in ['customtkinter', 'torch', 'cv2', 'pyclipper', 'mss', 'pyautogui', 'pynput']:
    tmp = collect_all(pkg)
    datas    += tmp[0]
    binaries += tmp[1]
    hiddenimports += tmp[2]

a = Analysis(
    [PROJECT + r'\Ai\RL\PPO_GUI.py'],
    pathex=[PROJECT, PROJECT + r'\.venv\Lib\site-packages'],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='ArenaBrain_GUI',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='ArenaBrain_GUI',
)
