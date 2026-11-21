"""Tkinter helpers.

DPI awareness: on a scaled Windows display (for example 175%) a non-aware Tk window is
stretched as a bitmap, which blurs the CT image and misplaces the brush edge. Awareness is
switched on before Tk starts; Tk scales fonts itself and pixel sizes go through px().
This affects only this process's own window, not any system setting.
"""

import sys


def enable_dpi_awareness():
    if sys.platform != "win32":
        return
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        try:
            import ctypes
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


def px(widget, n):
    """Converts a size given at 96 dpi to the screen's scale."""
    try:
        return int(round(n * widget.winfo_fpixels("1i") / 96.0))
    except Exception:
        return int(n)
