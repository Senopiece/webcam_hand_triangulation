import sys
import os

if sys.platform.startswith("win"):
    import win32api
    import win32process
    import win32con

    def set_high_priority():
        """Sets the current process and its subprocesses to high priority on Windows."""
        try:
            pid = os.getpid()
            handle = win32api.OpenProcess(win32con.PROCESS_ALL_ACCESS, True, pid)
            win32process.SetPriorityClass(handle, win32process.REALTIME_PRIORITY_CLASS)
        except Exception as e:
            print(f"Failed to set high priority: {e}", file=sys.stderr)

else:

    def set_high_priority():
        """Dummy function for Linux (does nothing)."""
        pass
