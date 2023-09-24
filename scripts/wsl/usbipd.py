import time
import os

# SOURCE: https://github.com/luxonis/depthai/issues/723
while True:
    output = os.popen("usbipd wsl list").read()  # List all USB devices
    rows = output.split("\n")
    for row in rows:
        if (
            "Movidius MyriadX" in row or "Luxonis Device" in row
        ) and "Not attached" in row:  # Check for OAK cameras that aren't attached
            busid = row.split(" ")[0]
            out = os.popen(
                f"usbipd wsl attach --busid {busid}"
            ).read()  # Attach an OAK camera
            print(out)
            print(f"Usbipd attached Myriad X on bus {busid}")  # Log
    time.sleep(0.5)
