# import tkinter as tk
# from tkinter import ttk
# import pygetwindow as gw
# import mss
# import numpy as np
# import cv2

# def start_realtime_capture():
#     selected_app = combobox.get()
#     if selected_app:
#         window = gw.getWindowsWithTitle(selected_app)[0]
#         window.activate()  

#         left, top, width, height = window.left, window.top, window.width, window.height
#         print(f"Left: {left}, Top: {top}, Width: {width}, Height: {height}")  # Debug info

#         with mss.mss() as sct:
#             monitor = {"top": top, "left": left, "width": width, "height": height}

#             while True:
#                 screenshot = sct.grab(monitor)
#                 img = np.array(screenshot)
                
#                 cv2.namedWindow("Real-time Screenshot", cv2.WINDOW_NORMAL)
#                 cv2.resizeWindow("Real-time Screenshot", 640, 480) 

#                 cv2.imshow("Real-time Screenshot", cv2.cvtColor(img, cv2.COLOR_BGRA2BGR))
                
#                 if cv2.waitKey(1) & 0xFF == ord('q'):
#                     break

#             cv2.destroyAllWindows()


# app_titles = gw.getAllTitles()
# app_titles = [title for title in app_titles if title]  

# root = tk.Tk()
# root.title("Select an Application")

# combobox = ttk.Combobox(root, values=app_titles)
# combobox.pack(pady=10)

# button = tk.Button(root, text="Start Real-time Capture", command=start_realtime_capture)
# button.pack(pady=10)

# root.mainloop()


import tkinter as tk
from tkinter import ttk
import pygetwindow as gw
import mss
import numpy as np
import cv2

def start_realtime_capture():
    selected_app = combobox.get()
    if selected_app:
        window = gw.getWindowsWithTitle(selected_app)[0]
        window.activate()  

        with mss.mss() as sct:
            left, top, width, height = window.left, window.top, window.width, window.height
            monitor = {"top": top, "left": left, "width": width, "height": height}
            
            cv2.namedWindow("Real-time Screenshot", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Real-time Screenshot", 640, 480)
            cv2.moveWindow("Real-time Screenshot", -10000, -10000)  # Di chuyển ra khỏi màn hình chính
            
            while True:
                screenshot = sct.grab(monitor)
                img = np.array(screenshot)

                cv2.imshow("Real-time Screenshot", cv2.cvtColor(img, cv2.COLOR_BGRA2BGR))
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cv2.destroyAllWindows()

app_titles = gw.getAllTitles()
app_titles = [title for title in app_titles if title]

root = tk.Tk()
root.title("Select an Application")

combobox = ttk.Combobox(root, values=app_titles)
combobox.pack(pady=10)

button = tk.Button(root, text="Start Real-time Capture", command=start_realtime_capture)
button.pack(pady=10)

root.mainloop()
