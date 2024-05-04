import tkinter as tk
from tkinter import filedialog
import subprocess

class CrowdCounterGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Crowd Counter")

        # Create labels to display face count
        self.face_count_label = tk.Label(self.master, text="Number of Faces Detected: 0")
        self.face_count_label.pack()

        # Create buttons
        self.upload_button = tk.Button(self.master, text="Upload Image", command=self.upload_image)
        self.upload_button.pack()

        self.camera_button = tk.Button(self.master, text="Use Camera", command=self.use_camera)
        self.camera_button.pack()

    def upload_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            # Run YOLOv5 detection script with the uploaded image
            command = f"python yolov5/detect.py --weights yolov5/runs/train/exp6/weights/best.pt --img 640 --conf 0.25 --source {file_path}"
            self.run_detection(command)

    def use_camera(self):
        # Run YOLOv5 detection script with the camera feed
        command = "python yolov5/detect.py --weights yolov5/runs/train/exp6/weights/best.pt --img 640 --conf 0.25 --source 0"
        self.run_detection(command)

    def run_detection(self, command):
        self.output = subprocess.Popen(command, shell=True, text=True, stdout=subprocess.PIPE)
        self.update_gui()

    def update_gui(self):
        # Read the output from the subprocess
        output_lines = self.output.stdout.readlines()
        face_count = 0
        for line in output_lines:
            line = line.decode().strip()
            if "Number of Faces Detected:" in line:
                face_count = int(line.split(":")[-1].strip())
                break  # Break once the face count is found
        # Update the face count label
        self.face_count_label.config(text=f"Number of Faces Detected: {face_count}")

        # Check for updates every 100 milliseconds
        self.master.after(100, self.update_gui)

def main():
    root = tk.Tk()
    app = CrowdCounterGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()