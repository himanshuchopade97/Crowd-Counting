import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from detect import run
import threading

class FaceDetectorGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Face Detector")

        # Create labels
        self.label_faces = tk.Label(self.master, text="Number of Faces: ")
        self.label_faces.pack()

        # Create canvas for image display
        self.canvas = tk.Canvas(self.master, width=640, height=480)
        self.canvas.pack()

        # Buttons
        self.btn_camera = tk.Button(self.master, text="Open Camera", command=self.open_camera)
        self.btn_camera.pack()

        self.btn_upload = tk.Button(self.master, text="Upload Image", command=self.upload_image)
        self.btn_upload.pack()

    def open_camera(self):
        # Run face detection on camera
        # results = run(source=0, nosave=True, view_img=False)

        # # Display number of faces
        # num_faces = run(source=0, nosave=True, view_img=False)
        # self.label_faces.config(text=f"Number of Faces: {num_faces}")

        # # Display live camera feed
        # self.display_camera_feed()
        # while self.master.winfo_exists():
        #     num_faces = run(source=0, nosave=True, view_img=False)
        #     self.label_faces.config(text=f"Number of Faces: {num_faces}")

        # # Display live camera feed
        #     self.display_camera_feed()
            camera_frame = run(source=0, nosave=True, view_img=False)
            num_faces = run(source=0, nosave=True, view_img=False)
            self.label_faces.config(text=f"Number of Faces: {num_faces}")
            self.display_camera_feed(camera_frame)     


    def upload_image(self):
        # Open file dialog to select image
        file_path = filedialog.askopenfilename()
        if file_path:
            # Run face detection on uploaded image
            
            # results = run(source=file_path, nosave=True, view_img=False)

            # Display number of faces
            num_faces = run(source=file_path, nosave=True, view_img=False)
            self.label_faces.config(text=f"Number of Faces: {num_faces}")

            # Display uploaded image
            self.display_uploaded_image(file_path)

    def display_camera_feed(self,camera_frame):
        # Get camera frame
        # camera_frame = run(source=0, nosave=True, view_img=False)
        image = Image.fromarray(camera_frame)

        # Resize image to fit canvas
        image.thumbnail((640, 480))

        # Convert image to PhotoImage
        photo = ImageTk.PhotoImage(image)

        # Display image on canvas
        self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        self.canvas.image = photo  # Keep a reference to prevent garbage collection

    def display_uploaded_image(self, file_path):
        # Open uploaded image
        image = Image.open(file_path)

        # Resize image to fit canvas
        image.thumbnail((640, 480))

        # Convert image to PhotoImage
        photo = ImageTk.PhotoImage(image)

        # Display image on canvas
        self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        self.canvas.image = photo  # Keep a reference to prevent garbage collection

def main():
    root = tk.Tk()
    app = FaceDetectorGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()