from flask import Flask, render_template, jsonify, request
import subprocess

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("gui.html")

@app.route("/camera", methods=["GET"])
def use_camera():
    # Run detect.py script with camera source
    command = "python yolov5/detect.py --weights yolov5/runs/train/exp6/weights/best.pt --img 640 --conf 0.25 --source 0"
    output = subprocess.run(command, shell=True, capture_output=True, text=True)
    # Parse output to get face count
    face_count = 0
    for line in output.stdout.splitlines():
        if "Number of Faces Detected:" in line:
            face_count = int(line.split(":")[-1].strip())
            break
    return jsonify({"face_count": face_count})

if __name__ == "__main__":
    app.run(debug=True)
