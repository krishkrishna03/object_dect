import os
from flask import Flask, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
import torch
import cv2
from PIL import Image

UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'static/results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4', 'avi', 'mov'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            if filename.lower().endswith(('.mp4', '.avi', '.mov')):
                # Process video file
                video_path = filepath
                results_video_path = os.path.join(app.config['RESULTS_FOLDER'], 'video_results.mp4')
                plain_video_path = os.path.join(app.config['RESULTS_FOLDER'], 'plain_video.mp4')
                cap = cv2.VideoCapture(video_path)
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out_results = cv2.VideoWriter(results_video_path, fourcc, cap.get(cv2.CAP_PROP_FPS), (int(cap.get(3)), int(cap.get(4))))
                out_plain = cv2.VideoWriter(plain_video_path, fourcc, cap.get(cv2.CAP_PROP_FPS), (int(cap.get(3)), int(cap.get(4))))

                detected_objects_set = set()

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    results = model(img)
                    annotated_frame = results.render()[0]
                    out_results.write(annotated_frame)
                    out_plain.write(frame)  # Write plain video to another file
                    
                    # Collect detected objects
                    predictions = results.pred[0].tolist()
                    for pred in predictions:
                        detected_objects_set.add(results.names[int(pred[5])])
                
                cap.release()
                out_results.release()
                out_plain.release()
                detected_objects = list(detected_objects_set)
                detected_objects.sort()  # Optional: sort detected objects alphabetically
                return redirect(url_for('uploaded_file', filename='video_results.mp4', plain_filename='plain_video.mp4', objects=detected_objects))
            
            else:
                # Process image file
                img = Image.open(filepath)
                results = model(img)
                results.save(save_dir=app.config['RESULTS_FOLDER'])
                predictions = results.pred[0].tolist()  # Get predictions
                detected_objects_set = set()
                for pred in predictions:
                    detected_objects_set.add(results.names[int(pred[5])])
                detected_objects = list(detected_objects_set)
                detected_objects.sort()  # Optional: sort detected objects alphabetically
                return redirect(url_for('uploaded_file', filename=filename, objects=detected_objects))
    
    return render_template('upload.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    plain_filename = request.args.get('plain_filename')
    detected_objects = request.args.getlist('objects')
    return render_template('uploaded.html', filename=filename, plain_filename=plain_filename, objects=detected_objects)

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(RESULTS_FOLDER, exist_ok=True)
    app.run(debug=True)
