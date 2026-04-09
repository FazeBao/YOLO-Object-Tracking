import os
import cv2
import uuid
from flask import Flask, request, render_template, jsonify, url_for
from werkzeug.utils import secure_filename
from ultralytics import YOLO

main = Flask(__name__, template_folder='.')

# --- CẤU HÌNH ĐƯỜNG DẪN ---
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Load model 
model = YOLO('best.pt') 

def process_image(input_path, output_path):
    # Logic dự đoán ảnh
    img_results = model.predict(source=input_path, conf=0.25, imgsz=640)
    for r in img_results:
        img_frame = r.plot()
        cv2.imwrite(output_path, img_frame) # Lưu ảnh

def process_video(input_path, output_path):
    # Logic đọc video
    cap = cv2.VideoCapture(input_path)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Sử dụng 'avc1' để có thể phát được trên trình duyệt thay vì mp4v
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Tracking
    results = model.track(source=input_path, stream=True, conf=0.25, imgsz=640, persist=True, tracker="botsort.yaml")
    
    for result in results:
        frame = result.plot()
        out.write(frame) # Ghi frame
        
    out.release()
    cap.release()

@main.route('/')
def index():
    return render_template('index.html')

@main.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'Không tìm thấy file'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Chưa chọn file'})

    # Tạo tên file ngẫu nhiên để tránh trùng lặp giữa nhiều người dùng
    ext = file.filename.rsplit('.', 1)[1].lower()
    filename = f"{uuid.uuid4().hex}.{ext}"
    input_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(input_path)

    output_filename = f"result_{filename}"
    output_path = os.path.join(RESULT_FOLDER, output_filename)

    file_type = ''
    try:
        # Kiểm tra loại file và gọi hàm xử lý tương ứng
        if ext in ['jpg', 'jpeg', 'png', 'webp']:
            process_image(input_path, output_path)
            file_type = 'image'
        elif ext in ['mp4', 'avi', 'mov']:
            process_video(input_path, output_path)
            file_type = 'video'
        else:
            return jsonify({'error': 'Định dạng không được hỗ trợ'})

        # Trả về URL của file kết quả
        result_url = url_for('static', filename=f'results/{output_filename}')
        return jsonify({'success': True, 'url': result_url, 'type': file_type})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    main.run(debug=True, port=5000)