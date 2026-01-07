import cv2
from deepface import DeepFace

# 1. Khởi tạo Webcam (0 là camera mặc định)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Không thể mở webcam")
    exit()

while True:
    # 2. Đọc từng khung hình
    ret, frame = cap.read()
    if not ret:
        break

    # Lật ngược ảnh cho giống gương (tùy chọn)
    frame = cv2.flip(frame, 1)

    try:
        # 3. Sử dụng DeepFace để phân tích cảm xúc
        # enforce_detection=False để không báo lỗi nếu không thấy mặt
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        
        # DeepFace trả về list, lấy phần tử đầu tiên
        face_data = result[0]
        
        # Lấy cảm xúc chủ đạo
        dominant_emotion = face_data['dominant_emotion']
        
        # Lấy tọa độ khuôn mặt để vẽ khung
        region = face_data['region']
        x, y, w, h = region['x'], region['y'], region['w'], region['h']

        # 4. Vẽ hình chữ nhật quanh mặt và ghi tên cảm xúc
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, dominant_emotion.upper(), (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    except Exception as e:
        # Bỏ qua lỗi nếu chưa nhận diện được gì
        pass

    # 5. Hiển thị kết quả
    cv2.imshow('Emotion Recognition', frame)

    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()