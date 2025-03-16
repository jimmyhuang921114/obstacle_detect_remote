import cv2

for i in range(8):  # 逐个测试 /dev/video0 ~ /dev/video7
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"✅ 摄像头 {i} 可用")
        cap.release()
    else:
        print(f"❌ 摄像头 {i} 不可用")
