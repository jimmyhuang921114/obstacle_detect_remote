import cv2
import numpy as np
from paddleocr import PaddleOCR, draw_ocr

# ✅ 初始化 PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang="en")  # 识别中文+英文，可改成 lang="en" 仅识别英文

# ✅ 打开摄像头
cap = cv2.VideoCapture(2)
  # 0 代表默认摄像头，可修改为 1, 2 选择其他摄像头

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("❌ 无法获取摄像头画面")
        break

    # ✅ OpenCV 处理：转换 BGR → RGB
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # ✅ OCR 文字识别
    results = ocr.ocr(img_rgb, cls=True)

    # ✅ 绘制识别框
    for line in results:
        for word_info in line:
            box, (text, confidence) = word_info

            # 转换坐标格式
            box = np.array(box, dtype=np.int32)

            # 绘制检测框
            cv2.polylines(frame, [box], isClosed=True, color=(0, 255, 0), thickness=2)

            # 显示识别结果
            cv2.putText(frame, f"{text} ({confidence:.2f})", (box[0][0], box[0][1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # ✅ 显示处理后的视频流
    cv2.imshow("PaddleOCR 实时识别", frame)

    # 按 `q` 退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 关闭摄像头和窗口
cap.release()
cv2.destroyAllWindows()
