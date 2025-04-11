import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
from io import BytesIO

# 얼굴 인식 모델 초기화
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)

def calculate_angle(left_eye, right_eye):
    delta_y = right_eye[1] - left_eye[1]
    delta_x = right_eye[0] - left_eye[0]
    angle_rad = np.arctan2(delta_y, delta_x)
    angle_deg = np.degrees(angle_rad)
    return angle_deg

def process_image(image_file):
    image = Image.open(image_file).convert("RGB")
    image_np = np.array(image)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    result = face_mesh.process(image_np)

    if not result.multi_face_landmarks:
        st.warning("😥 얼굴이 감지되지 않았어요.")
        return None, image

    landmarks = result.multi_face_landmarks[0].landmark
    h, w, _ = image_np.shape
    left_eye = (int(landmarks[33].x * w), int(landmarks[33].y * h))
    right_eye = (int(landmarks[263].x * w), int(landmarks[263].y * h))

    angle = calculate_angle(left_eye, right_eye)
    direction = "왼쪽" if angle > 0 else "오른쪽"
    text = f"{direction}으로 약 {abs(angle):.2f}도 기울어졌어요"

    # 이미지에 표시
    cv2.line(image_bgr, left_eye, right_eye, (0, 255, 0), 2)
    cv2.circle(image_bgr, left_eye, 4, (255, 0, 0), -1)
    cv2.circle(image_bgr, right_eye, 4, (255, 0, 0), -1)
    cv2.putText(image_bgr, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    image_annotated = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_final = Image.fromarray(image_annotated)
    return text, image_final

def image_to_bytes(image: Image.Image) -> bytes:
    buf = BytesIO()
    image.save(buf, format="PNG")
    byte_im = buf.getvalue()
    return byte_im

# Streamlit UI
st.title("🧠 얼굴 기울기 측정기")
st.write("정면 얼굴 사진을 업로드하면, 얼굴이 왼쪽이나 오른쪽으로 몇 도 기울어졌는지 알려드려요!")

uploaded_file = st.file_uploader("얼굴 사진을 업로드 해주세요", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="원본 이미지", use_column_width=True)

    result_text, result_image = process_image(uploaded_file)

    if result_text:
        st.success(result_text)
        st.image(result_image, caption="기울기 분석 결과", use_column_width=True)

        # 다운로드 버튼 추가
        img_bytes = image_to_bytes(result_image)
        st.download_button(
            label="📥 결과 이미지 다운로드",
            data=img_bytes,
            file_name="face_tilt_result.png",
            mime="image/png"
        )