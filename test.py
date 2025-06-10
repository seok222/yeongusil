import os
import cv2
import matplotlib.pyplot as plt

def check_labels_and_images(image_folder, label_folder, limit=5):
    # 이미지 폴더와 라벨 폴더 내의 모든 파일 확인
    image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
    label_files = [f for f in os.listdir(label_folder) if f.endswith('.txt')]

    print(f"Total images found: {len(image_files)}")
    print(f"Total labels found: {len(label_files)}")

    # 상위 5개의 이미지 파일만 확인
    image_files = image_files[:limit]

    # 각 이미지와 라벨 파일 매칭
    for image_file in image_files:
        # 이미지 파일 경로
        image_path = os.path.join(image_folder, image_file)

        # 라벨 파일 경로 (이미지 파일명에서 확장자 제외하고 .txt로 변경)
        label_file = os.path.splitext(image_file)[0] + '.txt'
        label_path = os.path.join(label_folder, label_file)

        # 라벨 파일이 존재하는지 확인
        if label_file in label_files:
            print(f"Label found for: {image_file}")
            # 이미지 로드
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 색상 변환 (OpenCV는 BGR로 읽음)

            # 라벨 파일 열기
            with open(label_path, 'r') as f:
                labels = f.readlines()

            # 라벨을 이미지에 그리기
            for label in labels:
                label = label.strip().split()
                class_id = int(label[0])
                x_center, y_center, w, h = map(float, label[1:])

                # 이미지 크기
                img_height, img_width, _ = image.shape

                # 바운딩 박스 좌표 계산
                x1 = int((x_center - w / 2) * img_width)
                y1 = int((y_center - h / 2) * img_height)
                x2 = int((x_center + w / 2) * img_width)
                y2 = int((y_center + h / 2) * img_height)

                # 라벨 클래스 ID와 함께 바운딩 박스 그리기
                cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)  # 박스 그리기 (빨강색)
                cv2.putText(image, f'Class {class_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)  # 라벨 텍스트 추가 (초록색)

            # 이미지 출력
            plt.imshow(image)
            plt.title(f"Image: {image_file}")
            plt.axis('off')  # 축 숨기기
            plt.show()

        else:
            print(f"Warning: No label found for {image_file}")

# 이미지 폴더와 라벨 폴더 경로 설정
image_folder = "F:/Data/Re/Training/images"
label_folder = "F:/Data/Re/Training/labels"

# 이미지와 라벨 확인
check_labels_and_images(image_folder, label_folder)
