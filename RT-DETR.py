import os
import json
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# 데이터 경로 설정 (변환된 YOLO 라벨 파일 경로)
data_yaml_path = "F:/Data/Re/Data.yaml"  # data.yaml 경로 (YOLO 데이터셋 설정 파일)

# 모델 로드 (pretrained=True일 경우 사전 학습된 모델 로드)
model = YOLO("F:/Data/yolo11n.pt")  # YOLOv11n 모델 파일 경로

def check_labels_and_images(image_folder, label_folder, output_folder, limit=5):
    # 이미지 폴더와 라벨 폴더 내의 모든 파일 확인
    image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
    label_files = [f for f in os.listdir(label_folder) if f.endswith('.txt')]

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

            # 결과 이미지 저장
            output_image_path = os.path.join(output_folder, f"labeled_{image_file}")
            cv2.imwrite(output_image_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))  # OpenCV는 BGR로 저장

            print(f"Saved labeled image: {output_image_path}")

# 학습 실행
if __name__ == '__main__':
    # 학습 진행
    results = model.train(
        data=data_yaml_path,  # data.yaml 경로 (데이터셋 설정)
        epochs=100,  # 에폭 수
        batch=4,  # 배치 크기
        imgsz=640,  # 이미지 크기
        lr0=0.001,  # 초기 학습률
        device=0,  # GPU 사용
        optimizer='Adam',  # Adam 옵티마이저 사용
        warmup_epochs=5,  # 워밍업 에폭 수
        single_cls=True,  # 클래스가 여러 개인 경우 False
        save=True,  # 학습 후 모델 저장
        save_period=10,  # 모델 저장 주기
        pretrained=True,  # 사전 학습된 모델 사용 여부
        verbose=True,  # 학습 진행 표시 여부
        seed=42,  # 랜덤 시드
        name="yolo11n_model"  # 모델 저장 이름
    )

    # 모델 학습 후 성능 출력
    metrics = results.metrics
    box_metrics = metrics.get('metrics', {}).get('box', {})

    # 성능 출력
    results_dict = {
        'precision': box_metrics.get('precision', 0),
        'recall': box_metrics.get('recall', 0),
        'mAP@0.5': box_metrics.get('map50', 0),
        'mAP@0.5:0.95': box_metrics.get('map', 0)
    }

    # 결과 저장
    save_path = 'F:/Data/yolo11n_metrics.json'
    with open(save_path, 'w') as f:
        json.dump(results_dict, f, indent=4)

    print(f"✅ 학습 결과 저장 완료: {save_path}")

    # 라벨링된 이미지 5장 저장
    image_folder = "F:/Data/Re/Training/images"
    label_folder = "F:/Data/Re/Training/labels"
    output_folder = "F:/Data/Re/Training/labeled_images"  # 라벨링된 이미지를 저장할 폴더

    # 폴더가 존재하지 않으면 생성
    os.makedirs(output_folder, exist_ok=True)

    # 이미지와 라벨 확인 및 저장
    check_labels_and_images(image_folder, label_folder, output_folder)
