import os
import json
import glob

# 클래스 이름 → ID 매핑
class_name_to_id = {
    "유리병": 0
}

def letterbox_resize_shape(img_w, img_h, target_size=(640, 640)):
    r = min(target_size[0] / img_h, target_size[1] / img_w)
    new_w, new_h = int(round(img_w * r)), int(round(img_h * r))
    pad_w, pad_h = target_size[1] - new_w, target_size[0] - new_h
    pad_w /= 2
    pad_h /= 2
    return r, pad_w, pad_h

def xyxy_to_yolo_letterbox(x1, y1, x2, y2, img_w, img_h, target_size=(640, 640)):
    r, pad_w, pad_h = letterbox_resize_shape(img_w, img_h, target_size)

    x1_new = x1 * r + pad_w
    y1_new = y1 * r + pad_h
    x2_new = x2 * r + pad_w
    y2_new = y2 * r + pad_h

    # YOLO 형식으로 정규화된 좌표 계산
    x_center = (x1_new + x2_new) / 2 / target_size[1]
    y_center = (y1_new + y2_new) / 2 / target_size[0]
    w = abs(x2_new - x1_new) / target_size[1]
    h = abs(y2_new - y1_new) / target_size[0]

    return [x_center, y_center, w, h]

def convert_json_to_yolo_letterbox(json_path, output_txt_path, target_size=(640, 640)):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    annotations = data.get("ANNOTATION_INFO", [])
    img_info = data.get("IMAGE_INFO", {})
    img_w = img_info.get("IMAGE_WIDTH")
    img_h = img_info.get("IMAGE_HEIGHT")

    if img_w is None or img_h is None:
        print(f"❌ 이미지 크기 누락: {json_path}")
        return

    yolo_lines = []

    for ann in annotations:
        cls_name = ann.get("CLASS", "")
        class_id = class_name_to_id.get(cls_name)

        if class_id is None:
            print(f"⚠️ 알 수 없는 클래스: {cls_name}")
            continue

        for point in ann.get("POINTS", []):
            if len(point) != 4:
                print(f"❌ 좌표 오류: {point}")
                continue

            x1, y1, x2, y2 = point
            yolo_box = xyxy_to_yolo_letterbox(x1, y1, x2, y2, img_w, img_h, target_size)

            if not all(0.0 <= v <= 1.0 for v in yolo_box):
                print(f"❌ 정규화 오류: {yolo_box}")
                continue

            yolo_lines.append(f"{class_id} {' '.join(f'{v:.6f}' for v in yolo_box)}")

    os.makedirs(os.path.dirname(output_txt_path), exist_ok=True)
    with open(output_txt_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(yolo_lines))

    print(f"✅ 변환 완료: {output_txt_path}")

def convert_folder_jsons_to_yolo_letterbox(json_dir, output_dir, target_size=(640, 640)):
    json_files = glob.glob(os.path.join(json_dir, "*.json"))
    print(f"📂 변환 시작: {json_dir} → {output_dir} ({len(json_files)}개 파일)")

    for json_file in json_files:
        filename = os.path.splitext(os.path.basename(json_file))[0]
        output_txt = os.path.join(output_dir, f"{filename}.txt")
        convert_json_to_yolo_letterbox(json_file, output_txt, target_size=target_size)

if __name__ == "__main__":
    # 훈련 데이터와 검증 데이터 경로 설정
    train_json_dir = "F:/Data/Re/Training/Label"
    train_label_output_dir = "F:/Data/Re/Training/labels"
    val_json_dir = "F:/Data/Re/Validation/Label"
    val_label_output_dir = "F:/Data/Re/Validation/labels"

    # 훈련 데이터와 검증 데이터에 대해 YOLO 포맷으로 변환
    convert_folder_jsons_to_yolo_letterbox(train_json_dir, train_label_output_dir)
    convert_folder_jsons_to_yolo_letterbox(val_json_dir, val_label_output_dir)

    print("🎯 모든 변환 완료")
