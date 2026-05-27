import os
import xml.etree.ElementTree as ET
import random
import sys

CONFIG = {
    'RAW_TRAIN_XML': './datasets/raw/DETRAC-Train-Annotations-XML',
    'RAW_TEST_XML': './datasets/raw/DETRAC-Test-Annotations-XML',
    'RAW_IMG': './datasets/raw/DETRAC-Images',
    'OUTPUT': './datasets/yolo',
    'TRAIN_VAL_SPLIT': 0.8,
    'RANDOM_SEED': 42
}

VEHICLE_MAP = {'car': 0, 'bus': 1, 'van': 2, 'others': 3}

DEFAULT_IMG_W = 960.0
DEFAULT_IMG_H = 540.0

def get_yolo_bbox(box_attrib, img_w, img_h):
    """
    Converts XML bounding box attributes to YOLO normalized format.
    """
    l, t, w, h = map(
        float,
        [
            box_attrib['left'],
            box_attrib['top'],
            box_attrib['width'],
            box_attrib['height']
        ]
    )
    x_c = (l + w / 2) / img_w
    y_c = (t + h / 2) / img_h
    return x_c, y_c, w / img_w, h / img_h

def process_sequence(xml_path, img_seq_dir, target_set):
    """
    Parses a UA-DETRAC XML annotation file and converts it to YOLO format.
    Creates symlinks for images and generates normalized .txt labels.
    """
    if not os.path.exists(img_seq_dir):
        print(f"[WARNING] Image directory not found: {img_seq_dir}. Skipping.")
        return

    tree = ET.parse(xml_path)
    root = tree.getroot()
    seq_name = root.attrib.get('name')

    # Extract image dimensions with fallback
    attr = root.find('sequence_attribute')
    if attr is not None and 'width' in attr.attrib and 'height' in attr.attrib:
        img_w = float(attr.attrib['width'])
        img_h = float(attr.attrib['height'])
    else:
        img_w = DEFAULT_IMG_W
        img_h = DEFAULT_IMG_H

    # Set up output directories
    labels_dir = os.path.join(CONFIG['OUTPUT'], target_set, 'labels')
    images_dir = os.path.join(CONFIG['OUTPUT'], target_set, 'images')
    os.makedirs(labels_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)

    frames = root.findall('.//frame')

    for frame in frames:
        seq_gt = []

        # Format frame number to 5 digits
        f_num = int(frame.attrib.get('num', 0))
        unique_name = f"{seq_name}_img{f_num:05d}"

        yolo_lab_path = os.path.join(labels_dir, f"{unique_name}.txt")

        src_img = os.path.abspath(os.path.join(img_seq_dir,
                                               f"img{f_num:05d}.jpg"))
        dst_img = os.path.join(images_dir, f"{unique_name}.jpg")

        # O(1) Check: Only process XML targets if the source image physically
        # exists
        if os.path.exists(src_img):
            targets = frame.findall('.//target')

            for target in targets:
                box = target.find('box')
                if box is None:
                    continue

                # Extract coordinates and class ID
                x, y, w, h = get_yolo_bbox(box.attrib, img_w, img_h)

                v_type_elem = target.find('attribute')
                v_type = v_type_elem.attrib.get('vehicle_type') \
                        if v_type_elem is not None else 'others'
                cls_id = VEHICLE_MAP.get(v_type, 3) # Defaults to 'others' (3)

                # Append to batch list
                seq_gt.append(f"{cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")

            # Batch write targets to .txt file
            if seq_gt:
                with open(yolo_lab_path, 'w') as f:
                    for line in seq_gt:
                        f.write(line + '\n')

            # Create symlink for the image (if not already linked)
            if not os.path.lexists(dst_img):
                os.symlink(src_img, dst_img)

def run_conversion():
    for k, path in list(CONFIG.items())[:3]:
        if not os.path.exists(path):
            print(f"[ERROR] Directory not found: {path}")
            sys.exit(1)

    random.seed(CONFIG['RANDOM_SEED'])
    print(f"Starting data restructuring (Seed: {CONFIG['RANDOM_SEED']})...")

    # 1. Process Train/Val Sets
    train_xml_files = [
        f for f in os.listdir(CONFIG['RAW_TRAIN_XML']) if f.endswith('.xml')
    ]
    random.shuffle(train_xml_files)
    split_idx = int(len(train_xml_files) * CONFIG['TRAIN_VAL_SPLIT'])

    for i, xml_file in enumerate(train_xml_files):
        target = 'train' if i < split_idx else 'val'
        seq_name = xml_file.replace('.xml', '')
        process_sequence(os.path.join(CONFIG['RAW_TRAIN_XML'], xml_file),
                         os.path.join(CONFIG['RAW_IMG'], seq_name), target)
        print(f"[{target.upper()}] Processed sequence: {seq_name}")

    # 2. Process Test Set
    test_xml_files = [
        f for f in os.listdir(CONFIG['RAW_TEST_XML']) if f.endswith('.xml')
    ]
    for xml_file in test_xml_files:
        seq_name = xml_file.replace('.xml', '')
        process_sequence(os.path.join(CONFIG['RAW_TEST_XML'], xml_file),
                         os.path.join(CONFIG['RAW_IMG'], seq_name), 'test')
        print(f"[TEST] Processed sequence: {seq_name}")

if __name__ == "__main__":
    run_conversion()
