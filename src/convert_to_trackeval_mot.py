import os
import xml.etree.ElementTree as ET
import random

CONFIG = {
    "RAW_TRAIN_XML": "./datasets/raw/DETRAC-Train-Annotations-XML",
    "RAW_TEST_XML": "./datasets/raw/DETRAC-Test-Annotations-XML",
    "OUTPUT": "./datasets/trackeval/data",
    "BENCHMARK": "ua-detrac",
    "TRAIN_VAL_SPLIT": 0.8,
    "RANDOM_SEED": 42
}


# =========================
# SAFE FLOAT
# =========================
def safe_float(v, default=0.0):
    try:
        return float(v)
    except (ValueError, TypeError):
        return default


# =========================
# WRITE GT
# =========================
def write_gt(frames, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    def frame_id(f):
        try:
            return int(f.attrib.get("num", 0)) + 1
        except (ValueError, TypeError):
            return 1

    frames = sorted(frames, key=frame_id)
    lines = []

    for f in frames:
        try:
            fid = int(f.attrib.get("num", 0)) + 1
        except (ValueError, TypeError):
            continue

        targets = f.findall(".//target")
        if len(targets) == 0:
            continue

        for obj in targets:
            box = obj.find("box")
            if box is None:
                continue

            x = safe_float(box.attrib.get("left"))
            y = safe_float(box.attrib.get("top"))
            w = safe_float(box.attrib.get("width"))
            h = safe_float(box.attrib.get("height"))

            if w <= 0 or h <= 0:
                continue

            try:
                oid = int(obj.attrib.get("id", -1))
            except (ValueError, TypeError):
                oid = -1

            attr = obj.find("attribute")

            # Visibility safety check
            trunc = 0.0
            if attr is not None:
                raw_trunc = attr.attrib.get("truncation_ratio", 0.0)
                trunc = safe_float(raw_trunc)

            visibility = max(0.0, min(1.0, 1.0 - trunc))

            # =========================
            # MOT FORMAT (STRICT)
            # =========================
            line = (
                f"{fid},{oid},{x:.2f},{y:.2f},{w:.2f},{h:.2f},"
                f"1,1,{visibility:.2f}"
            )
            lines.append(line)

    with open(path, "w") as file_out:
        file_out.write("\n".join(lines))


# =========================
# SEQINFO
# =========================
def write_seqinfo(seq_path, name, frames):
    os.makedirs(seq_path, exist_ok=True)

    seq_len = 0
    if frames:
        seq_len = max(int(f.attrib.get("num", 0)) for f in frames) + 1

    info_content = (
        f"[Sequence]\n"
        f"name={name}\n"
        f"imDir=img1\n"
        f"frameRate=30\n"
        f"seqLength={seq_len}\n"
        f"imWidth=960\n"
        f"imHeight=540\n"
        f"imExt=.jpg\n"
    )

    with open(os.path.join(seq_path, "seqinfo.ini"), "w") as f:
        f.write(info_content)


# =========================
# PROCESS SEQUENCE
# =========================
def process(xml_path, split_name):
    """
    Parses the XML and saves it to the correct split directory.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    default_name = os.path.basename(xml_path).replace(".xml", "")
    seq_name = root.attrib.get("name", default_name)

    frames = root.findall(".//frame")

    if len(frames) == 0:
        print(f"[SKIP] Empty: {seq_name}")
        return None

    base = os.path.join(
        CONFIG["OUTPUT"],
        "gt",
        f"{CONFIG['BENCHMARK']}-{split_name}",
        seq_name
    )

    gt_path = os.path.join(base, "gt", "gt.txt")

    write_gt(frames, gt_path)
    write_seqinfo(base, seq_name, frames)

    print(f"[OK] {split_name}: {seq_name} | frames: {len(frames)}")

    return seq_name


# =========================
# SEQMAP
# =========================
def write_seqmap(names, split_name):
    """
    Generates the corresponding seqmap file for each data split.
    """
    names = [n for n in names if n]
    if not names:
        return

    seqmap_dir = os.path.join(
        CONFIG["OUTPUT"],
        "gt",
        "seqmaps"
    )

    os.makedirs(seqmap_dir, exist_ok=True)

    out_path = os.path.join(
        seqmap_dir,
        f"{CONFIG['BENCHMARK']}-{split_name}.txt"
    )

    with open(out_path, "w") as f:
        f.write("name\n")
        for n in names:
            f.write(n + "\n")

    print(f"[OK] Seqmap: Saved to {out_path}")


# =========================
# RUN
# =========================
def run():
    random.seed(CONFIG["RANDOM_SEED"])
    msg = f"Starting data conversion (Seed: {CONFIG['RANDOM_SEED']})..."
    print(msg)

    # ---------------------------------------------------------
    # 1. PROCESS TRAIN & VAL SETS
    # ---------------------------------------------------------
    if os.path.exists(CONFIG["RAW_TRAIN_XML"]):
        train_xmls = [
            f for f in os.listdir(CONFIG["RAW_TRAIN_XML"])
            if f.endswith(".xml")
        ]
        random.shuffle(train_xmls)

        split_idx = int(len(train_xmls) * CONFIG["TRAIN_VAL_SPLIT"])

        train_seqs = []
        val_seqs = []

        for i, x in enumerate(train_xmls):
            split_name = "train" if i < split_idx else "val"
            if split_name == "train":
                continue

            xml_full_path = os.path.join(CONFIG["RAW_TRAIN_XML"], x)
            seq_name = process(xml_full_path, split_name)

            if seq_name:
                if split_name == "train":
                    train_seqs.append(seq_name)
                else:
                    val_seqs.append(seq_name)

        # write_seqmap(train_seqs, "train")
        write_seqmap(val_seqs, "val")
    else:
        print(f"[WARNING] Directory not found: {CONFIG['RAW_TRAIN_XML']}")

    # ---------------------------------------------------------
    # 2. PROCESS TEST SET
    # ---------------------------------------------------------
    if os.path.exists(CONFIG["RAW_TEST_XML"]):
        test_xmls = [
            f for f in os.listdir(CONFIG["RAW_TEST_XML"])
            if f.endswith(".xml")
        ]
        test_seqs = []

        for x in test_xmls:
            xml_full_path = os.path.join(CONFIG["RAW_TEST_XML"], x)
            seq_name = process(xml_full_path, "test")
            if seq_name:
                test_seqs.append(seq_name)

        write_seqmap(test_seqs, "test")
    else:
        print(f"[WARNING] Directory not found: {CONFIG['RAW_TEST_XML']}")


if __name__ == "__main__":
    run()
