import os

def file_exists(filepath) -> bool:
    return os.access(filepath, os.R_OK)

def transfer_image(index: int, missed_counter: int):
    src_path = "/home/rovsi/Projects/Vakifbank_Internship/Signature_Detection/KaggleDataSet/images/"+str(index)+".tif"
    #70% of data for training, 15% for validation, 15% for testing
    #545 images for training, 115 for validation, 115 for testing
    if index - missed_counter < 545:
        dest_path = "/home/rovsi/Projects/Vakifbank_Internship/Signature_Detection/Yolo_Dataset/images/Train/"+str(index)+".tif"
    elif index - missed_counter < 660:
        dest_path = "/home/rovsi/Projects/Vakifbank_Internship/Signature_Detection/Yolo_Dataset/images/Val/"+str(index)+".tif"
    else:
        dest_path = "/home/rovsi/Projects/Vakifbank_Internship/Signature_Detection/Yolo_Dataset/images/Test/"+str(index)+".tif"

    #copy and paste the image
    os.system(f"cp {src_path} {dest_path}")

def transfer_label(index: int, missed_counter: int):
    src_path = "/home/rovsi/Projects/Vakifbank_Internship/Signature_Detection/KaggleDataSet/All_Labels_Txt/"+str(index)+".txt"
    yolo_lines = normalize_gt_txt(src_path)
    #70% of data for training, 15% for validation, 15% for testing
    #545 images for training, 115 for validation, 115 for testing
    if index - missed_counter < 545:
        dest_path = "/home/rovsi/Projects/Vakifbank_Internship/Signature_Detection/Yolo_Dataset/labels/Train/"+str(index)+".txt"
    elif index - missed_counter < 660:
        dest_path = "/home/rovsi/Projects/Vakifbank_Internship/Signature_Detection/Yolo_Dataset/labels/Val/"+str(index)+".txt"
    else:
        dest_path = "/home/rovsi/Projects/Vakifbank_Internship/Signature_Detection/Yolo_Dataset/labels/Test/"+str(index)+".txt"

    #paste the normalized labels
    write_to_txt(dest_path, yolo_lines)

def normalize_gt_txt(gt_txt_path: str, class_id: int = 0):
    """
    Reads a GT txt file where each line is:
        x_min y_min x_max y_max
    (commas or spaces; multiple lines allowed)

    Returns a list of YOLO-formatted strings:
        "<class_id> x_center y_center width height"
    where coords are normalized to [0,1] using the image size.
    """

    h, w = 1000, 1000  # hardcoded for now

    yolo_lines = []
    with open(gt_txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            line = line.replace(",", " ")
            parts = [p for p in line.split() if p] # this returns the bounding box coords as x_min, y_min, x_max, y_max yes?
            if len(parts) < 4:
                continue

            x_min, y_min, x_max, y_max = map(float, parts)

            bw = x_max - x_min
            bh = y_max - y_min

            x_center = ((x_min + x_max) / 2.0) / w
            y_center = ((y_min + y_max) / 2.0) / h
            box_w = bw / w
            box_h = bh / h

            yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}")

    return yolo_lines

def write_to_txt(output_path: str, yolo_lines: list):
    with open(output_path, "w") as f:
        for line in yolo_lines:
            f.write(line + "\n")

def construct_dataset():

    missed_counter = 0

    for i in range (801):
        img_path = "/home/rovsi/Projects/Vakifbank_Internship/Signature_Detection/KaggleDataSet/images/"+str(i)+".tif"
        if file_exists(img_path):

            transfer_image(i, missed_counter)
            transfer_label(i, missed_counter)
        else:
            missed_counter += 1
            print(f"Image {i} is missing. Total missed: {missed_counter}")

    print("Dataset construction completed.")
    #Count the number of files in each folder
    train_images_count = len(os.listdir("/home/rovsi/Projects/Vakifbank_Internship/Signature_Detection/Yolo_Dataset/images/Train"))
    val_images_count = len(os.listdir("/home/rovsi/Projects/Vakifbank_Internship/Signature_Detection/Yolo_Dataset/images/Val"))
    test_images_count = len(os.listdir("/home/rovsi/Projects/Vakifbank_Internship/Signature_Detection/Yolo_Dataset/images/Test"))
    print(f"Total training images: {train_images_count}")
    print(f"Total validation images: {val_images_count}")
    print(f"Total testing images: {test_images_count}")

construct_dataset()
# print(normalize_gt_txt("/home/rovsi/Projects/Vakifbank_Internship/Signature_Detection/KaggleDataSet/All_Labels_Txt/24.txt"))
