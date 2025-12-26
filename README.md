# Signature_Detection_On_Documents

## Usefull commands:


## Relevant Directories:
All yolo dataset folders consist of train, validation and test folders alongside the data.yaml file.
### Yolo_Dataset_tif: 
    The dataset acquired from Tobacco800 with .tif documents including signatures and labeling, formatted for yolov11.
    Total images 775.
    Source: https://www.kaggle.com/datasets/mohamedadlyi/signature-detection-dataset?resource=download
### Yolo_Dataset_jpg:
    The dataset acquired from "signatures-xc8up" from roboflow with .jpg documents including signatures and labeling, formatted for yolov11.
    Total images 368:
    Source: https://universe.roboflow.com/roboflow-100/signatures-xc8up/dataset/2
### Yolo_Dataset_mixed:
    Combined dataset of jpg and tif datasets above. 
    Total images 1143.

### runs/signature_detect/sig_yolo11s_img896_1:
    First model trained on .tif dataset only. 
    Metrics:
        Results for mixed Test dataset:
                         Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 10/10 2.5it/s 4.1s
                           all        152        184      0.916      0.853      0.912      0.703
        Results for jpg Test dataset:
                        Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.5it/s 1.2s
                          all         37         54      0.851      0.648       0.74      0.471
        Results for tif Test dataset:
                        Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 8/8 2.4it/s 3.3s
                          all        115        130      0.918      0.954       0.98        0.8

### runs/signature_detect/sig_yolo11s_img896_2:
    Second model acquired with fine-tuning the first model sig_yolo11s_img896_1 on the mixed dataset. 
    Metrics:
        Results for mixed Test dataset:
                         Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 10/10 1.2it/s 8.4s
                           all        152        184      0.943       0.88      0.944      0.721
        Results for jpg Test dataset:
                         Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 3/3 2.3it/s 1.3s
                           all         37         54      0.917      0.704      0.835       0.56
        Results for tif Test dataset:
                         Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 8/8 2.4it/s 3.3s
                           all        115        130       0.95      0.954      0.979      0.778

### runs/detect:
    Certain validation and predictions for test files. 

## Relevant scripts:

### train_yolo: 
    Used to parse the command from terminal and train the yolo_model. 
    This "wrapper" is used instead of plain 
    '''Bash 
        yolo train ...
    
    for enhanced control and management, such as an extra auto validation after training.

    example usage: python train_yolo.py   --data /Yolo_Dataset_mixed/data.yaml   --model /runs/signature_detect/sig_yolo11s_img896_2/weights/best.pt   --imgsz 896   --epochs 30   --patience 10   --device 0   --name sig_yolo11s_img896_2_contd   --project runs/signature_detect --lr0 0.001 --lrf 0.1

    generalized usage: train_yolo.py [-h] --data DATA [--model MODEL] [--imgsz IMGSZ] [--epochs EPOCHS] [--batch BATCH] [--patience PATIENCE] [--seed SEED]
                     [--device DEVICE] [--workers WORKERS] [--name NAME] [--project PROJECT] [--lr0 LR0] [--lrf LRF]

options:
  -h, --help           show this help message and exit
  --data DATA          Path to data.yaml
  --model MODEL        e.g., yolo11s.pt or yolo11n.pt
  --imgsz IMGSZ
  --epochs EPOCHS
  --batch BATCH        -1 lets Ultralytics auto-pick; else set manually
  --patience PATIENCE
  --seed SEED
  --device DEVICE      GPU id like '0' or 'cpu'
  --workers WORKERS
  --name NAME
  --project PROJECT
  --lr0 LR0            Initial learning rate (None = Ultralytics default)
  --lrf LRF            Final LR fraction (None = Ultralytics default)

### viz_labels_yolo_format:
    This script is used to generate yolo-like vizualizations of labels on images, given the label and image directories.
    It ahs been tested valid on .jpg, .png and .tif files.

    usage: viz_labels_yolo_format.py [-h] --images IMAGES --labels LABELS [--data DATA] [--recursive] [--sample SAMPLE] [--start START] [--show_empty]

options:
  -h, --help       show this help message and exit
  --images IMAGES  Images directory (jpg/png/...)
  --labels LABELS  Labels directory (YOLO txt files)
  --data DATA      Optional data.yaml (for class names)
  --recursive      Search images recursively
  --sample SAMPLE  If >0, view random sample of this many images
  --start START    Start from a specific image filename (e.g. 001.jpg)
  --show_empty     Include images with no labels (default: yes)
