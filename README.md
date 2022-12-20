# tf2-captcha

## Create train data:
python generate_tfrecord.py -x [PATH_TO_IMAGES_FOLDER]/train -l [PATH_TO_ANNOTATIONS_FOLDER]/label_map.pbtxt -o [PATH_TO_ANNOTATIONS_FOLDER]/train.record

## Create test data:
python generate_tfrecord.py -x [PATH_TO_IMAGES_FOLDER]/test -l [PATH_TO_ANNOTATIONS_FOLDER]/label_map.pbtxt -o [PATH_TO_ANNOTATIONS_FOLDER]/test.record


## generate tfrecord
python scripts/generate_tfrecord.py -x C:\Users\eeng\Documents\src\python\tensorflow2\custom-object\captcha\workspace\images\train -l C:\Users\eeng\Documents\src\python\tensorflow2\custom-object\captcha\workspace\annotations/label_map.pbtxt -o C:\Users\eeng\Documents\src\python\tensorflow2\custom-object\captcha\workspace\annotations/train.record

python scripts/generate_tfrecord.py -x C:\Users\eeng\Documents\src\python\tensorflow2\custom-object\captcha\workspace\images\test -l C:\Users\eeng\Documents\src\python\tensorflow2\custom-object\captcha\workspace\annotations/label_map.pbtxt -o C:\Users\eeng\Documents\src\python\tensorflow2\custom-object\captcha\workspace\annotations/test.record


##training
python scripts/model_main_tf2.py --model_dir=models/my_ssd_resnet50_v1_fpn --pipeline_config_path=models/my_ssd_resnet50_v1_fpn/pipeline.config


##export training
python scripts/exporter_main_v2.py --input_type image_tensor --pipeline_config_path models/my_ssd_resnet50_v1_fpn/pipeline.config --trained_checkpoint_dir models/my_ssd_resnet50_v1_fpn/ --output_directory models/my_ssd_resnet50_v1_fpn/exported-models


##run hasil training
python scripts/my_saved_model.py