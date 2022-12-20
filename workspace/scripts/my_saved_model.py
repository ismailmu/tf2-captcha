import os
import pathlib
import tensorflow as tf
import time
import numpy as np
import matplotlib.pyplot as plt
import warnings

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.utils import ops as utils_ops

from os import listdir
from os.path import isfile, join
from PIL import Image

warnings.filterwarnings('ignore')           # Suppress Matplotlib warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)
tf.get_logger().setLevel('ERROR')           # Suppress TensorFlow logging (2)

# Enable GPU dynamic memory allocation
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    print('gpu : ', gpu)
    tf.config.experimental.set_memory_growth(gpu, True)

def get_images():
    image_paths = [f for f in listdir(FILE_PATH) if isfile(join(FILE_PATH, f))]
    return image_paths

BASE_PATH= 'C:/Users/eeng/Documents/src/python/tensorflow2/custom-object/captcha/workspace'
FILE_PATH =  BASE_PATH + '/images/validation/'
IMAGE_PATHS = get_images()
PATH_TO_LABELS = BASE_PATH + '/annotations/label_map.pbtxt'
PATH_TO_MODEL_DIR = BASE_PATH + '/models/my_ssd_resnet50_v1_fpn/exported-models/saved_model/'
THRESHOLD = .30

print('Loading model...', end='')
start_time = time.time()

# Load saved model and build the detection function
detect_fn = tf.saved_model.load(PATH_TO_MODEL_DIR)

end_time = time.time()
elapsed_time = end_time - start_time
print('Done! Took {} seconds'.format(elapsed_time))

category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                    use_display_name=True)
#print(category_index)

def load_image_into_numpy_array(path):
    return np.array(Image.open(path))

#cara deteksi urutan object dengan membandingkan xmin terkecil ke terbesar
def sort_result(detections):
    classes = [cls for cls in detections['detection_classes'][detections['detection_scores'] >= THRESHOLD]]
    classes = [category_index.get(cls)['name'] for cls in classes]
    #print(classes)
    boxes =  [bx for bx in detections['detection_boxes'][detections['detection_scores'] >= THRESHOLD]]
    xmin = [bx[1] for bx in boxes]
    #print(xmin)
    data = {}
    for x in range(len(xmin)):
        data[xmin[x]] = (classes[x],detections['detection_scores'][x])

    for x in sorted(data.keys()):
        print(data[x],end='\r\n')

#pantek IMAGE_PATHS untuk testing
#IMAGE_PATHS = ['59210eef-8671-11eb-85c1-005056c00008.png']
for image_path in IMAGE_PATHS:

    print('')
    print('Running inference for {}... '.format(image_path), end='\r\n')

    file = FILE_PATH + image_path
    #print(file)
    image_np = load_image_into_numpy_array(file)

    # Things to try:
    # Flip horizontally
    # image_np = np.fliplr(image_np).copy()

    # Convert image to grayscale
    image_np = np.tile(
         np.mean(image_np, 2, keepdims=True), (1, 1, 3)).astype(np.uint8)

    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image_np)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    # input_tensor = np.expand_dims(image_np, 0)
    detections = detect_fn(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                   for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    image_np_with_detections = image_np.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        detections['detection_boxes'],
        detections['detection_classes'],
        detections['detection_scores'],
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=200,
        min_score_thresh=THRESHOLD,
        agnostic_mode=False
    )
    sort_result(detections)
    plt.figure()
    plt.imshow(image_np_with_detections)
    plt.savefig(join(FILE_PATH,"result",image_path))
    print('Done',end='')

# sphinx_gallery_thumbnail_number = 2