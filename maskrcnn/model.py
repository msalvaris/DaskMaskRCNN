from torchvision import transforms as T
from maskrcnn_benchmark.modeling.roi_heads.mask_head.inference import Masker
import cv2
from maskrcnn_benchmark.utils import cv2_util

from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.modeling.detector import build_detection_model
import torch

CATEGORIES = [
    "__background",
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


def build_transforms(cfg, min_image_size=800):
    # we are loading images with OpenCV, so we don't need to convert them
    # to BGR, they are already! So all we need to do is to normalize
    # by 255 if we want to convert to BGR255 format, or flip the channels
    # if we want it to be in RGB in [0-1] range.
    if cfg.INPUT.TO_BGR255:
        to_bgr_transform = T.Lambda(lambda x: x * 255)
    else:
        to_bgr_transform = T.Lambda(lambda x: x[[2, 1, 0]])

    normalize_transform = T.Normalize(
        mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD
    )

    transform = T.Compose(
        [
            T.ToPILImage(),
            T.Resize(min_image_size),
            T.ToTensor(),
            to_bgr_transform,
            normalize_transform,
        ]
    )
    return transform


def select_top_predictions(predictions, confidence_threshold=0.7):
    scores = predictions.get_field("scores")
    keep = torch.nonzero(scores > confidence_threshold).squeeze(1)
    predictions = predictions[keep]
    scores = predictions.get_field("scores")
    _, idx = scores.sort(0, descending=True)
    return predictions[idx]


def overlay_boxes(image, predictions):
    """
    Adds the predicted boxes on top of the image

    Arguments:
        image (np.ndarray): an image as returned by OpenCV
        predictions (BoxList): the result of the computation by the model.
            It should contain the field `labels`.
    """
    labels = predictions.get_field("labels")
    boxes = predictions.bbox

    colors = compute_colors_for_labels(labels).tolist()

    for box, color in zip(boxes, colors):
        box = box.to(torch.int64)
        top_left, bottom_right = box[:2].tolist(), box[2:].tolist()
        image = cv2.rectangle(
            image, tuple(top_left), tuple(bottom_right), tuple(color), 1
        )

    return image


def overlay_mask(image, predictions):
    """
    Adds the instances contours for each predicted object.
    Each label has a different color.

    Arguments:
        image (np.ndarray): an image as returned by OpenCV
        predictions (BoxList): the result of the computation by the model.
            It should contain the field `mask` and `labels`.
    """
    masks = predictions.get_field("mask").numpy()
    labels = predictions.get_field("labels")

    colors = compute_colors_for_labels(labels).tolist()

    for mask, color in zip(masks, colors):
        thresh = mask[0, :, :, None]
        contours, hierarchy = cv2_util.findContours(
            thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        image = cv2.drawContours(image, contours, -1, color, 3)

    composite = image

    return composite


def compute_colors_for_labels(
    labels, palette=torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
):
    """
    Simple function that adds fixed colors depending on the class
    """
    colors = labels[:, None] * palette
    colors = (colors % 255).numpy().astype("uint8")
    return colors


def overlay_class_names(image, predictions):
    """
    Adds detected class names and scores in the positions defined by the
    top-left corner of the predicted bounding box

    Arguments:
        image (np.ndarray): an image as returned by OpenCV
        predictions (BoxList): the result of the computation by the model.
            It should contain the field `scores` and `labels`.
    """
    scores = predictions.get_field("scores").tolist()
    labels = predictions.get_field("labels").tolist()
    labels = [CATEGORIES[i] for i in labels]
    boxes = predictions.bbox

    template = "{}: {:.2f}"
    for box, score, label in zip(boxes, scores, labels):
        x, y = box[:2]
        s = template.format(label, score)
        cv2.putText(image, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return image


def create_preprocessing(cfg):
    transforms = build_transforms(cfg)

    def preprocess(image):
        return transforms(image)

    return preprocess


def load_model(cfg, cuda=True):
    device = torch.device("cuda" if cuda else "cpu")
    cfg = cfg.clone()
    model = build_detection_model(cfg)
    model.eval()
    model.to(device)

    save_dir = cfg.OUTPUT_DIR
    checkpointer = DetectronCheckpointer(cfg, model, save_dir=save_dir)
    _ = checkpointer.load(cfg.MODEL.WEIGHT)
    return model


def score_batch(model, img_batch, size_divisibility=32, cuda=True):
    device = torch.device("cuda" if cuda else "cpu")
    cpu = torch.device("cpu")
    image_list = to_image_list(img_batch, size_divisibility)
    image_list = image_list.to(device)
    with torch.no_grad():
        output = model(image_list)
        return [o.to(cpu) for o in output]


def add_annotations(orig_image, prediction, mask_on=True):
    show_mask_heatmaps = False
    mask_threshold = -1 if show_mask_heatmaps else 0.5
    masker = Masker(threshold=mask_threshold, padding=1)
    # reshape prediction (a BoxList) into the original image size
    height, width = orig_image.shape[:-1]
    prediction = prediction.resize((width, height))
    if prediction.has_field("mask"):
        # if we have masks, paste the masks in the right position
        # in the image, as defined by the bounding boxes
        masks = prediction.get_field("mask")
        # always single image is passed at a time
        masks = masker([masks], [prediction])[0]
        prediction.add_field("mask", masks)
    top_preds = select_top_predictions(prediction)
    result = orig_image.copy()
    result = overlay_boxes(result, top_preds)
    if mask_on:  # cfg.MODEL.MASK_ON:
        result = overlay_mask(result, top_preds)
    return overlay_class_names(result, top_preds)


def clean_gpu_mem():
    torch.cuda.empty_cache()