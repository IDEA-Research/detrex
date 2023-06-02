import atexit
import bisect
from copy import copy
import multiprocessing as mp
from collections import deque
from copy import deepcopy
import cv2
import torch
import torchvision.transforms.functional as F

import detectron2.data.transforms as T
from detectron2.data import MetadataCatalog
from detectron2.structures import Instances
from detectron2.utils.visualizer import (
    ColorMode, 
    Visualizer,
    _create_text_labels,
    )
from detectron2.utils.video_visualizer import (
    _DetectedInstance,
    VideoVisualizer,
)

class MOTVideoVisualizer(VideoVisualizer):
    
    def draw_instance_track(self, frame, predictions):
        """
        Draw instance-level prediction results on an image.

        Args:
            frame (ndarray): an RGB image of shape (H, W, C), in the range [0, 255].
            predictions (Instances): the output of an instance detection/segmentation
                model. Following fields will be used to draw:
                "pred_boxes", "pred_classes", "scores", "pred_masks" (or "pred_masks_rle").

        Returns:
            output (VisImage): image object with visualizations.
        """
        frame_visualizer = Visualizer(frame, self.metadata)
        num_instances = len(predictions)
        if num_instances == 0:
            return frame_visualizer.output

        boxes = predictions.boxes.numpy() if predictions.has("boxes") else None
        scores = predictions.scores if predictions.has("scores") else None
        classes = predictions.labels.numpy() if predictions.has("labels") else None
        keypoints = predictions.pred_keypoints if predictions.has("pred_keypoints") else None
        colors = predictions.COLOR if predictions.has("COLOR") else [None] * len(predictions)
        periods = predictions.obj_idxes if predictions.has("obj_idxes") else None
        period_threshold = self.metadata.get("period_threshold", -1)
        visibilities = (
            [True] * len(predictions)
            if periods is None
            else [x > period_threshold for x in periods]
        )

        if predictions.has("pred_masks"):
            masks = predictions.pred_masks
            # mask IOU is not yet enabled
            # masks_rles = mask_util.encode(np.asarray(masks.permute(1, 2, 0), order="F"))
            # assert len(masks_rles) == num_instances
        else:
            masks = None

        if not predictions.has("COLOR"):
            if predictions.has("obj_idxes"):
                predictions.ID = predictions.obj_idxes.numpy()
                colors = self._assign_colors_by_id(predictions)
            else:
                # ToDo: clean old assign color method and use a default tracker to assign id
                detected = [
                    _DetectedInstance(classes[i], boxes[i], mask_rle=None, color=colors[i], ttl=8)
                    for i in range(num_instances)
                ]
                colors = self._assign_colors(detected)

        labels = _create_text_labels(classes, scores, self.metadata.get("thing_classes", None))

        if self._instance_mode == ColorMode.IMAGE_BW:
            # any() returns uint8 tensor
            frame_visualizer.output.reset_image(
                frame_visualizer._create_grayscale_image(
                    (masks.any(dim=0) > 0).numpy() if masks is not None else None
                )
            )
            alpha = 0.3
        else:
            alpha = 0.5

        labels = (
            None
            if labels is None
            else [y[0] for y in filter(lambda x: x[1], zip(labels, visibilities))]
        )  # noqa
        assigned_colors = (
            None
            if colors is None
            else [y[0] for y in filter(lambda x: x[1], zip(colors, visibilities))]
        )  # noqa
        frame_visualizer.overlay_instances(
            boxes=None if masks is not None else boxes[visibilities],  # boxes are a bit distracting
            masks=None if masks is None else masks[visibilities],
            labels=labels,
            keypoints=None if keypoints is None else keypoints[visibilities],
            assigned_colors=assigned_colors,
            alpha=alpha,
        )

        return frame_visualizer.output


def filter_predictions_with_area(predictions, area_threshold=100):
    if "track_instances" in predictions:
        preds = predictions["track_instances"]
        wh = preds.boxes[:, 2:4] - preds.boxes[:, 0:2]
        areas = wh[:, 0] * wh[:, 1]
        keep_idxs = areas > area_threshold
        predictions = copy(predictions) # don't modify the original
        predictions["track_instances"] = preds[keep_idxs]
    return predictions
    
def filter_predictions_with_confidence(predictions, confidence_threshold=0.5):
    if "track_instances" in predictions:
        preds = predictions["track_instances"]
        keep_idxs = preds.scores > confidence_threshold
        predictions = copy(predictions) # don't modify the original
        predictions["track_instances"] = preds[keep_idxs]
    return predictions


class VisualizationDemo(object):
    def __init__(
        self,
        model,
        min_size_test=800,
        max_size_test=1333,
        img_format="RGB",
        metadata_dataset="coco_2017_val",
        instance_mode=ColorMode.IMAGE,
        parallel=False,
    ):
        """
        Args:
            cfg (CfgNode):
            instance_mode (ColorMode):
            parallel (bool): whether to run the model in different processes from visualization.
                Useful since the visualization logic can be slow.
        """
        self.metadata = MetadataCatalog.get(
            metadata_dataset if metadata_dataset is not None else "__unused"
        )
        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode

        self.parallel = parallel
        if parallel:
            assert False
        else:
            self.predictor = DefaultPredictor(
                model=model,
                min_size_test=min_size_test,
                max_size_test=max_size_test,
                img_format=img_format,
                metadata_dataset=metadata_dataset,
            )

    def run_on_image(self, image, threshold=0.5):
        """
        Args:
            image (np.ndarray): an image of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.

        Returns:
            predictions (dict): the output of the model.
            vis_output (VisImage): the visualized image output.
        """
        vis_output = None
        predictions = self.predictor(image)
        predictions = filter_predictions_with_confidence(predictions, threshold)
        predictions = filter_predictions_with_area(predictions)
        # Convert image from OpenCV BGR format to Matplotlib RGB format.
        image = image[:, :, ::-1]
        visualizer = Visualizer(image, self.metadata, instance_mode=self.instance_mode)
        if "panoptic_seg" in predictions:
            panoptic_seg, segments_info = predictions["panoptic_seg"]
            vis_output = visualizer.draw_panoptic_seg_predictions(
                panoptic_seg.to(self.cpu_device), segments_info
            )
        else:
            if "sem_seg" in predictions:
                vis_output = visualizer.draw_sem_seg(
                    predictions["sem_seg"].argmax(dim=0).to(self.cpu_device)
                )
            if "instances" in predictions:
                instances = predictions["instances"].to(self.cpu_device)
                vis_output = visualizer.draw_instance_predictions(predictions=instances)
            if "track_instances" in predictions:
                predictions = predictions["track_instances"].to(self.cpu_device)
                vis_output = visualizer.draw_instance_predictions(predictions)

        return predictions, vis_output

    def _frame_from_video(self, video):
        while video.isOpened():
            success, frame = video.read()
            if success:
                yield frame
            else:
                break

    def run_on_video(self, video, threshold=0.5):
        """
        Visualizes predictions on frames of the input video.

        Args:
            video (cv2.VideoCapture): a :class:`VideoCapture` object, whose source can be
                either a webcam or a video file.

        Yields:
            ndarray: BGR visualizations of each video frame.
        """
        video_visualizer = MOTVideoVisualizer(self.metadata, self.instance_mode)

        def process_predictions(frame, predictions, threshold):
            predictions = filter_predictions_with_confidence(predictions, threshold)
            predictions = filter_predictions_with_area(predictions)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if "panoptic_seg" in predictions:
                panoptic_seg, segments_info = predictions["panoptic_seg"]
                vis_frame = video_visualizer.draw_panoptic_seg_predictions(
                    frame, panoptic_seg.to(self.cpu_device), segments_info
                )
            elif "instances" in predictions:
                predictions = predictions["instances"].to(self.cpu_device)
                vis_frame = video_visualizer.draw_instance_predictions(frame, predictions)
            elif "sem_seg" in predictions:
                vis_frame = video_visualizer.draw_sem_seg(
                    frame, predictions["sem_seg"].argmax(dim=0).to(self.cpu_device)
                )
            elif "track_instances" in predictions:
                predictions = predictions["track_instances"].to(self.cpu_device)
                vis_frame = video_visualizer.draw_instance_track(frame, predictions)

            # Converts Matplotlib RGB format to OpenCV BGR format
            vis_frame = cv2.cvtColor(vis_frame.get_image(), cv2.COLOR_RGB2BGR)
            return vis_frame

        frame_gen = self._frame_from_video(video)
        if self.parallel:
            buffer_size = self.predictor.default_buffer_size

            frame_data = deque()

            for cnt, frame in enumerate(frame_gen):
                frame_data.append(frame)
                self.predictor.put(frame)

                if cnt >= buffer_size:
                    frame = frame_data.popleft()
                    predictions = self.predictor.get()
                    yield process_predictions(frame, predictions, threshold)

            while len(frame_data):
                frame = frame_data.popleft()
                predictions = self.predictor.get()
                yield process_predictions(frame, predictions, threshold)
        else:
            for frame in frame_gen:
                yield process_predictions(frame, self.predictor(frame), threshold)


class DefaultPredictor:
    def __init__(
        self,
        model,
        min_size_test=800,
        max_size_test=1536,
        img_format="RGB",
        mean = [0.485, 0.456, 0.406],
        std = [0.229, 0.224, 0.225],
        metadata_dataset="coco_2017_val",
    ):
        self.model = model
        # self.model.eval()
        self.mean = mean
        self.std = std
        
        self.metadata = MetadataCatalog.get(metadata_dataset)

        # checkpointer = DetectionCheckpointer(self.model)
        # checkpointer.load(init_checkpoint)

        # self.aug = T.ResizeShortestEdge([min_size_test, min_size_test], max_size_test)
        self.img_height = min_size_test
        self.img_width = max_size_test

        self.input_format = img_format
        self.track_instances = None
        assert self.input_format in ["RGB", "BGR"], self.input_format

    def __call__(self, original_image):
        """
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).

        Returns:
            predictions (dict):
                the output of the model for one image only.
                See :doc:`/tutorials/models` for details about the format.
        """
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            # Apply pre-processing to image.
            if self.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            self.seq_h, self.seq_w = original_image.shape[:2]
            scale = self.img_height / min(self.seq_h, self.seq_w)
            if max(self.seq_h, self.seq_w) * scale > self.img_width:
                scale = self.img_width / max(self.seq_h, self.seq_w)
            target_h = int(self.seq_h * scale)
            target_w = int(self.seq_w * scale)
            image = cv2.resize(original_image, (target_w, target_h))

            image = F.normalize(F.to_tensor(image), self.mean, self.std)
            image = image.to(self.model.device)
            image = image.unsqueeze(0)

            res = self.model.inference_single_image(image, (height, width), self.track_instances)
            
            self.track_instances = res['track_instances']
            predictions = deepcopy(res)
            if len(predictions['track_instances']):
                scores = predictions['track_instances'].scores.reshape(-1, self.model.g_size)
                keep_idxs = torch.arange(len(predictions['track_instances']), device=scores.device).reshape(-1, self.model.g_size)
                keep_idxs = keep_idxs.gather(1, scores.max(-1)[1].reshape(-1, 1)).reshape(-1)
                predictions['track_instances'] = predictions['track_instances'][keep_idxs]
            
            return predictions
