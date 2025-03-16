import os
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

# Requires TensorFlow 2.0+
from distutils.version import LooseVersion

assert LooseVersion(tf.__version__) >= LooseVersion("2.0")

tf.compat.v1.disable_eager_execution()

import logging

logging.getLogger("absl").setLevel("ERROR")
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


def load_weights(self, filepath, by_name=False, exclude=None):
    """Modified version of the corresponding Keras function with
    the addition of multi-GPU support and the ability to exclude
    some layers from loading.
    exclude: list of layer names to exclude
    """
    import h5py
    from tensorflow.python.keras.saving import hdf5_format

    if exclude:
        by_name = True

    if h5py is None:
        raise ImportError("`load_weights` requires h5py.")
    with h5py.File(filepath, mode="r") as f:
        if "layer_names" not in f.attrs and "model_weights" in f:
            f = f["model_weights"]

        # In multi-GPU training, we wrap the model. Get layers
        # of the inner model because they have the weights.
        keras_model = self.keras_model
        layers = (
            keras_model.inner_model.layers
            if hasattr(keras_model, "inner_model")
            else keras_model.layers
        )

        # Exclude some layers
        if exclude:
            layers = filter(lambda l: l.name not in exclude, layers)

        if by_name:
            hdf5_format.load_weights_from_hdf5_group_by_name(f, layers)
        else:
            hdf5_format.load_weights_from_hdf5_group(f, layers)

    # Update the log directory
    self.set_log_dir(filepath)


def detect(self, images, verbose=0):
    """Runs the detection pipeline.
    images: List of images, potentially of different sizes.
    Returns a list of dicts, one dict per image. The dict contains:
    rois: [N, (y1, x1, y2, x2)] detection bounding boxes
    class_ids: [N] int class IDs
    scores: [N] float probability scores for the class IDs
    masks: [H, W, N] instance binary masks
    """
    assert self.mode == "inference", "Create model in inference mode."
    assert (
        len(images) == self.config.BATCH_SIZE
    ), "len(images) must be equal to BATCH_SIZE"

    if verbose:
        log("Processing {} images".format(len(images)))
        for image in images:
            log("image", image)

    # Mold inputs to format expected by the neural network
    molded_images, image_metas, windows = self.mold_inputs(images)

    # Validate image sizes
    # All images in a batch MUST be of the same size
    image_shape = molded_images[0].shape
    for g in molded_images[1:]:
        assert (
            g.shape == image_shape
        ), "After resizing, all images must have the same size. Check IMAGE_RESIZE_MODE and image sizes."

    # Anchors
    anchors = self.get_anchors(image_shape)
    # Duplicate across the batch dimension because Keras requires it
    # TODO: can this be optimized to avoid duplicating the anchors?
    anchors = np.broadcast_to(anchors, (self.config.BATCH_SIZE,) + anchors.shape)

    if verbose:
        log("molded_images", molded_images)
        log("image_metas", image_metas)
        log("anchors", anchors)
    # Run object detection
    detections, _, _, mrcnn_mask, _, _, _ = self.keras_model.predict(
        [molded_images, image_metas, anchors], verbose=0
    )
    # Process detections
    results = []
    for i, image in enumerate(images):
        final_rois, final_class_ids, final_scores, final_masks = self.unmold_detections(
            detections[i],
            mrcnn_mask[i],
            image.shape,
            molded_images[i].shape,
            windows[i],
        )
        results.append(
            {
                "rois": final_rois,
                "class_ids": final_class_ids,
                "scores": final_scores,
                "masks": final_masks,
            }
        )
    return results


def log(text, array=None):
    """Prints a text message. And, optionally, if a Numpy array is provided it
    prints it's shape, min, and max values.
    """
    if array is not None:
        text = text.ljust(25)
        text += "shape: {:20}  ".format(str(array.shape))
        if array.size:
            text += "min: {:10.5f}  max: {:10.5f}".format(array.min(), array.max())
        else:
            text += "min: {:10}  max: {:10}".format("", "")
        text += "  {}".format(array.dtype)
    print(text)


class MaskRCNN(object):
    """Encapsulates the Mask RCNN model functionality.
    The actual Keras model is in the keras_model property.
    """

    def __init__(self, mode, config, model_dir):
        """
        mode: Either "training" or "inference"
        config: A Sub-class of the Config class
        model_dir: Directory to save training logs and trained weights
        """
        assert mode in ["training", "inference"]
        self.mode = mode
        self.config = config
        self.model_dir = model_dir
        self.set_log_dir()
        self.keras_model = self.build(mode=mode, config=config)

    def build(self, mode, config):
        """Build Mask R-CNN architecture.
        input_shape: The shape of the input image.
        mode: Either "training" or "inference". The inputs and
            outputs of the model differ accordingly.
        """
        assert mode in ["training", "inference"]

        # Image size must be dividable by 2 multiple times
        h, w = config.IMAGE_SHAPE[:2]
        if h / 2**6 != int(h / 2**6) or w / 2**6 != int(w / 2**6):
            raise Exception(
                "Image size must be dividable by 2 at least 6 times "
                "to avoid fractions when downscaling and upscaling."
                "For example, use 256, 320, 384, 448, 512, ... etc. "
            )

        # Inputs
        input_image = KL.Input(
            shape=[None, None, config.IMAGE_SHAPE[2]], name="input_image"
        )
        input_image_meta = KL.Input(
            shape=[config.IMAGE_META_SIZE], name="input_image_meta"
        )
        if mode == "training":
            # RPN GT
            input_rpn_match = KL.Input(
                shape=[None, 1], name="input_rpn_match", dtype=tf.int32
            )
            input_rpn_bbox = KL.Input(
                shape=[None, 4], name="input_rpn_bbox", dtype=tf.float32
            )

            # Detection GT (class IDs, bounding boxes, and masks)
            # 1. GT Class IDs (zero padded)
            input_gt_class_ids = KL.Input(
                shape=[None], name="input_gt_class_ids", dtype=tf.int32
            )
            # 2. GT Boxes in pixels (zero padded)
            # [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)] in image coordinates
            input_gt_boxes = KL.Input(
                shape=[None, 4], name="input_gt_boxes", dtype=tf.float32
            )
            # Normalize coordinates
            gt_boxes = KL.Lambda(
                lambda x: norm_boxes_graph(x, K.shape(input_image)[1:3])
            )(input_gt_boxes)
            # 3. GT Masks (zero padded)
            # [batch, height, width, MAX_GT_INSTANCES]
            if config.USE_MINI_MASK:
                input_gt_masks = KL.Input(
                    shape=[config.MINI_MASK_SHAPE[0], config.MINI_MASK_SHAPE[1], None],
                    name="input_gt_masks",
                    dtype=bool,
                )
            else:
                input_gt_masks = KL.Input(
                    shape=[config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1], None],
                    name="input_gt_masks",
                    dtype=bool,
                )
        elif mode == "inference":
            # Anchors in normalized coordinates
            input_anchors = KL.Input(shape=[None, 4], name="input_anchors")

        # Build the shared convolutional layers.
        # Bottom-up Layers
        # Returns a list of the last layers of each stage, 5 in total.
        # Don't create the thead (stage 5), so we pick the 4th item in the list.
        if callable(config.BACKBONE):
            _, C2, C3, C4, C5 = config.BACKBONE(
                input_image, stage5=True, train_bn=config.TRAIN_BN
            )
        else:
            _, C2, C3, C4, C5 = resnet_graph(
                input_image, config.BACKBONE, stage5=True, train_bn=config.TRAIN_BN
            )
        # Top-down Layers
        # TODO: add assert to varify feature map sizes match what's in config
        P5 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name="fpn_c5p5")(C5)
        P4 = KL.Add(name="fpn_p4add")(
            [
                KL.UpSampling2D(size=(2, 2), name="fpn_p5upsampled")(P5),
                KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name="fpn_c4p4")(C4),
            ]
        )
        P3 = KL.Add(name="fpn_p3add")(
            [
                KL.UpSampling2D(size=(2, 2), name="fpn_p4upsampled")(P4),
                KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name="fpn_c3p3")(C3),
            ]
        )
        P2 = KL.Add(name="fpn_p2add")(
            [
                KL.UpSampling2D(size=(2, 2), name="fpn_p3upsampled")(P3),
                KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name="fpn_c2p2")(C2),
            ]
        )
        # Attach 3x3 conv to all P layers to get the final feature maps.
        P2 = KL.Conv2D(
            config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p2"
        )(P2)
        P3 = KL.Conv2D(
            config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p3"
        )(P3)
        P4 = KL.Conv2D(
            config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p4"
        )(P4)
        P5 = KL.Conv2D(
            config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p5"
        )(P5)
        # P6 is used for the 5th anchor scale in RPN. Generated by
        # subsampling from P5 with stride of 2.
        P6 = KL.MaxPooling2D(pool_size=(1, 1), strides=2, name="fpn_p6")(P5)

        # Note that P6 is used in RPN, but not in the classifier heads.
        rpn_feature_maps = [P2, P3, P4, P5, P6]
        mrcnn_feature_maps = [P2, P3, P4, P5]

        # Anchors
        if mode == "training":
            anchors = self.get_anchors(config.IMAGE_SHAPE)
            # Duplicate across the batch dimension because Keras requires it
            # TODO: can this be optimized to avoid duplicating the anchors?
            anchors = np.broadcast_to(anchors, (config.BATCH_SIZE,) + anchors.shape)

            # A hack to get around Keras's bad support for constants
            # This class returns a constant layer
            class ConstLayer(tf.keras.layers.Layer):
                def __init__(self, x, name=None):
                    super(ConstLayer, self).__init__(name=name)
                    self.x = tf.Variable(x)

                def call(self, input):
                    return self.x

            anchors = ConstLayer(anchors, name="anchors")(input_image)
        else:
            anchors = input_anchors

        # RPN Model
        rpn = build_rpn_model(
            config.RPN_ANCHOR_STRIDE,
            len(config.RPN_ANCHOR_RATIOS),
            config.TOP_DOWN_PYRAMID_SIZE,
        )
        # Loop through pyramid layers
        layer_outputs = []  # list of lists
        for p in rpn_feature_maps:
            layer_outputs.append(rpn([p]))
        # Concatenate layer outputs
        # Convert from list of lists of level outputs to list of lists
        # of outputs across levels.
        # e.g. [[a1, b1, c1], [a2, b2, c2]] => [[a1, a2], [b1, b2], [c1, c2]]
        output_names = ["rpn_class_logits", "rpn_class", "rpn_bbox"]
        outputs = list(zip(*layer_outputs))
        outputs = [
            KL.Concatenate(axis=1, name=n)(list(o))
            for o, n in zip(outputs, output_names)
        ]

        rpn_class_logits, rpn_class, rpn_bbox = outputs

        # Generate proposals
        # Proposals are [batch, N, (y1, x1, y2, x2)] in normalized coordinates
        # and zero padded.
        proposal_count = (
            config.POST_NMS_ROIS_TRAINING
            if mode == "training"
            else config.POST_NMS_ROIS_INFERENCE
        )
        rpn_rois = ProposalLayer(
            proposal_count=proposal_count,
            nms_threshold=config.RPN_NMS_THRESHOLD,
            name="ROI",
            config=config,
        )([rpn_class, rpn_bbox, anchors])

        if mode == "training":
            # Class ID mask to mark class IDs supported by the dataset the image
            # came from.
            active_class_ids = KL.Lambda(
                lambda x: parse_image_meta_graph(x)["active_class_ids"]
            )(input_image_meta)

            if not config.USE_RPN_ROIS:
                # Ignore predicted ROIs and use ROIs provided as an input.
                input_rois = KL.Input(
                    shape=[config.POST_NMS_ROIS_TRAINING, 4],
                    name="input_roi",
                    dtype=np.int32,
                )
                # Normalize coordinates
                target_rois = KL.Lambda(
                    lambda x: norm_boxes_graph(x, K.shape(input_image)[1:3])
                )(input_rois)
            else:
                target_rois = rpn_rois

            # Generate detection targets
            # Subsamples proposals and generates target outputs for training
            # Note that proposal class IDs, gt_boxes, and gt_masks are zero
            # padded. Equally, returned rois and targets are zero padded.
            rois, target_class_ids, target_bbox, target_mask = DetectionTargetLayer(
                config, name="proposal_targets"
            )([target_rois, input_gt_class_ids, gt_boxes, input_gt_masks])

            # Network Heads
            # TODO: verify that this handles zero padded ROIs
            mrcnn_class_logits, mrcnn_class, mrcnn_bbox = fpn_classifier_graph(
                rois,
                mrcnn_feature_maps,
                input_image_meta,
                config.POOL_SIZE,
                config.NUM_CLASSES,
                train_bn=config.TRAIN_BN,
                fc_layers_size=config.FPN_CLASSIF_FC_LAYERS_SIZE,
            )

            mrcnn_mask = build_fpn_mask_graph(
                rois,
                mrcnn_feature_maps,
                input_image_meta,
                config.MASK_POOL_SIZE,
                config.NUM_CLASSES,
                train_bn=config.TRAIN_BN,
            )

            # TODO: clean up (use tf.identify if necessary)
            output_rois = KL.Lambda(lambda x: x * 1, name="output_rois")(rois)

            # Losses
            rpn_class_loss = KL.Lambda(
                lambda x: rpn_class_loss_graph(*x), name="rpn_class_loss"
            )([input_rpn_match, rpn_class_logits])
            rpn_bbox_loss = KL.Lambda(
                lambda x: rpn_bbox_loss_graph(config, *x), name="rpn_bbox_loss"
            )([input_rpn_bbox, input_rpn_match, rpn_bbox])
            class_loss = KL.Lambda(
                lambda x: mrcnn_class_loss_graph(*x), name="mrcnn_class_loss"
            )([target_class_ids, mrcnn_class_logits, active_class_ids])
            bbox_loss = KL.Lambda(
                lambda x: mrcnn_bbox_loss_graph(*x), name="mrcnn_bbox_loss"
            )([target_bbox, target_class_ids, mrcnn_bbox])
            mask_loss = KL.Lambda(
                lambda x: mrcnn_mask_loss_graph(*x), name="mrcnn_mask_loss"
            )([target_mask, target_class_ids, mrcnn_mask])

            # Model
            inputs = [
                input_image,
                input_image_meta,
                input_rpn_match,
                input_rpn_bbox,
                input_gt_class_ids,
                input_gt_boxes,
                input_gt_masks,
            ]
            if not config.USE_RPN_ROIS:
                inputs.append(input_rois)
            outputs = [
                rpn_class_logits,
                rpn_class,
                rpn_bbox,
                mrcnn_class_logits,
                mrcnn_class,
                mrcnn_bbox,
                mrcnn_mask,
                rpn_rois,
                output_rois,
                rpn_class_loss,
                rpn_bbox_loss,
                class_loss,
                bbox_loss,
                mask_loss,
            ]
            model = KM.Model(inputs, outputs, name="mask_rcnn")
        else:
            # Network Heads
            # Proposal classifier and BBox regressor heads
            mrcnn_class_logits, mrcnn_class, mrcnn_bbox = fpn_classifier_graph(
                rpn_rois,
                mrcnn_feature_maps,
                input_image_meta,
                config.POOL_SIZE,
                config.NUM_CLASSES,
                train_bn=config.TRAIN_BN,
                fc_layers_size=config.FPN_CLASSIF_FC_LAYERS_SIZE,
            )

            # Detections
            # output is [batch, num_detections, (y1, x1, y2, x2, class_id, score)] in
            # normalized coordinates
            detections = DetectionLayer(config, name="mrcnn_detection")(
                [rpn_rois, mrcnn_class, mrcnn_bbox, input_image_meta]
            )

            # Create masks for detections
            detection_boxes = KL.Lambda(lambda x: x[..., :4])(detections)
            mrcnn_mask = build_fpn_mask_graph(
                detection_boxes,
                mrcnn_feature_maps,
                input_image_meta,
                config.MASK_POOL_SIZE,
                config.NUM_CLASSES,
                train_bn=config.TRAIN_BN,
            )

            model = KM.Model(
                [input_image, input_image_meta, input_anchors],
                [
                    detections,
                    mrcnn_class,
                    mrcnn_bbox,
                    mrcnn_mask,
                    rpn_rois,
                    rpn_class,
                    rpn_bbox,
                ],
                name="mask_rcnn",
            )

        # Add multi-GPU support.
        if config.GPU_COUNT > 1:
            from parallel_model import ParallelModel

            model = ParallelModel(model, config.GPU_COUNT)

        return model
