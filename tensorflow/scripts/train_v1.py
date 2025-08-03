import os
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
import time

BATCH_SIZE = 8
IMAGE_HEIGHT = 416
IMAGE_WIDTH = 416
NUM_CLASSES = 3
EPOCHS = 5
LEARNING_RATE =  0.0001
CHECKPOINT_PATH = "valorant_detector_model/checkpoints"
MODEL_SAVE_PATH = "valorant_detector_model/saved_model"
MAX_BOXES = 9

os.makedirs(CHECKPOINT_PATH, exist_ok=True)
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

def parse_tfrecord_fn(example):
    feature_description = {
        'image/height': tf.io.FixedLenFeature([], tf.int64),
        'image/width': tf.io.FixedLenFeature([], tf.int64),
        'image/filename': tf.io.FixedLenFeature([], tf.string),
        'image/source_id': tf.io.FixedLenFeature([], tf.string),
        'image/encoded': tf.io.FixedLenFeature([], tf.string),
        'image/format': tf.io.FixedLenFeature([], tf.string),
        'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
        'image/object/class/text': tf.io.VarLenFeature(tf.string),
        'image/object/class/label': tf.io.VarLenFeature(tf.int64),
    }
    
    example = tf.io.parse_single_example(example, feature_description)
    
    image = tf.image.decode_jpeg(example['image/encoded'], channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [IMAGE_HEIGHT, IMAGE_WIDTH])
    
    xmins = tf.sparse.to_dense(example['image/object/bbox/xmin'])
    xmaxs = tf.sparse.to_dense(example['image/object/bbox/xmax'])
    ymins = tf.sparse.to_dense(example['image/object/bbox/ymin'])
    ymaxs = tf.sparse.to_dense(example['image/object/bbox/ymax'])
    
    height = tf.cast(IMAGE_HEIGHT, tf.float32)
    width = tf.cast(IMAGE_WIDTH, tf.float32)
    
    # shape [num_boxes, 4]
    num_boxes = tf.shape(xmins)[0]
    boxes = tf.stack([
        ymins * height,
        xmins * width,
        ymaxs * height,
        xmaxs * width
    ], axis=1)
    
    # no boxes
    boxes = tf.cond(
        tf.equal(num_boxes, 0),
        lambda: tf.zeros([1, 4], dtype=tf.float32),
        lambda: boxes
    )
    
    class_labels = tf.sparse.to_dense(example['image/object/class/label'])
    class_labels = tf.cond(
        tf.equal(num_boxes, 0),
        lambda: tf.zeros([1], dtype=tf.int64),
        lambda: class_labels
    )
    
    class_one_hot = tf.one_hot(class_labels, NUM_CLASSES)

    def pad_or_truncate(tensor, max_size, padded_shape):
        tensor_shape = tf.shape(tensor)
        tensor_size = tensor_shape[0]
        
        def pad_tensor():
            padding = [[0, max_size - tensor_size]] + [[0, 0]] * (len(padded_shape) - 1)
            return tf.pad(tensor, padding)
        
        def truncate_tensor():
            return tensor[:max_size]
        
        return tf.cond(
            tf.less(tensor_size, max_size),
            pad_tensor,
            truncate_tensor
        )
    
    boxes = pad_or_truncate(boxes, MAX_BOXES, [MAX_BOXES, 4])
    class_one_hot = pad_or_truncate(class_one_hot, MAX_BOXES, [MAX_BOXES, NUM_CLASSES])

    return image, {'boxes': boxes, 'classes': class_one_hot, 'num_boxes': tf.minimum(num_boxes, MAX_BOXES)}

def load_dataset(tfrecord_path, batch_size=BATCH_SIZE):
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    dataset = dataset.map(parse_tfrecord_fn, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(buffer_size=256)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

def build_detection_model(input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3)):
    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape
    )
    
    base_model.trainable = False
    features = base_model.output
    
    conv = layers.Conv2D(256, kernel_size=3, padding='same', activation='relu')(features)
    conv = layers.Conv2D(256, kernel_size=3, padding='same', activation='relu')(conv)
    
    num_anchors = 5
    box_output = layers.Conv2D(num_anchors * 4, kernel_size=1, padding='same')(conv)
    box_output = layers.Reshape((-1, 4))(box_output)
    
    class_output = layers.Conv2D(num_anchors * NUM_CLASSES, kernel_size=1, padding='same')(conv)
    class_output = layers.Reshape((-1, NUM_CLASSES))(class_output)
    class_output = layers.Activation('softmax')(class_output)
    
    model = models.Model(
        inputs=base_model.input,
        outputs={
            'boxes': box_output,
            'classes': class_output
        }
    )
    
    return model

class DetectionLoss:
    def __init__(self):
        self.box_loss_scale = 1.0
        self.cls_loss_scale = 1.0
        
    def match_predictions_to_targets(self, y_true, y_pred):
        """Match predicted boxes to ground truth boxes using IoU"""
        batch_size = tf.shape(y_true['boxes'])[0]
        num_gt_boxes = tf.shape(y_true['boxes'])[1]  # MAX_BOXES (9)
        num_pred_boxes = tf.shape(y_pred['boxes'])[1]  # predicted boxes
        
        matched_indices = []
        matched_gt_boxes = []
        matched_pred_boxes = []
        matched_gt_classes = []
        matched_pred_classes = []
        valid_matches = []
        
        for b in range(batch_size):
            gt_boxes = y_true['boxes'][b]  #[MAX_BOXES, 4]
            gt_classes = y_true['classes'][b]  # [MAX_BOXES, NUM_CLASSES]
            num_valid_boxes = y_true['num_boxes'][b]
            
            pred_boxes = y_pred['boxes'][b]  # [num_pred_boxes, 4]
            pred_classes = y_pred['classes'][b]  # [num_pred_boxes, NUM_CLASSES]
            
            valid_gt_boxes = gt_boxes[:num_valid_boxes]
            valid_gt_classes = gt_classes[:num_valid_boxes]
            
            sampled_indices = tf.random.uniform(
                shape=[tf.minimum(num_valid_boxes, MAX_BOXES), 5],
                minval=0,
                maxval=num_pred_boxes,
                dtype=tf.int32
            )
            
            for i in range(tf.minimum(num_valid_boxes, MAX_BOXES)):
                for j in range(5):  # Sample 5 predictions per ground truth TODO: Need to research better way to do this
                    idx = sampled_indices[i, j]
                    matched_indices.append(tf.stack([b, i, idx]))
                    matched_gt_boxes.append(valid_gt_boxes[i])
                    matched_pred_boxes.append(pred_boxes[idx])
                    matched_gt_classes.append(valid_gt_classes[i])
                    matched_pred_classes.append(pred_classes[idx])
                    valid_matches.append(tf.constant(1.0))
        
        if len(matched_indices) > 0:
            matched_indices = tf.stack(matched_indices)
            matched_gt_boxes = tf.stack(matched_gt_boxes)
            matched_pred_boxes = tf.stack(matched_pred_boxes)
            matched_gt_classes = tf.stack(matched_gt_classes)
            matched_pred_classes = tf.stack(matched_pred_classes)
            valid_matches = tf.stack(valid_matches)
        else:
            matched_indices = tf.zeros([0, 3], dtype=tf.int32)
            matched_gt_boxes = tf.zeros([0, 4], dtype=tf.float32)
            matched_pred_boxes = tf.zeros([0, 4], dtype=tf.float32)
            matched_gt_classes = tf.zeros([0, NUM_CLASSES], dtype=tf.float32)
            matched_pred_classes = tf.zeros([0, NUM_CLASSES], dtype=tf.float32)
            valid_matches = tf.zeros([0], dtype=tf.float32)
        
        return {
            'indices': matched_indices,
            'gt_boxes': matched_gt_boxes,
            'pred_boxes': matched_pred_boxes,
            'gt_classes': matched_gt_classes,
            'pred_classes': matched_pred_classes,
            'valid': valid_matches
        }
    
    def focal_loss(self, y_true, y_pred):
        """Focal loss for classification"""
        alpha = 0.25
        gamma = 2.0

        y_pred = tf.clip_by_value(y_pred, 1e-8, 1.0 - 1e-8)

        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = y_true * tf.math.pow(1.0 - y_pred, gamma) * alpha
        focal = weight * cross_entropy
        
        return tf.reduce_sum(focal, axis=-1)
    
    def smooth_l1_loss(self, y_true, y_pred):
        """Smooth L1 loss for box regression"""
        diff = y_true - y_pred
        abs_diff = tf.abs(diff)
        smooth_l1 = tf.where(abs_diff < 1.0, 
                             0.5 * tf.square(diff),
                             abs_diff - 0.5)
        
        return tf.reduce_sum(smooth_l1, axis=-1)
    
    def __call__(self, y_true, y_pred):
        matches = self.match_predictions_to_targets(y_true, y_pred)
        
        if tf.equal(tf.shape(matches['valid'])[0], 0):
            return tf.constant(0.0), tf.constant(0.0), tf.constant(0.0)
        
        box_loss = self.smooth_l1_loss(matches['gt_boxes'], matches['pred_boxes'])
        box_loss = tf.reduce_mean(box_loss) * self.box_loss_scale
        
        cls_loss = self.focal_loss(matches['gt_classes'], matches['pred_classes'])
        cls_loss = tf.reduce_mean(cls_loss) * self.cls_loss_scale
        
        total_loss = box_loss + cls_loss
        
        return total_loss, box_loss, cls_loss


def train_model(train_dataset, val_dataset, epochs=EPOCHS):

    model = build_detection_model()
    loss_fn = DetectionLoss()
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    train_loss_results = []
    val_loss_results = []

    ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, CHECKPOINT_PATH, max_to_keep=5)

    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print(f"Restored from {ckpt_manager.latest_checkpoint}")

    for epoch in range(epochs):
        start_time = time.time()
        epoch_train_loss = 0
        epoch_train_box_loss = 0
        epoch_train_cls_loss = 0
        step = 0

        for x_batch, y_batch in train_dataset:
            with tf.GradientTape() as tape:
                # Forward 
                y_pred = model(x_batch, training=True)
                total_loss, box_loss, cls_loss = loss_fn(y_batch, y_pred)

            gradients = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            
            epoch_train_loss += total_loss
            epoch_train_box_loss += box_loss
            epoch_train_cls_loss += cls_loss
            step += 1

            if step % 5 == 0:
                print(f"Step {step}, Loss: {total_loss:.4f}, Box Loss: {box_loss:.4f}, Class Loss: {cls_loss:.4f}")

        if step > 0:
            epoch_train_loss /= step
            epoch_train_box_loss /= step
            epoch_train_cls_loss /= step
        
        epoch_val_loss = 0
        epoch_val_box_loss = 0
        epoch_val_cls_loss = 0
        val_step = 0
        
        for x_val, y_val in val_dataset:
            y_pred_val = model(x_val, training=False)
            val_total_loss, val_box_loss, val_cls_loss = loss_fn(y_val, y_pred_val)
            
            epoch_val_loss += val_total_loss
            epoch_val_box_loss += val_box_loss
            epoch_val_cls_loss += val_cls_loss
            val_step += 1
        
        if val_step > 0:
            epoch_val_loss /= val_step
            epoch_val_box_loss /= val_step
            epoch_val_cls_loss /= val_step

        ckpt_save_path = ckpt_manager.save()
        print(f"Saved checkpoint for epoch {epoch+1} at {ckpt_save_path}")
        train_loss_results.append(epoch_train_loss)
        val_loss_results.append(epoch_val_loss)

        print(f"Epoch {epoch+1}/{epochs}, "
              f"Time: {time.time() - start_time:.2f}s, "
              f"Train Loss: {epoch_train_loss:.4f}, "
              f"Val Loss: {epoch_val_loss:.4f}")

    export_path = os.path.join(MODEL_SAVE_PATH, "saved_model")
    tf.saved_model.save(model, export_path)
    print(f"Model saved to {MODEL_SAVE_PATH}")

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss_results)
    plt.plot(val_loss_results)
    plt.title('Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.savefig('training_history.png')
    
    return model

def main():
    print("Loading datasets...")
    train_dataset = load_dataset("../tf_files/train.record")
    val_dataset = load_dataset("../tf_files/test.record")
    
    print("Building and training model...")
    model = train_model(train_dataset, val_dataset)
    
    print(f"Model saved to {MODEL_SAVE_PATH}")
    print("Training complete!")

if __name__ == "__main__":
    main()