import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
import io
import time
import datetime

BATCH_SIZE = 16
IMAGE_SIZE = (416, 416)
NUM_CLASSES = 2  # enemy and enemy_head
EPOCHS = 100
LEARNING_RATE = 1e-4

MODEL_DIR = 'trained_models/custom_detector'
os.makedirs(MODEL_DIR, exist_ok=True)

TRANSCRIPT_FILE = os.path.join(MODEL_DIR, 'training_transcript.txt')

def log_to_transcript(message):
    """Write a message to the transcript file with proper encoding handling."""
    try:
        # Convert any non-ASCII characters to their closest ASCII equivalent or replace with '?'
        message_ascii = str(message).encode('ascii', 'replace').decode('ascii')
        
        with open(TRANSCRIPT_FILE, 'a', encoding='utf-8', errors='replace') as f:
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            f.write(f"[{timestamp}] {message_ascii}\n")      
        print(message)
    except Exception as e:
        print(f"Error writing to transcript: {e}")
        print(f"Original message: {message[:50]}...")
try:
    with open(TRANSCRIPT_FILE, 'w', encoding='utf-8', errors='replace') as f:
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        f.write(f"[{timestamp}] Starting training session\n")
        f.write(f"Model configuration: IMAGE_SIZE={IMAGE_SIZE}, BATCH_SIZE={BATCH_SIZE}, EPOCHS={EPOCHS}, LEARNING_RATE={LEARNING_RATE}\n\n")
except Exception as e:
    print(f"Error initializing transcript file: {e}")

def parse_tfrecord(example_proto):
    """Parse TFRecord file format with simplified anchor assignment."""
    feature_description = {
        'image/encoded': tf.io.FixedLenFeature([], tf.string),
        'image/height': tf.io.FixedLenFeature([], tf.int64),
        'image/width': tf.io.FixedLenFeature([], tf.int64),
        'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
        'image/object/class/label': tf.io.VarLenFeature(tf.int64)
    }
    
    example = tf.io.parse_single_example(example_proto, feature_description)

    image = tf.image.decode_jpeg(example['image/encoded'], channels=3)
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32) / 255.0  # Normalize to [0,1]
    xmin = tf.sparse.to_dense(example['image/object/bbox/xmin'])
    ymin = tf.sparse.to_dense(example['image/object/bbox/ymin'])
    xmax = tf.sparse.to_dense(example['image/object/bbox/xmax'])
    ymax = tf.sparse.to_dense(example['image/object/bbox/ymax'])
  
    class_labels = tf.sparse.to_dense(example['image/object/class/label']) - 1

    width = xmax - xmin
    height = ymax - ymin
    center_x = xmin + width / 2
    center_y = ymin + height / 2
    
    S = 13
    B = 3
    target = tf.zeros((S, S, B * 5 + NUM_CLASSES))
    
    for i in range(tf.shape(xmin)[0]):
        grid_x = tf.cast(center_x[i] * S, tf.int32)
        grid_y = tf.cast(center_y[i] * S, tf.int32)
        
        grid_x = tf.clip_by_value(grid_x, 0, S - 1)
        grid_y = tf.clip_by_value(grid_y, 0, S - 1)
        
        rel_x = center_x[i] * S - tf.cast(grid_x, tf.float32)
        rel_y = center_y[i] * S - tf.cast(grid_y, tf.float32)
        
        # rule-based anchor assignment based on box size (small -> large)
        w = width[i]
        h = height[i]
        box_size = w * h
        
        if box_size < 0.1:
            anchor_idx = 0
        elif box_size < 0.25:
            anchor_idx = 1
        else:
            anchor_idx = 2
        
        # If class_id is 1 (enemy_head), prefer the first anchor (small objects)
        if tf.equal(class_labels[i], 1):
            anchor_idx = 0

        box_offset = anchor_idx * 5

        if grid_x >= 0 and grid_x < S and grid_y >= 0 and grid_y < S:
            target = tf.tensor_scatter_nd_update(
                target, 
                [[grid_y, grid_x, box_offset + 4]], 
                [1.0]
            )

            target = tf.tensor_scatter_nd_update(
                target, 
                [[grid_y, grid_x, box_offset], 
                 [grid_y, grid_x, box_offset + 1],
                 [grid_y, grid_x, box_offset + 2], 
                 [grid_y, grid_x, box_offset + 3]],
                [rel_x, rel_y, width[i], height[i]]
            )

            class_idx = B * 5 + tf.cast(class_labels[i], tf.int32)
            target = tf.tensor_scatter_nd_update(
                target, 
                [[grid_y, grid_x, class_idx]],
                [1.0]
            )
    
    return image, target

def create_dataset(tfrecord_path, is_training=True):
    """Create a TensorFlow dataset from TFRecord files."""
    try:
        dataset = tf.data.TFRecordDataset(tfrecord_path, buffer_size=262144)
        dataset = dataset.map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
        
        if is_training:
            dataset = dataset.map(augment_data, num_parallel_calls=tf.data.AUTOTUNE)
            dataset = dataset.shuffle(buffer_size=1000)
        
        dataset = dataset.batch(BATCH_SIZE)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    except Exception as e:
        log_to_transcript(f"Error creating dataset from {tfrecord_path}: {str(e)}")
        raise e

def augment_data(image, target):
    """Apply data augmentation to the training data."""
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    
    # Random flip (horizontal)
    flip = tf.random.uniform(shape=[], minval=0, maxval=1) > 0.5
    if flip:
        image = tf.image.flip_left_right(image)
        # We also need to update the target accordingly
    
    return image, target

def create_improved_model():
    """Create an improved detection model with FPN"""
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(416, 416, 3),
        include_top=False,
        weights='imagenet'
    )

    for layer in base_model.layers[-30:]:
        layer.trainable = True

    c3 = base_model.get_layer('block_6_expand_relu').output  # 52x52
    c4 = base_model.get_layer('block_13_expand_relu').output  # 26x26
    c5 = base_model.output  # 13x13
    
    p5 = tf.keras.layers.Conv2D(256, 1, padding='same')(c5)
    p4 = tf.keras.layers.Add()([
        tf.keras.layers.UpSampling2D(size=(2, 2))(p5),
        tf.keras.layers.Conv2D(256, 1, padding='same')(c4)
    ])
    p3 = tf.keras.layers.Add()([
        tf.keras.layers.UpSampling2D(size=(2, 2))(p4),
        tf.keras.layers.Conv2D(256, 1, padding='same')(c3)
    ])
    
    S = 13
    B = 3

    p5 = tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu')(p5)
    p5 = tf.keras.layers.BatchNormalization()(p5)
  
    output = tf.keras.layers.Conv2D(B * 5 + NUM_CLASSES, 1, activation=None)(p5)

    model = tf.keras.Model(
        inputs=base_model.input, 
        outputs=output
    )
    
    return model

ANCHORS = [
    [0.05, 0.07],  # specifically for enemy heads
    [0.15, 0.20],  # for distant enemies
    [0.35, 0.50]   # for regular enemies
]

def yolo_loss(y_true, y_pred):
    """Custom YOLO-style loss function with improved numerical stability."""
    S = 13
    B = 3

    lambda_coord = 5.0
    lambda_noobj = 0.5

    y_pred = tf.clip_by_value(y_pred, -1e10, 1e10)
    
    total_xy_loss = 0.0
    total_wh_loss = 0.0
    total_obj_loss = 0.0
    total_noobj_loss = 0.0
    total_class_loss = 0.0
    
    epsilon = 1e-7
    
    for b in range(B):
        box_offset = b * 5
        
        true_xy = y_true[..., box_offset:box_offset+2]
        pred_xy = y_pred[..., box_offset:box_offset+2]
        
        true_wh = y_true[..., box_offset+2:box_offset+4]
        pred_wh = y_pred[..., box_offset+2:box_offset+4]
        
        true_conf = y_true[..., box_offset+4:box_offset+5]
        pred_conf = y_pred[..., box_offset+4:box_offset+5]

        object_mask = true_conf

        xy_diff = true_xy - pred_xy
        xy_loss = tf.reduce_sum(
            object_mask * tf.square(xy_diff)
        )

        safe_true_wh = tf.maximum(true_wh, epsilon)
        safe_pred_wh = tf.maximum(pred_wh, epsilon)

        wh_diff = tf.sqrt(safe_true_wh) - tf.sqrt(safe_pred_wh)
        wh_loss = tf.reduce_sum(
            object_mask * tf.square(wh_diff)
        )

        obj_loss = tf.reduce_sum(
            object_mask * tf.square(true_conf - pred_conf)
        )

        noobj_loss = tf.reduce_sum(
            (1 - object_mask) * tf.square(true_conf - pred_conf)
        )

        total_xy_loss += xy_loss
        total_wh_loss += wh_loss
        total_obj_loss += obj_loss
        total_noobj_loss += noobj_loss
    
    class_start_idx = B * 5
    true_class = y_true[..., class_start_idx:]
    pred_class = y_pred[..., class_start_idx:]
    
    max_conf = tf.reduce_max(
        tf.reshape(y_true[..., 4:B*5:5], [-1, S, S, B]), 
        axis=-1,
        keepdims=True
    )
    
    class_loss = tf.reduce_sum(
        max_conf * tf.square(true_class - pred_class)
    )
    
    total_class_loss = class_loss
    
    total_loss = (
        lambda_coord * (total_xy_loss + total_wh_loss) + 
        total_obj_loss + 
        lambda_noobj * total_noobj_loss + 
        total_class_loss
    )
    
    batch_size = tf.cast(tf.shape(y_true)[0], tf.float32)
    normalized_loss = total_loss / (batch_size + epsilon)

    normalized_loss = tf.where(
        tf.math.is_nan(normalized_loss),
        1e6,
        normalized_loss
    )

    tf.summary.scalar('xy_loss', total_xy_loss / batch_size)
    tf.summary.scalar('wh_loss', total_wh_loss / batch_size)
    tf.summary.scalar('obj_loss', total_obj_loss / batch_size)
    tf.summary.scalar('noobj_loss', total_noobj_loss / batch_size)
    tf.summary.scalar('class_loss', total_class_loss / batch_size)
    
    return normalized_loss

class TrainingProgressCallback(tf.keras.callbacks.Callback):
    def __init__(self, log_frequency=5):
        super(TrainingProgressCallback, self).__init__()
        self.log_frequency = log_frequency
        self.batch_logs = []
        self.epoch_summaries = []
        
    def on_train_begin(self, logs=None):
        log_to_transcript("=== Training Started ===")

        total_params = self.model.count_params()
        trainable_params = sum([K.count_params(w) for w in self.model.trainable_weights])
        non_trainable_params = total_params - trainable_params
        
        log_to_transcript(f"Model structure:")
        log_to_transcript(f"Total layers: {len(self.model.layers)}")
        log_to_transcript(f"Total parameters: {total_params:,}")
        log_to_transcript(f"Trainable parameters: {trainable_params:,}")
        log_to_transcript(f"Non-trainable parameters: {non_trainable_params:,}")
        
    def on_batch_end(self, batch, logs=None):
        self.batch_logs.append(logs)
        if batch % self.log_frequency == 0:
            loss = logs.get('loss')
            log_to_transcript(f"Batch {batch} - Loss: {loss:.6f}")
    
    def on_epoch_end(self, epoch, logs=None):
        epoch_loss = logs.get('loss')
        val_loss = logs.get('val_loss')
        
        try:
            if hasattr(self.model.optimizer, '_learning_rate'):
                lr = float(tf.keras.backend.get_value(self.model.optimizer._learning_rate))
            elif hasattr(self.model.optimizer, 'lr'):
                lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
            else:
                lr = 0.0
                log_to_transcript("Warning: Could not retrieve learning rate from optimizer")
        except Exception as e:
            lr = 0.0
            log_to_transcript(f"Error retrieving learning rate: {str(e)}")
        
        # Log epoch summary
        epoch_summary = (
            f"=== Epoch {epoch+1}/{self.params['epochs']} Summary ===\n"
            f"Training Loss: {epoch_loss:.6f}\n"
            f"Validation Loss: {val_loss:.6f}\n"
            f"Learning Rate: {lr:.8f}"
        )
        log_to_transcript(epoch_summary)
        self.epoch_summaries.append(epoch_summary)

        self.batch_logs = []
    
    def on_train_end(self, logs=None):
        log_to_transcript("=== Training Completed ===")

        if hasattr(self.model, 'history') and self.model.history is not None:
            try:
                val_losses = self.model.history.history.get('val_loss', [])
                if val_losses:
                    best_epoch = np.argmin(val_losses)
                    log_to_transcript(f"Best epoch: {best_epoch + 1}")
                    log_to_transcript(f"Best validation loss: {val_losses[best_epoch]:.6f}")
            except Exception as e:
                log_to_transcript(f"Could not determine best epoch: {str(e)}")

class NanLossCallback(tf.keras.callbacks.Callback):
    """Callback that terminates training when NaN loss is encountered."""
    
    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        loss = logs.get('loss')
        if loss is not None and (np.isnan(loss) or np.isinf(loss)):
            print(f"\n\nNaN or Inf loss detected: {loss} at batch {batch}, terminating training.")
            try:
                if hasattr(self.model.optimizer, '_learning_rate'):
                    lr = float(tf.keras.backend.get_value(self.model.optimizer._learning_rate))
                elif hasattr(self.model.optimizer, 'lr'):
                    lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
                else:
                    lr = "unknown"
            except Exception:
                lr = "unknown"

            log_to_transcript(f"NaN/Inf loss detected at batch {batch}. Current learning rate: {lr}")
            log_to_transcript("This usually indicates numerical instability. Check model parameters and learning rate.")

            for i, layer in enumerate(self.model.layers):
                if len(layer.weights) > 0:
                    weights = layer.weights[0].numpy()
                    log_to_transcript(f"Layer {i}: {layer.name} - "
                                      f"min weight: {np.min(weights)}, "
                                      f"max weight: {np.max(weights)}, "
                                      f"mean: {np.mean(weights)}, "
                                      f"std: {np.std(weights)}")
            self.model.stop_training = True

class DetectionVisualizationCallback(tf.keras.callbacks.Callback):
    def __init__(self, validation_dataset, log_dir, max_samples=3, frequency=2):
        super(DetectionVisualizationCallback, self).__init__()
        self.validation_dataset = validation_dataset
        self.log_dir = log_dir
        self.max_samples = max_samples
        self.frequency = frequency
        self.class_names = ['enemy', 'enemy_head']
        
    def _numpy_target_to_boxes(self, target):
        """Convert the target grid format to bounding boxes for visualization."""
        S = 13
        B = 3
        
        boxes = []
        scores = []
        classes = []
        
        for row in range(S):
            for col in range(S):
                max_confidence = 0
                best_box = None
                best_class_id = None

                for b in range(B):
                    box_idx = b * 5
                    confidence = target[row, col, box_idx + 4]
                    
                    if confidence > max_confidence:
                        max_confidence = confidence

                        x = target[row, col, box_idx + 0]
                        y = target[row, col, box_idx + 1]
                        w = target[row, col, box_idx + 2]
                        h = target[row, col, box_idx + 3]
     
                        abs_x = (col + x) / S
                        abs_y = (row + y) / S

                        xmin = abs_x - w / 2
                        ymin = abs_y - h / 2
                        xmax = abs_x + w / 2
                        ymax = abs_y + h / 2
                        
                        xmin = max(0, min(1, xmin))
                        ymin = max(0, min(1, ymin))
                        xmax = max(0, min(1, xmax))
                        ymax = max(0, min(1, ymax))
                        
                        best_box = [xmin, ymin, xmax, ymax]
                
                if max_confidence > 0.2:
                    # Get class predictions
                    class_scores = target[row, col, B*5:]
                    class_id = np.argmax(class_scores)
                    class_score = class_scores[class_id]
                    
                    if class_score > 0.2:
                        boxes.append(best_box)
                        scores.append(max_confidence * class_score)
                        classes.append(class_id)
        
        return np.array(boxes), np.array(scores), np.array(classes)
    
    def _numpy_post_process(self, prediction):
        """Convert the prediction grid format to bounding boxes with NMS."""
        S = 13
        B = 3

        all_boxes = []
        all_scores = []
        all_classes = []

        for row in range(S):
            for col in range(S):
                for b in range(B):
                    box_idx = b * 5
                    x = prediction[row, col, box_idx + 0]
                    y = prediction[row, col, box_idx + 1]
                    w = prediction[row, col, box_idx + 2]
                    h = prediction[row, col, box_idx + 3]
                    confidence = prediction[row, col, box_idx + 4]
  
                    if confidence < 0.2:
                        continue
                    
                    abs_x = (col + x) / S
                    abs_y = (row + y) / S

                    xmin = abs_x - w / 2
                    ymin = abs_y - h / 2
                    xmax = abs_x + w / 2
                    ymax = abs_y + h / 2

                    xmin = max(0, min(1, xmin))
                    ymin = max(0, min(1, ymin))
                    xmax = max(0, min(1, xmax))
                    ymax = max(0, min(1, ymax))

                    class_scores = prediction[row, col, B*5:]
                    class_id = np.argmax(class_scores)
                    class_score = class_scores[class_id]

                    score = confidence * class_score
                    
                    if score >= 0.2:
                        all_boxes.append([xmin, ymin, xmax, ymax])
                        all_scores.append(score)
                        all_classes.append(class_id)
        
        if len(all_boxes) > 0:
            all_boxes = np.array(all_boxes)
            all_scores = np.array(all_scores)
            all_classes = np.array(all_classes)

            boxes_tensor = tf.convert_to_tensor(all_boxes, dtype=tf.float32)
            scores_tensor = tf.convert_to_tensor(all_scores, dtype=tf.float32)
            
            selected_indices = tf.image.non_max_suppression(
                boxes_tensor, scores_tensor, max_output_size=20, 
                iou_threshold=0.5
            ).numpy()
            
            selected_boxes = all_boxes[selected_indices]
            selected_scores = all_scores[selected_indices]
            selected_classes = all_classes[selected_indices]
            
            return selected_boxes, selected_scores, selected_classes
        else:
            return np.array([]), np.array([]), np.array([])
    
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.frequency != 0:
            return
            
        log_to_transcript(f"Generating detection visualizations for epoch {epoch+1}...")

        epoch_dir = os.path.join(self.log_dir, f"epoch_{epoch+1}")
        os.makedirs(epoch_dir, exist_ok=True)

        for batch_idx, (images, targets) in enumerate(self.validation_dataset.take(1)):
            for i in range(min(self.max_samples, images.shape[0])):
                try:
                    image = images[i]
                    target = targets[i]

                    prediction = self.model.predict(tf.expand_dims(image, 0))[0]

                    np_pred = prediction.numpy() if hasattr(prediction, 'numpy') else prediction
                    np_target = target.numpy() if hasattr(target, 'numpy') else target
                    np_image = image.numpy() if hasattr(image, 'numpy') else image

                    true_boxes, true_scores, true_classes = self._numpy_target_to_boxes(np_target)
                    pred_boxes, pred_scores, pred_classes = self._numpy_post_process(np_pred)

                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
                    colors = ['red', 'blue']  # enemy, enemy_head
                    image_np = (np_image * 255).astype(np.uint8)

                    ax1.imshow(image_np)
                    ax1.set_title('Ground Truth', fontsize=14)
                    
                    for box, score, class_id in zip(true_boxes, true_scores, true_classes):
                        xmin, ymin, xmax, ymax = box
                        height, width = image_np.shape[:2]

                        xmin *= width
                        xmax *= width
                        ymin *= height
                        ymax *= height

                        rect = plt.Rectangle(
                            (xmin, ymin), xmax - xmin, ymax - ymin,
                            linewidth=2, edgecolor=colors[int(class_id)], facecolor='none'
                        )
                        ax1.add_patch(rect)

                        label = f"{self.class_names[int(class_id)]}: {score:.2f}"
                        ax1.text(
                            xmin, ymin - 5, label,
                            color=colors[int(class_id)], fontsize=12, fontweight='bold'
                        )

                    ax2.imshow(image_np)
                    ax2.set_title('Model Predictions', fontsize=14)
                    
                    for box, score, class_id in zip(pred_boxes, pred_scores, pred_classes):
                        xmin, ymin, xmax, ymax = box
                        height, width = image_np.shape[:2]

                        xmin *= width
                        xmax *= width
                        ymin *= height
                        ymax *= height

                        rect = plt.Rectangle(
                            (xmin, ymin), xmax - xmin, ymax - ymin,
                            linewidth=2, edgecolor=colors[int(class_id)], facecolor='none'
                        )
                        ax2.add_patch(rect)
                        label = f"{self.class_names[int(class_id)]}: {score:.2f}"
                        ax2.text(
                            xmin, ymin - 5, label,
                            color=colors[int(class_id)], fontsize=12, fontweight='bold'
                        )
                    
                    ax1.axis('off')
                    ax2.axis('off')
                    plt.tight_layout()

                    viz_path = os.path.join(epoch_dir, f'sample_{i+1}.png')
                    plt.savefig(viz_path)
                    plt.close(fig)
                    
                    log_to_transcript(f"Saved visualization to {viz_path}")
                except Exception as e:
                    log_to_transcript(f"Error generating visualization for image {i}: {str(e)}")
                    import traceback
                    log_to_transcript(traceback.format_exc())

            break

def count_dataset_size(dataset):
    """Count the number of batches in a dataset without consuming it."""
    try:
        count_dataset = dataset.map(lambda x, y: 1)
        count_dataset = count_dataset.reduce(0, lambda x, y: x + y)
        return int(count_dataset.numpy())
    except Exception as e:
        log_to_transcript(f"Error counting dataset size: {str(e)}")
        count = 0
        for _ in dataset.take(10):
            count += 1
        return count * 10

def train_model(train_tfrecord, val_tfrecord):
    log_to_transcript(f"Loading data from {train_tfrecord} and {val_tfrecord}")
    
    viz_dir = os.path.join(MODEL_DIR, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)

    train_dataset = create_dataset(train_tfrecord, is_training=True)
    val_dataset = create_dataset(val_tfrecord, is_training=False)

    log_to_transcript("Estimating dataset sizes...")
    
    try:

        train_count_dataset = tf.data.TFRecordDataset(train_tfrecord)
        val_count_dataset = tf.data.TFRecordDataset(val_tfrecord)
        
        train_count = sum(1 for _ in train_count_dataset)
        val_count = sum(1 for _ in val_count_dataset)
        
        train_batches = (train_count + BATCH_SIZE - 1) // BATCH_SIZE
        val_batches = (val_count + BATCH_SIZE - 1) // BATCH_SIZE
        
        log_to_transcript(f"Train dataset size: ~{train_batches} batches (approx. {train_count} examples)")
        log_to_transcript(f"Validation dataset size: ~{val_batches} batches (approx. {val_count} examples)")
    except Exception as e:
        log_to_transcript(f"Error estimating dataset sizes: {str(e)}")
        log_to_transcript("Continuing with training...")

    log_to_transcript("Creating model...")
    model = create_improved_model()
    
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE * 0.3,
        clipnorm=1.0,
    )

    model.compile(
        optimizer=optimizer,
        loss=yolo_loss
    )
    
    checkpoint_path = os.path.join(MODEL_DIR, "model_checkpoint.weights.h5")
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path,
        save_best_only=True,
        save_weights_only=True,
        monitor='val_loss',
        mode='min'
    )
    
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=os.path.join(MODEL_DIR, "logs"),
        update_freq='epoch'
    )
    
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )

    progress_callback = TrainingProgressCallback(log_frequency=5)
    nan_callback = NanLossCallback()
    visualization_callback = DetectionVisualizationCallback(
        validation_dataset=val_dataset,
        log_dir=viz_dir,
        max_samples=3,
        frequency=5
    )

    log_to_transcript("Starting model training...")
    start_time = time.time()
    
    try:
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=EPOCHS,
            callbacks=[
                checkpoint_callback,
                tensorboard_callback,
                early_stopping,
                reduce_lr,
                progress_callback,
                nan_callback,
                visualization_callback
            ],
            steps_per_epoch=min(100, train_batches) if train_batches else None,
            validation_steps=min(20, val_batches) if val_batches else None
        )
        
        training_time = (time.time() - start_time) / 60.0
        log_to_transcript(f"Model training completed in {training_time:.2f} minutes")
        
        model_save_path = os.path.join(MODEL_DIR, "final_model.keras")
        model.save(model_save_path)
        log_to_transcript(f"Model saved to {model_save_path}")
        
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper right')
        
        # Save the plot
        plot_path = os.path.join(MODEL_DIR, 'training_history.png')
        plt.savefig(plot_path)
        log_to_transcript(f"Training history plot saved to {plot_path}")
        
        return model, history
        
    except Exception as e:
        log_to_transcript(f"Error during training: {str(e)}")
        raise e
    

def main():
    """Main function to train and test the model."""
    log_to_transcript("=== Starting Valorany Detector ===")
    train_tfrecord = "../tf_files/train.record"
    val_tfrecord = "../tf_files/valid.record"
    sample_image_path = "sample_images/1.jpg"
    
    log_to_transcript(f"Training data: {train_tfrecord}")
    log_to_transcript(f"Validation data: {val_tfrecord}")
    
    if not os.path.exists(train_tfrecord):
        log_to_transcript(f"ERROR: Training data file not found at {train_tfrecord}")
        return
    
    if not os.path.exists(val_tfrecord):
        log_to_transcript(f"ERROR: Validation data file not found at {val_tfrecord}")
        return
    
    model, history = train_model(train_tfrecord, val_tfrecord)
    
    log_to_transcript("=== Valorant Detector Completed ===")

if __name__ == "__main__":
    from tensorflow.keras import backend as K
    main()