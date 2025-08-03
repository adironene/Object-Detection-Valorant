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
EPOCHS = 0
LEARNING_RATE = 0.0001
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
    
    # keep boxes in normalized form [0-1]
    boxes = tf.stack([
        ymins, xmins, ymaxs, xmaxs
    ], axis=1)
    
    # no boxes
    num_boxes = tf.shape(xmins)[0]
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
    """Simplified model with straightforward architecture"""
    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape
    )
    
    base_model.trainable = False
    features = base_model.output
    
    conv = layers.Conv2D(256, kernel_size=3, padding='same', activation='relu')(features)
    conv = layers.Conv2D(256, kernel_size=3, padding='same', activation='relu')(conv)
    
    num_anchors = 3
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

class SimplifiedDetectionLoss:
    def __init__(self):
        self.box_loss_scale = 0.01
        self.cls_loss_scale = 1.0
    
    def __call__(self, y_true, y_pred):
        y_true_boxes = y_true['boxes']
        y_true_classes = y_true['classes']
        num_boxes = y_true['num_boxes']
        
        y_pred_boxes = y_pred['boxes']
        y_pred_classes = y_pred['classes']
        
        batch_size = tf.shape(y_true_boxes)[0]

        total_box_loss = tf.constant(0.0, dtype=tf.float32)
        total_cls_loss = tf.constant(0.0, dtype=tf.float32)
        num_valid_boxes_total = tf.constant(0, dtype=tf.int32)
        
        for b in range(batch_size):
            n_boxes = tf.cast(num_boxes[b], tf.int32)
            if n_boxes <= 0:
                continue

            valid_gt_boxes = y_true_boxes[b, :n_boxes, :]
            valid_gt_classes = y_true_classes[b, :n_boxes, :]
            

            n_preds = tf.minimum(tf.shape(y_pred_boxes)[1], 100)
            pred_boxes = y_pred_boxes[b, :n_preds, :]
            pred_classes = y_pred_classes[b, :n_preds, :]
            
            for i in range(n_boxes):
                gt_box = valid_gt_boxes[i]
                gt_class = valid_gt_classes[i]
                
                # find closest prediction by L2 distance
                box_diffs = tf.reduce_sum(tf.square(pred_boxes - tf.expand_dims(gt_box, 0)), axis=1)
                closest_idx = tf.argmin(box_diffs)

                closest_box = pred_boxes[closest_idx]
                closest_class = pred_classes[closest_idx]

                # huber/Smooth L1
                box_diff = tf.abs(gt_box - closest_box)
                box_loss = tf.where(
                    box_diff < 1.0,
                    0.5 * tf.square(box_diff),
                    box_diff - 0.5
                )
                box_loss = tf.reduce_sum(box_loss) * self.box_loss_scale

                cls_loss = -tf.reduce_sum(gt_class * tf.math.log(tf.clip_by_value(closest_class, 1e-7, 1.0))) * self.cls_loss_scale
                total_box_loss += box_loss
                total_cls_loss += cls_loss
                num_valid_boxes_total += 1
        
        num_valid_boxes_total = tf.maximum(num_valid_boxes_total, 1)

        avg_box_loss = total_box_loss / tf.cast(num_valid_boxes_total, tf.float32)
        avg_cls_loss = total_cls_loss / tf.cast(num_valid_boxes_total, tf.float32)
        total_loss = avg_box_loss + avg_cls_loss
        
        return total_loss, avg_box_loss, avg_cls_loss

def train_model(train_dataset, val_dataset, epochs=EPOCHS):

    model = build_detection_model()
    loss_fn = SimplifiedDetectionLoss()

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=LEARNING_RATE,
        decay_steps=1000,
        decay_rate=0.9,
        staircase=True
    )
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    train_loss_results = []
    val_loss_results = []

    ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, CHECKPOINT_PATH, max_to_keep=5)

    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print(f"Restored from {ckpt_manager.latest_checkpoint}")

    @tf.function
    def train_step(x, y):
        with tf.GradientTape() as tape:
            y_pred = model(x, training=True)
            total_loss, box_loss, cls_loss = loss_fn(y, y_pred)
            
        gradients = tape.gradient(total_loss, model.trainable_variables)
        gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
        
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        return total_loss, box_loss, cls_loss
    
    @tf.function
    def val_step(x, y):
        y_pred = model(x, training=False)
        total_loss, box_loss, cls_loss = loss_fn(y, y_pred)
        return total_loss, box_loss, cls_loss

    for epoch in range(epochs):
        start_time = time.time()
        
        epoch_train_loss = 0
        epoch_train_box_loss = 0
        epoch_train_cls_loss = 0
        step = 0
        
        for x_batch, y_batch in train_dataset:
            total_loss, box_loss, cls_loss = train_step(x_batch, y_batch)
            
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
        val_step_count = 0
        
        for x_val, y_val in val_dataset:
            val_total_loss, val_box_loss, val_cls_loss = val_step(x_val, y_val)
            
            epoch_val_loss += val_total_loss
            epoch_val_box_loss += val_box_loss
            epoch_val_cls_loss += val_cls_loss
            val_step_count += 1

        if val_step_count > 0:
            epoch_val_loss /= val_step_count
            epoch_val_box_loss /= val_step_count
            epoch_val_cls_loss /= val_step_count

        ckpt_save_path = ckpt_manager.save()
        print(f"Saved checkpoint for epoch {epoch+1} at {ckpt_save_path}")

        train_loss_results.append(epoch_train_loss)
        val_loss_results.append(epoch_val_loss)

        print(f"Epoch {epoch+1}/{epochs}, "
              f"Time: {time.time() - start_time:.2f}s, "
              f"Train Loss: {epoch_train_loss:.4f}, "
              f"Train Box Loss: {epoch_train_box_loss:.4f}, "
              f"Train Cls Loss: {epoch_train_cls_loss:.4f}, "
              f"Val Loss: {epoch_val_loss:.4f}")

        if epoch == 2:
            print("Unfreezing base model for fine-tuning...")
            base_model = None
            for layer in model.layers:
                if isinstance(layer, tf.keras.models.Model):
                    base_model = layer
                    break
            
            if base_model is not None:
                for layer in base_model.layers[-20:]:
                    layer.trainable = True
                print(f"Unfroze last 20 layers of the base model")
            else:
                print("Could not find base model to unfreeze")

            optimizer.learning_rate = lr_schedule.initial_learning_rate * 0.1
            print(f"Reduced learning rate to {optimizer.learning_rate.numpy()}")

    export_path = os.path.join(MODEL_SAVE_PATH, "saved_model")
    tf.saved_model.save(model, export_path)
    print(f"Model saved to {MODEL_SAVE_PATH}")

    plt.figure(figsize=(12, 8))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss_results)
    plt.plot(val_loss_results)
    plt.title('Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.subplot(1, 2, 2)
    plt.plot([loss.numpy() for loss in train_loss_results])
    plt.title('Training Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
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