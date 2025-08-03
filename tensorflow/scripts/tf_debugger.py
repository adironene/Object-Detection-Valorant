import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import sys

def inspect_tfrecord(tfrecord_path):

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
    
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    parsed_dataset = dataset.map(lambda example: tf.io.parse_single_example(
        example, feature_description))
    
    file = open('../tf_files_description.txt', 'w')
    box_counts = []
    image_sizes = []
    class_distribution = {}
    
    file.write(f"Analyzing TFRecord file: {tfrecord_path}\n")
    file.write("=" * 20)
    file.write("\n")
    
    count = 0
    for example in parsed_dataset:
        height = example['image/height'].numpy()
        width = example['image/width'].numpy()
        filename = example['image/filename'].numpy().decode('utf-8')
        
        xmins = tf.sparse.to_dense(example['image/object/bbox/xmin']).numpy()
        xmaxs = tf.sparse.to_dense(example['image/object/bbox/xmax']).numpy()
        ymins = tf.sparse.to_dense(example['image/object/bbox/ymin']).numpy()
        ymaxs = tf.sparse.to_dense(example['image/object/bbox/ymax']).numpy()
        
        class_labels = tf.sparse.to_dense(example['image/object/class/label']).numpy()
        class_texts = tf.sparse.to_dense(example['image/object/class/text']).numpy()
        
        for cls in class_labels:
            cls_key = int(cls)
            if cls_key in class_distribution:
                class_distribution[cls_key] += 1
            else:
                class_distribution[cls_key] = 1
        
        num_boxes = len(xmins)
        box_counts.append(num_boxes)
        image_sizes.append((height, width))
        
        if count < 5:
            file.write(f"Example {count+1}:\n")
            file.write(f"  Filename: {filename}\n")
            file.write(f"  Image size: {width}x{height}\n")
            file.write(f"  Number of boxes: {num_boxes}\n")
            
            class_count = {}
            for i, cls in enumerate(class_labels):
                cls_name = class_texts[i].decode('utf-8')
                if cls_name in class_count:
                    class_count[cls_name] += 1
                else:
                    class_count[cls_name] = 1
            file.write(f"  Classes: {class_count}\n")
            
            for i in range(min(3, num_boxes)):
                file.write(f"  Box {i+1}: [{xmins[i]:.4f}, {ymins[i]:.4f}, {xmaxs[i]:.4f}, {ymaxs[i]:.4f}], Class: {class_texts[i].decode('utf-8')}\n")
            file.write("-" * 40)
            file.write("\n")
        
        count += 1
    
    file.write("\nDataset Summary:\n")
    file.write(f"Total number of examples: {count}\n")
    file.write(f"Average number of boxes per image: {np.mean(box_counts):.2f}\n")
    file.write(f"Min number of boxes: {np.min(box_counts)}\n")
    file.write(f"Max number of boxes: {np.max(box_counts)}\n")
    
    file.write("\nClass distribution:\n")
    for cls_id, count in class_distribution.items():
        file.write(f"  Class {cls_id}: {count} instances\n")
    file.close()
    return box_counts, image_sizes, class_distribution

def main():
    if len(sys.argv) < 2:
        print("Usage: include tf path as param")
        sys.exit(1)
    
    tfrecord_path = sys.argv[1]
    inspect_tfrecord(tfrecord_path)

if __name__ == "__main__":
    main()