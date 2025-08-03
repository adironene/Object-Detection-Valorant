import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from PIL import Image
import argparse
import time
from collections import defaultdict
import pandas as pd

IMAGE_SIZE = (416, 416)
NUM_CLASSES = 2
CLASS_NAMES = ['enemy', 'enemy_head']

def load_model(model_path):
    """
    Load the trained model from the specified path.
    """
    print(f"Loading model from {model_path}...")
    try:
        model = tf.keras.models.load_model(
            model_path, 
            compile=False
        )
        print("Model loaded successfully!")
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise e

def preprocess_image(image_path):
    """
    Load and preprocess an image for inference.
    """
    print(f"Processing image: {image_path}")
    try:
        image = tf.keras.preprocessing.image.load_img(image_path, target_size=IMAGE_SIZE)
        image_array = tf.keras.preprocessing.image.img_to_array(image)
    except Exception as e:
        print(f"TensorFlow image loading failed: {str(e)}")
        try:
            image = Image.open(image_path).convert('RGB')
            image = image.resize(IMAGE_SIZE)
            image_array = np.array(image)
        except Exception as e2:
            print(f"PIL image loading also failed: {str(e2)}")
            raise e2
    
    image_array = image_array / 255.0
    image_tensor = tf.expand_dims(image_array, 0)
    
    return image_tensor, image_array

def post_process_detections(y_pred, confidence_threshold=0.4, iou_threshold=0.45):
    """
    Convert model predictions to bounding boxes with NMS.
    """
    S = 13
    B = 3

    all_boxes = []
    all_scores = []
    all_classes = []

    for row in range(S):
        for col in range(S):
            for b in range(B):
                box_idx = b * 5
                x = y_pred[row, col, box_idx + 0]
                y = y_pred[row, col, box_idx + 1]
                w = y_pred[row, col, box_idx + 2]
                h = y_pred[row, col, box_idx + 3]
                confidence = y_pred[row, col, box_idx + 4]

                if confidence < confidence_threshold:
                    continue
                abs_x = (col + x) / S
                abs_y = (row + y) / S
                abs_w = w
                abs_h = h

                xmin = abs_x - abs_w / 2
                ymin = abs_y - abs_h / 2
                xmax = abs_x + abs_w / 2
                ymax = abs_y + abs_h / 2
                
                xmin = max(0, min(1, xmin))
                ymin = max(0, min(1, ymin))
                xmax = max(0, min(1, xmax))
                ymax = max(0, min(1, ymax))

                class_scores = y_pred[row, col, B*5:]
                class_id = np.argmax(class_scores)
                class_score = class_scores[class_id]

                score = confidence * class_score
                
                if score >= confidence_threshold:
                    all_boxes.append([xmin, ymin, xmax, ymax])
                    all_scores.append(score)
                    all_classes.append(class_id)

    if len(all_boxes) > 0:
        all_boxes = np.array(all_boxes)
        all_scores = np.array(all_scores)
        all_classes = np.array(all_classes)
        selected_indices = tf.image.non_max_suppression(
            all_boxes, all_scores, max_output_size=20, 
            iou_threshold=iou_threshold
        ).numpy()
        
        selected_boxes = all_boxes[selected_indices]
        selected_scores = all_scores[selected_indices]
        selected_classes = all_classes[selected_indices]
        
        return selected_boxes, selected_scores, selected_classes
    else:
        return np.array([]), np.array([]), np.array([])

def calculate_iou(box1, box2):
    """
    Calculate IoU between two bounding boxes.
    Each box is in format [xmin, ymin, xmax, ymax]
    """

    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - intersection_area

    if union_area == 0:
        return 0.0

    return intersection_area / union_area

def load_annotations_from_csv(csv_path, image_filename):
    """
    Load annotations from a CSV file for a specific image.
    The CSV format typically includes: filename, width, height, class, xmin, ymin, xmax, ymax
    """
    try:
        df = pd.read_csv(csv_path)
        print(f"CSV columns: {df.columns.tolist()}")
        
        image_df = df[df['filename'] == image_filename]
        
        if len(image_df) == 0:
            print(f"No annotations found for {image_filename}")
            return np.array([]), np.array([])
        
        gt_boxes = []
        gt_classes = []
        
        for _, row in image_df.iterrows():
            if 'xmin' in df.columns and 'ymin' in df.columns:
                xmin = row['xmin']
                ymin = row['ymin']
                xmax = row['xmax']
                ymax = row['ymax']
                
                if 'width' in df.columns and 'height' in df.columns:
                    img_width = row['width']
                    img_height = row['height']

                    xmin = xmin / img_width
                    ymin = ymin / img_height
                    xmax = xmax / img_width
                    ymax = ymax / img_height
                else:

                    pass
                    
            elif 'x' in df.columns and 'y' in df.columns:
                center_x = row['x']
                center_y = row['y']
                box_width = row['width']
                box_height = row['height']
                
                xmin = center_x - box_width / 2
                ymin = center_y - box_height / 2
                xmax = center_x + box_width / 2
                ymax = center_y + box_height / 2
            else:
                print(f"Unknown CSV format. Columns: {df.columns.tolist()}")
                return np.array([]), np.array([])

            if 'class' in df.columns:
                class_name = row['class']
                if class_name in CLASS_NAMES:
                    class_id = CLASS_NAMES.index(class_name)
                else:
                    print(f"Unknown class: {class_name}")
                    continue
            elif 'label' in df.columns:
                class_name = row['label']
                if class_name in CLASS_NAMES:
                    class_id = CLASS_NAMES.index(class_name)
                else:
                    print(f"Unknown label: {class_name}")
                    continue
            elif 'class_id' in df.columns:
                class_id = int(row['class_id'])
            else:
                print("No class information found in CSV")
                continue

            xmin = max(0, min(1, xmin))
            ymin = max(0, min(1, ymin))
            xmax = max(0, min(1, xmax))
            ymax = max(0, min(1, ymax))
            
            gt_boxes.append([xmin, ymin, xmax, ymax])
            gt_classes.append(class_id)
    
    except Exception as e:
        print(f"Error reading CSV file: {str(e)}")
        return np.array([]), np.array([])
    
    return np.array(gt_boxes), np.array(gt_classes)

def evaluate_detections(pred_boxes, pred_classes, pred_scores, gt_boxes, gt_classes, iou_threshold=0.5):
    """
    Evaluate predictions against ground truth using IoU threshold.
    Returns metrics for precision and recall calculation.
    """
    num_gt = len(gt_boxes)
    num_pred = len(pred_boxes)
    gt_matched = np.zeros(num_gt, dtype=bool)

    true_positives = 0
    false_positives = 0

    sorted_indices = np.argsort(pred_scores)[::-1]
    
    for pred_idx in sorted_indices:
        pred_box = pred_boxes[pred_idx]
        pred_class = pred_classes[pred_idx]
        
        best_iou = 0
        best_gt_idx = -1

        for gt_idx in range(num_gt):
            if gt_matched[gt_idx]:
                continue
                
            if gt_classes[gt_idx] != pred_class:
                continue
            
            iou = calculate_iou(pred_box, gt_boxes[gt_idx])
            
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_iou >= iou_threshold:
            true_positives += 1
            gt_matched[best_gt_idx] = True
        else:
            false_positives += 1

    false_negatives = num_gt - np.sum(gt_matched)
    
    return true_positives, false_positives, false_negatives

def evaluate_test_set(model, test_dir, confidence_threshold=0.4, iou_threshold_nms=0.45, iou_threshold_eval=0.5):
    """
    Evaluate the model on the entire test set using CSV annotations.
    """
    csv_path = os.path.join(test_dir, '_annotations.csv')
    
    if not os.path.exists(csv_path):
        print(f"Error: Annotations file not found at {csv_path}")
        return 0, 0, 0
    
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(test_dir, ext)))
    
    print(f"Found {len(image_files)} images in test set")
    
    all_true_positives = 0
    all_false_positives = 0
    all_false_negatives = 0

    class_metrics = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})

    for idx, image_path in enumerate(image_files):
        image_filename = os.path.basename(image_path)
        print(f"\nProcessing image {idx+1}/{len(image_files)}: {image_filename}")
        
        try:
            gt_boxes, gt_classes = load_annotations_from_csv(csv_path, image_filename)
            print(f"Ground truth: {len(gt_boxes)} objects")
            image_tensor, image_array = preprocess_image(image_path)
            prediction = model.predict(image_tensor, verbose=0)[0]
            pred_boxes, pred_scores, pred_classes = post_process_detections(
                prediction, confidence_threshold, iou_threshold_nms
            )
            print(f"Predictions: {len(pred_boxes)} objects")
            tp, fp, fn = evaluate_detections(
                pred_boxes, pred_classes, pred_scores, 
                gt_boxes, gt_classes, iou_threshold_eval
            )
            
            all_true_positives += tp
            all_false_positives += fp
            all_false_negatives += fn
            for class_id in range(NUM_CLASSES):
                gt_mask = gt_classes == class_id
                pred_mask = pred_classes == class_id
                
                if np.any(gt_mask) or np.any(pred_mask):
                    tp_class, fp_class, fn_class = evaluate_detections(
                        pred_boxes[pred_mask], 
                        pred_classes[pred_mask], 
                        pred_scores[pred_mask],
                        gt_boxes[gt_mask], 
                        gt_classes[gt_mask], 
                        iou_threshold_eval
                    )
                    
                    class_metrics[class_id]['tp'] += tp_class
                    class_metrics[class_id]['fp'] += fp_class
                    class_metrics[class_id]['fn'] += fn_class
            
        except Exception as e:
            print(f"Error processing image {image_path}: {str(e)}")
            import traceback
            print(traceback.format_exc())
    precision = all_true_positives / (all_true_positives + all_false_positives + 1e-6)
    recall = all_true_positives / (all_true_positives + all_false_negatives + 1e-6)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-6)
    
    print("\n" + "="*50)
    print("OVERALL EVALUATION RESULTS")
    print("="*50)
    print(f"IoU Threshold: {iou_threshold_eval}")
    print(f"Confidence Threshold: {confidence_threshold}")
    print(f"Total True Positives: {all_true_positives}")
    print(f"Total False Positives: {all_false_positives}")
    print(f"Total False Negatives: {all_false_negatives}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1_score:.4f}")
    
    print("\n" + "="*50)
    print("PER-CLASS EVALUATION RESULTS")
    print("="*50)
    
    for class_id in range(NUM_CLASSES):
        tp = class_metrics[class_id]['tp']
        fp = class_metrics[class_id]['fp']
        fn = class_metrics[class_id]['fn']
        
        if tp + fp + fn > 0:
            class_precision = tp / (tp + fp + 1e-6)
            class_recall = tp / (tp + fn + 1e-6)
            class_f1 = 2 * (class_precision * class_recall) / (class_precision + class_recall + 1e-6)
            
            print(f"\nClass: {CLASS_NAMES[class_id]}")
            print(f"  True Positives: {tp}")
            print(f"  False Positives: {fp}")
            print(f"  False Negatives: {fn}")
            print(f"  Precision: {class_precision:.4f}")
            print(f"  Recall: {class_recall:.4f}")
            print(f"  F1 Score: {class_f1:.4f}")
    
    return precision, recall, f1_score

def main():
    parser = argparse.ArgumentParser(description='Object Detection Inference and Evaluation Script')
    parser.add_argument('--model', type=str, default='trained_models/custom_detector/final_model.keras',
                      help='Path to the trained model')
    parser.add_argument('--test_dir', type=str, default='test',
                      help='Path to test directory containing images and _annotations.csv')
    parser.add_argument('--conf_thresh', type=float, default=0.4,
                      help='Confidence threshold for detections')
    parser.add_argument('--iou_thresh_nms', type=float, default=0.4,
                      help='IoU threshold for non-maximum suppression')
    parser.add_argument('--iou_thresh_eval', type=float, default=0.4,
                      help='IoU threshold for evaluation (default: 0.5)')
    parser.add_argument('--save_results', type=str, default=None,
                      help='Path to save evaluation results')
    
    args = parser.parse_args()
    model = load_model(args.model)
    precision, recall, f1_score = evaluate_test_set(
        model, 
        args.test_dir, 
        args.conf_thresh, 
        args.iou_thresh_nms,
        args.iou_thresh_eval
    )

    if args.save_results:
        with open(args.save_results, 'w') as f:
            f.write(f"IoU Threshold: {args.iou_thresh_eval}\n")
            f.write(f"Confidence Threshold: {args.conf_thresh}\n")
            f.write(f"Precision: {precision:.4f}\n")
            f.write(f"Recall: {recall:.4f}\n")
            f.write(f"F1 Score: {f1_score:.4f}\n")
        print(f"\nResults saved to {args.save_results}")

if __name__ == "__main__":
    main()