import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from PIL import Image
import argparse
import time

IMAGE_SIZE = (416, 416)
NUM_CLASSES = 2
CLASS_NAMES = ['enemy', 'enemy_head']
COLORS = ['red', 'blue']

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
    Returns both the preprocessed tensor and the original image array.
    """
    print(f"Processing image: {image_path}")
    try:
        original_image = tf.keras.preprocessing.image.load_img(image_path)
        original_array = tf.keras.preprocessing.image.img_to_array(original_image)

        model_image = tf.keras.preprocessing.image.load_img(image_path, target_size=IMAGE_SIZE)
        model_array = tf.keras.preprocessing.image.img_to_array(model_image)
    except Exception as e:
        print(f"TensorFlow image loading failed: {str(e)}")
        try:
            original_image = Image.open(image_path).convert('RGB')
            original_array = np.array(original_image)

            model_image = original_image.resize(IMAGE_SIZE)
            model_array = np.array(model_image)
        except Exception as e2:
            print(f"PIL image loading also failed: {str(e2)}")
            raise e2

    model_array_normalized = model_array / 255.0
    model_tensor = tf.expand_dims(model_array_normalized, 0)
    
    return model_tensor, original_array

def post_process_detections(y_pred, confidence_threshold=0.4, iou_threshold=0.45):
    """
    Convert model predictions to bounding boxes with NMS.
    Returns boxes in normalized coordinates (0-1).
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

def visualize_detections(image_array, boxes, scores, class_ids, output_path=None, show_plot=True):
    """
    Visualize detection results on an image.
    Image should be in its original size.
    """
    if image_array.max() <= 1.0:
        image_np = (image_array * 255).astype(np.uint8)
    else:
        image_np = image_array.astype(np.uint8)

    height, width = image_np.shape[:2]
    fig_width = 12
    fig_height = fig_width * height / width
    
    fig, ax = plt.subplots(1, figsize=(fig_width, fig_height))
    ax.imshow(image_np)

    print(f"Detection results: {len(boxes)} objects found")

    for i, (box, score, class_id) in enumerate(zip(boxes, scores, class_ids)):
        xmin, ymin, xmax, ymax = box

        xmin *= width
        xmax *= width
        ymin *= height
        ymax *= height

        rect = plt.Rectangle(
            (xmin, ymin), xmax - xmin, ymax - ymin,
            linewidth=2, edgecolor=COLORS[class_id], facecolor='none'
        )
        ax.add_patch(rect)
        
        label = f"{CLASS_NAMES[class_id]}: {score:.2f}"
        ax.text(
            xmin, ymin - 5, label,
            color=COLORS[class_id], fontsize=12, fontweight='bold'
        )

        print(f"  - {label} at [{xmin:.1f}, {ymin:.1f}, {xmax:.1f}, {ymax:.1f}]")

    ax.set_title(f"Detected {len(boxes)} objects: {sum(class_ids == 0)} enemies, {sum(class_ids == 1)} headshots", fontsize=14)
    
    plt.axis('off')
    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        print(f"Detection visualization saved to {output_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close(fig)

def process_directory(model, input_dir, output_dir, confidence_threshold=0.4, iou_threshold=0.45, show_plot=False):
    """
    Process all images in a directory and save the visualization results.
    """
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(input_dir, ext)))
    
    print(f"Found {len(image_files)} images to process")
    
    total_time = 0
    for image_path in image_files:
        try:
            filename = os.path.basename(image_path)
            base_name = os.path.splitext(filename)[0]
            output_path = os.path.join(output_dir, f"{base_name}_detection.png")

            model_tensor, original_array = preprocess_image(image_path)
  
            start_time = time.time()
            prediction = model.predict(model_tensor, verbose=0)[0]
            inference_time = (time.time() - start_time) * 1000
            total_time += inference_time
            
            print(f"Inference completed in {inference_time:.2f} ms")

            boxes, scores, classes = post_process_detections(
                prediction, confidence_threshold, iou_threshold
            )

            visualize_detections(original_array, boxes, scores, classes, output_path, show_plot)
            
        except Exception as e:
            print(f"Error processing image {image_path}: {str(e)}")
            import traceback
            print(traceback.format_exc())

    if len(image_files) > 0:
        avg_time = total_time / len(image_files)
        print(f"Average inference time: {avg_time:.2f} ms per image")

def process_single_image(model, image_path, output_path=None, confidence_threshold=0.4, iou_threshold=0.45, show_plot=True):
    """
    Process a single image and visualize the detection results.
    """
    try:
        model_tensor, original_array = preprocess_image(image_path)
        start_time = time.time()
        prediction = model.predict(model_tensor, verbose=0)[0]
        inference_time = (time.time() - start_time) * 1000
        
        print(f"Inference completed in {inference_time:.2f} ms")
        
        boxes, scores, classes = post_process_detections(
            prediction, confidence_threshold, iou_threshold
        )
        
        if output_path is None and not show_plot:
            base_dir = os.path.dirname(image_path)
            filename = os.path.basename(image_path)
            base_name = os.path.splitext(filename)[0]
            output_path = os.path.join(base_dir, f"{base_name}_detection.png")

        visualize_detections(original_array, boxes, scores, classes, output_path, show_plot)
        
        return boxes, scores, classes
        
    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return None, None, None

def main():
    parser = argparse.ArgumentParser(description='Object Detection Inference Script')
    parser.add_argument('--model', type=str, default='trained_models/custom_detector/final_model.keras',
                      help='Path to the trained model')
    parser.add_argument('--input', type=str, required=True,
                      help='Path to input image or directory of images')
    parser.add_argument('--output', type=str, default=None,
                      help='Path to output directory for results')
    parser.add_argument('--conf_thresh', type=float, default=0.4,
                      help='Confidence threshold for detections')
    parser.add_argument('--iou_thresh', type=float, default=0.45,
                      help='IoU threshold for non-maximum suppression')
    parser.add_argument('--show', action='store_true',
                      help='Show visualization plots')
    
    args = parser.parse_args()

    model = load_model(args.model)
    if os.path.isdir(args.input):
        output_dir = args.output if args.output else os.path.join(args.input, 'detections')
        process_directory(
            model, args.input, output_dir, 
            args.conf_thresh, args.iou_thresh, args.show
        )
    else:
        process_single_image(
            model, args.input, args.output,
            args.conf_thresh, args.iou_thresh, args.show
        )

if __name__ == "__main__":
    main()
    """
        python .\test_model.py --model .\trained_models\custom_detector\final_model.keras --input ..\test1.jpg --conf_thresh 0.4 --iou_thresh 0 --show
    """