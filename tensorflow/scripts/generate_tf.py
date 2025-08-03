import os
import sys
import pandas as pd
import tensorflow as tf
from object_detection.utils import dataset_util
from collections import namedtuple

"""
    matches with label_map.pbtxt
"""
def class_to_int(row_label):
    if row_label == 'enemy':
        return 1
    elif row_label == 'enemy_head':
        return 2
    else:
        print(f"couldn't convert {row_label}")
        return None
    
def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    grouped = df.groupby(group)
    matched = []
    for filename, group_df in grouped:
        matched.append(data(filename, group_df))
    return matched

def create_tf(group, path):
    with tf.io.gfile.GFile(os.path.join(path, group.filename), 'rb') as fid:
        encoded_jpg = fid.read()

    filename = group.filename.encode('utf8')
    img_format = b'jpg'
    wid = int(group.object.iloc[0]['width'])
    height = int(group.object.iloc[0]['height'])

    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    class_str = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin']/wid)
        xmaxs.append(row['xmax']/wid)
        ymins.append(row['ymin']/height)
        ymaxs.append(row['ymax']/height)
        class_str.append(row['class'].encode('utf8'))
        classes.append(class_to_int(row['class']))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(wid),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(img_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(class_str),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example

def main():
    try:
        file_path = sys.argv[1]
        img_dir = sys.argv[2]
        output_path = sys.argv[3]
        data = pd.read_csv(file_path)
        print(f"Loaded {len(data)} annotation rows")
        grouped = split(data, 'filename')
        with tf.io.TFRecordWriter(output_path) as writer:
            for group in grouped:
                print(f"Processing image: {group.filename}")
                tf_example = create_tf(group, img_dir)
                writer.write(tf_example.SerializeToString())
        print(f"TFRecord file created at: {output_path}")
    except Exception as e:
        print(e)

if __name__ == "__main__":
    main()