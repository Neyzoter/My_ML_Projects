# -*- coding: utf-8 -*-
import os
import scipy.io
import scipy.misc
from matplotlib.pyplot import imshow
from keras import backend as K
from yolo_utils import read_classes,read_anchors,yolo_head,yolo_eval,preprocess_image,generate_colors, draw_boxes
from keras.models import load_model

sess = K.get_session()

class_names = read_classes("model_data/coco_classes.txt")
anchors = read_anchors("model_data/yolo_anchors.txt")
image_shape = (1932., 1932.)   

# 下载yolo模型
yolo_model = load_model("model_data/yolo608.h5") 

# 输出模型
yolo_model.summary()

# yolo_model的输出为一个(m,19,19,5,85)，需要转化为张量的形式
yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))

# 过滤boxes
scores, boxes, classes = yolo_eval(yolo_outputs, image_shape)

# 预测一副图片
def predict(sess, image_file):
    """
    Runs the graph stored in "sess" to predict boxes for "image_file". Prints and plots the preditions.
    
    Arguments:
    sess -- your tensorflow/Keras session containing the YOLO graph
    image_file -- name of an image stored in the "images" folder.
    
    Returns:
    out_scores -- tensor of shape (None, ), scores of the predicted boxes
    out_boxes -- tensor of shape (None, 4), coordinates of the predicted boxes
    out_classes -- tensor of shape (None, ), class index of the predicted boxes
    
    Note: "None" actually represents the number of predicted boxes, it varies between 0 and max_boxes. 
    """

    # Preprocess your image
    image, image_data = preprocess_image("images/" + image_file, model_image_size = (608, 608))
    print(type(image))
    # Run the session with the correct tensors and choose the correct placeholders in the feed_dict.
    # 运行
    out_scores, out_boxes, out_classes = sess.run([scores, boxes, classes],feed_dict={yolo_model.input: image_data,K.learning_phase(): 0})

    # Print predictions info
    print('Found {} boxes for {}'.format(len(out_boxes), image_file))
    # Generate colors for drawing bounding boxes.
    colors = generate_colors(class_names)
    # Draw bounding boxes on the image file
    draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors)
    # Save the predicted bounding box on the image
    image.save(os.path.join("out", image_file), quality=90)
    # Display the results in the notebook
    output_image = scipy.misc.imread(os.path.join("out", image_file))
    imshow(output_image)
    
    return out_scores, out_boxes, out_classes


out_scores, out_boxes, out_classes = predict(sess, "me.jpg")

