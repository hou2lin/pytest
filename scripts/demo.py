#!/usr/bin/env python
import sys
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
import argparse
import tools.find_mxnet
import mxnet as mx
import os
import importlib
import sys
import cv2
from detect.detector import Detector

CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')

def get_detector(net, prefix, epoch, data_shape, mean_pixels, ctx,
                 nms_thresh=0.5, force_nms=True):
    """
    wrapper for initialize a detector

    Parameters:
    ----------
    net : str
        test network name
    prefix : str
        load model prefix
    epoch : int
        load model epoch
    data_shape : int
        resize image shape
    mean_pixels : tuple (float, float, float)
        mean pixel values (R, G, B)
    ctx : mx.ctx
        running context, mx.cpu() or mx.gpu(?)
    force_nms : bool
        force suppress different categories
    """

    

    sys.path.append(os.path.join(os.getcwd(), 'symbol'))
    net = importlib.import_module("symbol_" + net) \
        .get_symbol(len(CLASSES), nms_thresh, force_nms)
    detector = Detector(net, prefix + "_" + str(data_shape), epoch, \
        data_shape, mean_pixels, ctx=ctx)
    return detector

class image_converter(object):
  def __init__(self):
    #self.image_pub = rospy.Publisher("/image_topic_2",Image)
    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber("/zed/left/image_raw_color",Image, self.callback)
    self.image_info = rospy.Subscriber("/zed/left/camera_info",	CameraInfo,self.godetect)
    self.image_sub2 = rospy.Subscriber("/zed/depth/depth_registered",Image, self.callback2)
    #rospy.Timer(rospy.Duration(2), self.callback2)



  def godetect(self,data):
    # parse image list
    image_list = ['/home/wolfram/catkin_ws/src/pytsest/scripts/a.jpg']

    detector = get_detector('vgg16_reduced', os.path.join(os.getcwd(), 'model', 'ssd'), 0,
                            300, (123, 117, 104),
                            mx.gpu(0), 0.5, True)
    # run detection
    strr="Capture"
    center = detector.detect_and_visualize(cv_image, cv_image_d, strr,image_list, None, None,
                                  CLASSES, 0.6, True)
    #global b
    #b = cv_image_d[center[1], center[0]]
    #print "x= {} , y= {}, value= {}".format(
    #            center[0], center[1], b) 

  def callback(self,data):
    try:
      global cv_image
      cv_image = self.bridge.imgmsg_to_cv2(data,"bgr8")
      #cv2.imshow("Capture",cv_image)
      #cv2.waitKey(0)
      #print "cShow"
    except CvBridgeError as e:
      print(e)

  def callback2(self,data):
    try:
      global cv_image_d
      cv_image_d = self.bridge.imgmsg_to_cv2(data)
      #cv2.namedWindow("Capture1")
      #cv2.imshow("Capture1",cv_image_d)
      #cv2.waitKey(3)
    except CvBridgeError as e:
      print(e)
    
  
if __name__ == '__main__':
    cv2.startWindowThread()
    #cv2.namedWindow("Capture")
    ic = image_converter()
    rospy.init_node('image_converter',anonymous=True)
    try:
      rospy.spin()
    except KeyboardInterrupt:
      pass#print("Shutting down")
    cv2.destroyAllWindows()
    
