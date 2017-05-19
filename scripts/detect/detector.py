import mxnet as mx
import numpy as np
import cv2
from tools.image_processing import resize, transform
from tools.rand_sampler import RandSampler
#cap = cv2.VideoCapture(0)
#ret,img = cap.read()
#cap.release()
class DetIter(mx.io.DataIter):
    """
    Detection Iterator, which will feed data and label to network
    Optional data augmentation is performed when providing batch

    Parameters:
    ----------
    imdb : Imdb
        image database
    batch_size : int
        batch size
    data_shape : int or (int, int)
        image shape to be resized
    mean_pixels : float or float list
        [R, G, B], mean pixel values
    rand_samplers : list
        random cropping sampler list, if not specified, will
        use original image only
    rand_mirror : bool
        whether to randomly mirror input images, default False
    shuffle : bool
        whether to shuffle initial image list, default False
    rand_seed : int or None
        whether to use fixed random seed, default None
    max_crop_trial : bool
        if random crop is enabled, defines the maximum trial time
        if trial exceed this number, will give up cropping
    is_train : bool
        whether in training phase, default True, if False, labels might
        be ignored
    """
    def __init__(self, imdb, batch_size, data_shape, \
                 mean_pixels=[128, 128, 128], rand_samplers=[], \
                 rand_mirror=False, shuffle=False, rand_seed=None, \
                 is_train=True, max_crop_trial=50):
        super(DetIter, self).__init__()

        self._imdb = imdb
        self.batch_size = batch_size
        if isinstance(data_shape, int):
            data_shape = (data_shape, data_shape)
        self._data_shape = data_shape
        self._mean_pixels = mean_pixels
        if not rand_samplers:
            self._rand_samplers = []
        else:
            if not isinstance(rand_samplers, list):
                rand_samplers = [rand_samplers]
            assert isinstance(rand_samplers[0], RandSampler), "Invalid rand sampler"
            self._rand_samplers = rand_samplers
        self.is_train = is_train
        self._rand_mirror = rand_mirror
        self._shuffle = shuffle
        if rand_seed:
            np.random.seed(rand_seed) # fix random seed
        self._max_crop_trial = max_crop_trial

        self._current = 0
        self._size = imdb.num_images
        self._index = np.arange(self._size)

        self._data = None
        self._label = None
        self._get_batch()

    @property
    def provide_data(self):
        return [(k, v.shape) for k, v in self._data.items()]

    @property
    def provide_label(self):
        if self.is_train:
            return [(k, v.shape) for k, v in self._label.items()]
        else:
            return []

    def reset(self):
        self._current = 0
        if self._shuffle:
            np.random.shuffle(self._index)

    def iter_next(self):
        return self._current < self._size

    def next(self):
        if self.iter_next():
            self._get_batch()
            data_batch = mx.io.DataBatch(data=self._data.values(),
                                   label=self._label.values(),
                                   pad=self.getpad(), index=self.getindex())
            self._current += self.batch_size
            return data_batch
        else:
            raise StopIteration

    def getindex(self):
        return self._current / self.batch_size

    def getpad(self):
        pad = self._current + self.batch_size - self._size
        return 0 if pad < 0 else pad

    def _get_batch(self):
        """
        Load data/label from dataset
        """
        batch_data = []
        batch_label = []
        for i in range(self.batch_size):
            if (self._current + i) >= self._size:
                if not self.is_train:
                    continue
                # use padding from middle in each epoch
                idx = (self._current + i + self._size / 2) % self._size
                index = self._index[idx]
            else:
                index = self._index[self._current + i]
            # index = self.debug_index
            im_path = self._imdb.image_path_from_index(index)
	    #cap = cv2.VideoCapture(0)
	    #ret,img = cap.read()
	    #cap.release()
            #img = cv2.imread(im_path)
            gt = self._imdb.label_from_index(index).copy() if self.is_train else None
            data, label = self._data_augmentation(img, gt)
            batch_data.append(data)
            if self.is_train:
                batch_label.append(label)
        # pad data if not fully occupied
        for i in range(self.batch_size - len(batch_data)):
            assert len(batch_data) > 0
            batch_data.append(batch_data[0] * 0)
        self._data = {'data': mx.nd.array(np.array(batch_data))}
        if self.is_train:
            self._label = {'label': mx.nd.array(np.array(batch_label))}
        else:
            self._label = {'label': None}

    def _data_augmentation(self, data, label):
        """
        perform data augmentations: crop, mirror, resize, sub mean, swap channels...
        """
        if self.is_train and self._rand_samplers:
            rand_crops = []
            for rs in self._rand_samplers:
                rand_crops += rs.sample(label)
            num_rand_crops = len(rand_crops)
            # randomly pick up one as input data
            if num_rand_crops > 0:
                index = int(np.random.uniform(0, 1) * num_rand_crops)
                width = data.shape[1]
                height = data.shape[0]
                crop = rand_crops[index][0]
                xmin = int(crop[0] * width)
                ymin = int(crop[1] * height)
                xmax = int(crop[2] * width)
                ymax = int(crop[3] * height)
                if xmin >= 0 and ymin >= 0 and xmax <= width and ymax <= height:
                    data = data[ymin:ymax, xmin:xmax, :]
                else:
                    # padding mode
                    new_width = xmax - xmin
                    new_height = ymax - ymin
                    offset_x = 0 - xmin
                    offset_y = 0 - ymin
                    data_bak = data
                    data = np.full((new_height, new_width, 3), 128.)
                    data[offset_y:offset_y+height, offset_x:offset_x + width, :] = data_bak
                label = rand_crops[index][1]

        if self.is_train and self._rand_mirror:
            if np.random.uniform(0, 1) > 0.5:
                data = cv2.flip(data, 1)
                valid_mask = np.where(label[:, 0] > -1)[0]
                tmp = 1.0 - label[valid_mask, 1]
                label[valid_mask, 1] = 1.0 - label[valid_mask, 3]
                label[valid_mask, 3] = tmp

        if self.is_train:
            interp_methods = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, \
                              cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]
        else:
            interp_methods = [cv2.INTER_LINEAR]
        interp_method = interp_methods[int(np.random.uniform(0, 1) * len(interp_methods))]
        data = resize(data, self._data_shape, interp_method)
        data = transform(data, self._mean_pixels)
        return data, label
#import mxnet as mx
#import numpy as np
from timeit import default_timer as timer
from dataset.testdb import TestDB
#from dataset.iterator import DetIter

class Detector(object):
    """
    SSD detector which hold a detection network and wraps detection API

    Parameters:
    ----------
    symbol : mx.Symbol
        detection network Symbol
    model_prefix : str
        name prefix of trained model
    epoch : int
        load epoch of trained model
    data_shape : int
        input data resize shape
    mean_pixels : tuple of float
        (mean_r, mean_g, mean_b)
    batch_size : int
        run detection with batch size
    ctx : mx.ctx
        device to use, if None, use mx.cpu() as default context
    """
    def __init__(self, symbol, model_prefix, epoch, data_shape, mean_pixels, \
                 batch_size=1, ctx=None):
        self.ctx = ctx
        if self.ctx is None:
            self.ctx = mx.cpu()
        _, args, auxs = mx.model.load_checkpoint(model_prefix, epoch)
        self.mod = mx.mod.Module(symbol, context=ctx)
        self.data_shape = data_shape
        self.mod.bind(data_shapes=[('data', (batch_size, 3, data_shape, data_shape))])
        self.mod.set_params(args, auxs)
        self.data_shape = data_shape
        self.mean_pixels = mean_pixels

    def detect(self, det_iter, show_timer=False):
        """
        detect all images in iterator

        Parameters:
        ----------
        det_iter : DetIter
            iterator for all testing images
        show_timer : Boolean
            whether to print out detection exec time

        Returns:
        ----------
        list of detection results
        """
        num_images = det_iter._size
        if not isinstance(det_iter, mx.io.PrefetchingIter):
            det_iter = mx.io.PrefetchingIter(det_iter)
        start = timer()
        detections = self.mod.predict(det_iter).asnumpy()
        time_elapsed = timer() - start
        if show_timer:
            print "Detection time for {} images: {:.4f} sec".format(
                num_images, time_elapsed)
        result = []
        for i in range(detections.shape[0]):
            det = detections[i, :, :]
            res = det[np.where(det[:, 0] >= 0)[0]]
            result.append(res)
        #print "1"
        return result

    def im_detect(self, im_list, root_dir=None, extension=None, show_timer=False):
        """
        wrapper for detecting multiple images

        Parameters:
        ----------
        im_list : list of str
            image path or list of image paths
        root_dir : str
            directory of input images, optional if image path already
            has full directory information
        extension : str
            image extension, eg. ".jpg", optional

        Returns:
        ----------
        list of detection results in format [det0, det1...], det is in
        format np.array([id, score, xmin, ymin, xmax, ymax]...)
        """
        test_db = TestDB(im_list, root_dir=root_dir, extension=extension)
        test_iter = DetIter(test_db, 1, self.data_shape, self.mean_pixels,
                            is_train=False)
        return self.detect(test_iter, show_timer)

    def visualize_detection(self, cv_image_d, stra, img, dets, classes=[], thresh=0.6):
        """
        visualize detections in one image

        Parameters:
        ----------
        img : numpy.array
            image, in bgr format
        dets : numpy.array
            ssd detections, numpy.array([[id, score, x1, y1, x2, y2]...])
            each row is one object
        classes : tuple or list of str
            class names
        thresh : float
            score threshold
        """

	"""
        import matplotlib.pyplot as plt
        import random
        plt.imshow(img)
        height = img.shape[0]
        width = img.shape[1]
        colors = dict()
        for i in range(dets.shape[0]):
            cls_id = int(dets[i, 0])
            if cls_id >= 0:
                score = dets[i, 1]
                if score > thresh:
                    if cls_id not in colors:
                        colors[cls_id] = (random.random(), random.random(), random.random())
                    xmin = int(dets[i, 2] * width)
                    ymin = int(dets[i, 3] * height)
                    xmax = int(dets[i, 4] * width)
                    ymax = int(dets[i, 5] * height)
                    rect = plt.Rectangle((xmin, ymin), xmax - xmin,
                                         ymax - ymin, fill=False,
                                         edgecolor=colors[cls_id],
                                         linewidth=3.5)
                    plt.gca().add_patch(rect)
                    class_name = str(cls_id)
                    if classes and len(classes) > cls_id:
                        class_name = classes[cls_id]
                    plt.gca().text(xmin, ymin - 2,
                                    '{:s} {:.3f}'.format(class_name, score),
                                    bbox=dict(facecolor=colors[cls_id], alpha=0.5),
                                    fontsize=12, color='white')
        plt.show()
	"""
    	import random
    	height = img.shape[0]
    	width = img.shape[1]
    	colors = dict()
    	# print dets.shape[0]
        #print "3"
    	for i in range(dets.shape[0]):
            cls_id = int(dets[i, 0])
            if cls_id >= 0:
                score = dets[i, 1]
                if score > thresh:
                    if cls_id not in colors:
                        colors[cls_id] = (0,0,255)
                    xmin = int(dets[i, 2] * width)
                    ymin = int(dets[i, 3] * height)
                    xmax = int(dets[i, 4] * width)
                    ymax = int(dets[i, 5] * height)
                    centerx = int(0.5*(xmin + xmax))
                    centery = int(0.5*(ymin + ymax))
                    b = cv_image_d[centery, centerx]
                    cv2.rectangle(
                        img, (xmin, ymax), (xmax, ymin), color=colors[cls_id])
                    class_name = str(cls_id)
                    if classes and len(classes) > cls_id:
                        class_name = classes[cls_id]
                    cv2.putText(img, '{:s} {:.3f}'.format(class_name, score),
                                (xmin, ymin - 2), 0, 1.2, colors[cls_id], 2)
                    cv2.putText(img, 'Distance: {:.3f}'.format(b),
                                (xmin, ymin + 20), 0, 1.2, colors[cls_id], 2)
                    print "x= {} , y= {}, value= {}".format(xmin, ymin, b) 
        #print "4"
        #print "{}".format(width)
	cv2.namedWindow(stra)
	cv2.imshow(stra,img)
	#cv2.waitKey(0)
        #print "7"
	#cv2.destroyAllWindows()
        #(b, g, r) = img[center[1], center[0]]
        #print "x= {} , y= {}, value= {}".format(
        #        center[0], center[1], b) 
        return centerx
    def detect_and_visualize(self, image, cv_image_d, stra,im_list, root_dir=None, extension=None,
                             classes=[], thresh=0.6, show_timer=False):
        """
        wrapper for im_detect and visualize_detection

        Parameters:
        ----------
        im_list : list of str or str
            image path or list of image paths
        root_dir : str or None
            directory of input images, optional if image path already
            has full directory information
        extension : str or None
            image extension, eg. ".jpg", optional

        Returns:
        ----------

        """
	global img
        #import cv2
	#cap = cv2.VideoCapture(0)
	#ret,img = cap.read()
	#cap.release()
	#cv2.namedWindow(stra)
        #img[:, :, (0, 1, 2)] = img[:, :, (2, 1, 0)]
	#while(1):
	    #ret,img = cap.read()
        img = image;
	dets = self.im_detect(im_list, root_dir, extension, show_timer=show_timer)
        #print "2"
	for k, det in enumerate(dets):
            center = self.visualize_detection(cv_image_d, stra, img, det, classes, thresh)
        return center
        #print "99"
	#if cv2.waitKey(1) & 0xFF == ord('q'):
	#    break
	#cap.release()
	#cv2.waitKey(0)
	#cv2.destroyAllWindows()
	
        #dets = self.im_detect(im_list, root_dir, extension, show_timer=show_timer)
        #if not isinstance(im_list, list):
         #   im_list = [im_list]
        #assert len(dets) == len(im_list)
        #for k, det in enumerate(dets):
	#	self.visualize_detection(stra, img, det, classes, thresh)
            #img = cv2.imread(im_list[k])
	    
	#cap.release()#cap = cv2.VideoCapture(0)
	    #ret,img = cap.read()
            #img[:, :, (0, 1, 2)] = img[:, :, (2, 1, 0)]
            #self.visualize_detection(stra,img, det, classes, thresh)
