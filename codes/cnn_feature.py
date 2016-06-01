import h5py
import json
import sys
import cv2
import os.path as osp
import numpy as np
from tools import Timer

from config import CAFFE_ROOT, DATA_ROOT
sys.path.insert(0, osp.join(CAFFE_ROOT, 'python'))
import caffe

BS = 1


def cnn_patch_feature(net, transformer, image, bbox, include_whole_image=False):
    im_list = [image] if include_whole_image else []
    for j in xrange(bbox.shape[0]):
        h1, w1, h2, w2 = bbox[j, :]  # (h1, w1, h2, w2)
        im_list.append(image[h1:h2, w1:w2, :])
    im_list = [transformer.preprocess('data', im)[np.newaxis, :, :, :] for im in im_list]
    feat = []
    for i in xrange(0, bbox.shape[0], BS):
        net.blobs['data'].data[...] = np.concatenate(im_list[i:i+BS], axis=0)
        net.forward()
        feat.append(net.blobs['pool5'].data.reshape([BS, -1]))
    return np.vstack(feat)


def cnn_whole_feature(net, transformer, image):
    im = transformer.preprocess('data', image)[np.newaxis, :, :, :]
    net.blobs['data'].data[...] = im
    net.forward()
    feat = net.blobs['pool5'].data
    return feat.flatten()


def load_network(proto_txt, caffe_model, device):
    if 'gpu' in device:
        caffe.set_mode_gpu()
        device_id = int(device.split('gpu')[-1])
        caffe.set_device(device_id)
    else:
        caffe.set_mode_cpu()
    # load network
    net = caffe.Net(proto_txt, caffe_model, caffe.TEST)

    # tansformer
    mu = np.load(osp.join(CAFFE_ROOT, 'models', 'ResNet', 'ResNet_mean.npy'))
    mu = mu.mean(1).mean(1)
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))  # move image channels to outermost dimension
    transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
    transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
    transformer.set_channel_swap('data', (2, 1, 0))  # swap channels from RGB to BGR
    # reshape input
    net.blobs['data'].reshape(BS, 3, 224, 224)

    return net, transformer


def _script_for_patch_feature(head, tail, device, dataset='mscoco', data_split='train'):
    data_dir = osp.join(DATA_ROOT, dataset)

    # set the model path here
    prototxt = osp.join(CAFFE_ROOT, 'models', 'ResNet', 'ResNet-152-deploy.prototxt')
    caffemodel = osp.join(CAFFE_ROOT, 'models', 'ResNet', 'ResNet-152-model.caffemodel')

    # load network
    net, transformer = load_network(prototxt, caffemodel, device)

    # prepare image files
    cap_file = osp.join(data_dir, 'captions_{}.json'.format(data_split))
    image_ids = json.load(open(cap_file, 'r'))['image_ids']
    im_files = ['COCO_{}2014_'.format(data_split) + str(i).zfill(12) + '.jpg' for i in image_ids]
    im_files = [osp.join(data_dir, data_split+'2014', i) for i in im_files]
    with h5py.File(osp.join(data_dir, 'bbox.h5'), 'r') as f:
        bbox = np.array(f[data_split])

    # initialize h5 file
    save_file = osp.join(data_dir, 'features', 'features_30res.h5')
    with h5py.File(save_file) as f:
        if data_split not in f:
            f.create_dataset(data_split, [len(im_files), 30, 2048], 'float32')

    # computing
    timer = Timer()
    print '\n\n... computing'
    for i in xrange(head, tail):
        timer.tic()
        im = caffe.io.load_image(im_files[i])  # (h, w, c)
        with h5py.File(save_file) as f:
            feat = cnn_patch_feature(net, transformer, im, bbox[i, :, :])
            f[data_split][i, :] = feat
        print '[{:d}]  {}  [{:.3f} sec]'.format(i, osp.split(im_files[i])[-1], timer.toc())


if __name__ == '__main__':
    _script_for_patch_feature(0, 82783, 'gpu0', 'mscoco', 'train')


