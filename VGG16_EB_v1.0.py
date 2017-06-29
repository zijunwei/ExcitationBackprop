# This is a Excitation Backpropogation network using VGG16:
import tensorflow as tf
import tensorflow.contrib.slim as slim
from nets.vgg import vgg_16 as vgg
from nets.vgg import vgg_arg_scope
import preprocessing.imagenet_utils as imagenet_utils
import preprocessing.vgg as preprocessing
import PIL.Image as Image
import numpy as np
import os
import glob

def main(argv=None):
    model_path = './modelparams/vgg_16.ckpt'
    labels_to_names, label_names = imagenet_utils.create_readable_names_for_imagenet_labels()

    with tf.Graph().as_default() as graph:
        vgg_inputs = tf.placeholder(dtype=tf.float32, shape=[None, 224, 224, 3], name='input_image')
        with slim.arg_scope(vgg_arg_scope()):
            vgg_output, vgg_endpoints = vgg(vgg_inputs, is_training=False)


        vgg_variables = [v for v in tf.trainable_variables()]





        #todo: Compute gradient layer by layer



        init_fn = slim.assign_from_checkpoint_fn(model_path, slim.get_model_variables('vgg_16'))

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            init_fn(sess)

            test_image_name = 'demo.jpg'

            print 'Processing: {:s}'.format(test_image_name)
            s_image = (Image.open(test_image_name).convert('RGB'))

            s_image = s_image.resize([224, 224], resample=Image.BICUBIC)
            s_image = np.asarray(s_image, dtype=np.float32)
            if len(s_image.shape) < 3:
                print "Image is not RGB 3Dimensional data"
                return
            s_image = preprocessing.mean_image_subtraction(s_image)
            s_image = np.expand_dims(s_image, axis=0)

            s_output = sess.run(vgg_output, feed_dict={vgg_inputs:s_image})
            s_output = s_output[0]
            sorted_cates = np.argsort(-s_output)
            selected_cates = sorted_cates[0:5]

            for i, catId in enumerate(selected_cates):
            # !!!: Notice here this is a catId+1 instead of catID
                print '{:d}\t category: {:s}; confidence:{:.4f},\tlabelID: {:s}'.format(i, labels_to_names[catId+1], s_output[catId], label_names[catId])


if __name__ == '__main__':
    main()












