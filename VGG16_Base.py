# This is a basic network using VGG16: load an image, process it and forward pass it to vggnet to get the result
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

        init_fn = slim.assign_from_checkpoint_fn(model_path, slim.get_model_variables('vgg_16'))

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            init_fn(sess)
            test_image_dir = '/Users/zwei/datasets/imagenet2012/images/val/n03929660'
            image_list = glob.glob(os.path.join(test_image_dir, '*.JPEG'))
            image_label = 'n03929660'
            image_index = label_names.index(image_label)
            # test_image_name = 'ILSVRC2012_val_00047770.JPEG'
            n_images = len(image_list)
            top_1 = 0
            top_5 = 0
            for test_image_name in image_list:
                print 'Processing: {:s}'.format(test_image_name)
                s_image = (Image.open(os.path.join(test_image_dir, test_image_name)).convert('RGB'))

                s_image = s_image.resize([224, 224], resample=Image.BICUBIC)
                s_image = np.asarray(s_image, dtype=np.float32)
                if len(s_image.shape) < 3:
                    continue
                s_image = preprocessing.mean_image_subtraction(s_image)
                s_image = np.expand_dims(s_image, axis=0)

                s_output = sess.run(vgg_output, feed_dict={vgg_inputs:s_image})
                s_output = s_output[0]
                sorted_cates = np.argsort(-s_output)
                selected_cates = sorted_cates[0:5]
                if image_index == selected_cates[0]:
                    top_1 += 1
                if image_index in selected_cates:
                    top_5 += 1


                # for i, catId in enumerate(selected_cates):
                #!!!: Notice here this is a catId+1 instead of catID
                #     print '{:d}\t category: {:s}; confidence:{:.4f},\tlabelID: {:s}'.format(i, labels_to_names[catId+1], s_output[catId], label_names[catId])

            print "Top 1 Error:\t{:.2f}\t, Top 5 Error:\t{:.2f}".format(top_1*1./n_images, top_5*1./n_images)

if __name__ == '__main__':
    main()












