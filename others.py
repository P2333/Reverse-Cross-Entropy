import time
import six
import sys

import cifar_input
import resnet_model_cifar
import mnist_input
import resnet_model_mnist
import numpy as np
import tensorflow as tf
import t_sne

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('dataset', '', 'cifar10 or cifar100.')
tf.app.flags.DEFINE_string('mode', 'train', 'train or eval.')
tf.app.flags.DEFINE_string('train_data_path', '',
                           'Filepattern for training data.')
tf.app.flags.DEFINE_string('eval_data_path', '',
                           'Filepattern for eval data')
# tf.app.flags.DEFINE_integer('image_size', 0, 'Image side length.')
tf.app.flags.DEFINE_string('train_dir', '',
                           'Directory to keep training outputs.')
tf.app.flags.DEFINE_string('eval_dir', '',
                           'Directory to keep eval outputs.')
tf.app.flags.DEFINE_integer('eval_batch_count', 50,
                            'Number of batches to eval.')
tf.app.flags.DEFINE_bool('eval_once', False,
                         'Whether evaluate the model only once.')
tf.app.flags.DEFINE_string('log_root', '',
                           'Directory to keep the checkpoints. Should be a '
                           'parent directory of FLAGS.train_dir/eval_dir.')
tf.app.flags.DEFINE_integer('num_gpus', 0,
                            'Number of gpus used for training. (0 or 1)')
tf.app.flags.DEFINE_integer('num_residual_units', 5,
                            'num of residual units')
tf.app.flags.DEFINE_integer('total_steps', 100000, '')
tf.app.flags.DEFINE_string('Optimizer', 'mom',
                           'The optimizer used to train the model.')
tf.app.flags.DEFINE_bool('lr_decay', False,
                           'Whether use lr_decay when training cifar100.')
tf.app.flags.DEFINE_bool('RCE_train', False,
                         'Whether use RCE to train the model.')

num_classes = 10
if FLAGS.dataset == 'cifar10':
    image_size = 32
    num_channel = 3
    model_name = resnet_model_cifar
    input_name = cifar_input

elif FLAGS.dataset == 'mnist':
    image_size = 28
    num_channel = 1
    model_name = resnet_model_mnist
    input_name = mnist_input

elif FLAGS.dataset == 'cifar100':
    image_size = 32
    num_channel = 3
    model_name = resnet_model_cifar
    input_name = cifar_input
else:
    print('Unrecognized dataset')
    image_size = None
    num_channel = None
    model_name = None
    input_name = None

if FLAGS.RCE_train == True:
    f1 = 'RCE'
else:
    f1 = 'CE'

def kernel_para(hps):
    # Construct graph, eval_data_path is the path of TRAINING dataset
    images, labels = input_name.build_input(
        FLAGS.dataset, FLAGS.eval_data_path, hps.batch_size, FLAGS.mode)  # FLAGS.mode='attack', batch_size=200
    Res = model_name.ResNet(hps, images, FLAGS.mode, Reuse=False)
    Res.build_graph()
    saver = tf.train.Saver()

    # Open session and restore checkpoint
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    tf.train.start_queue_runners(sess)
    sess.run(tf.global_variables_initializer())

    ckpt_state = tf.train.get_checkpoint_state(FLAGS.log_root)  # Choose dir according to rt
    tf.logging.info('Loading checkpoint %s', ckpt_state.model_checkpoint_path)
    saver.restore(sess, ckpt_state.model_checkpoint_path)

    #Calculate tsne_logits
    tsne_logits = np.reshape(np.array([]),(0,64))
    labels_all = np.array([])


    for i in six.moves.range(FLAGS.eval_batch_count):
        print('The %d batch in total %d' % (i, FLAGS.eval_batch_count))
        (tsne_logits_help,labels_part) = sess.run([Res.t_SNE_logits,tf.argmax(labels, 1)])
        tsne_logits = np.concatenate((tsne_logits, tsne_logits_help),axis=0)
        labels_all = np.concatenate((labels_all, labels_part), axis=0)

    print(tsne_logits.shape)
    np.savetxt('training_logits_'+f1, tsne_logits)
    np.savetxt('training_logitslabels_' + f1, labels_all)
    return None

def t_SNE_logits(hps,num_batch):
    # Construct graph
    images, labels = input_name.build_input(
        FLAGS.dataset, FLAGS.eval_data_path, hps.batch_size, FLAGS.mode)
    Res = model_name.ResNet(hps, images, FLAGS.mode, Reuse=False)
    Res.build_graph()
    saver = tf.train.Saver()

    # Open session and restore checkpoint
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    tf.train.start_queue_runners(sess)
    sess.run(tf.global_variables_initializer())

    ckpt_state = tf.train.get_checkpoint_state(FLAGS.log_root)  # Choose dir according to rt
    tf.logging.info('Loading checkpoint %s', ckpt_state.model_checkpoint_path)
    saver.restore(sess, ckpt_state.model_checkpoint_path)

    logits_nor = Res.t_SNE_logits

    dim_logits = 64
    # if hps.batch_size!=tf.shape(logits_nor)[0]:
    #     print('Error!!!!!')
    #     return
    logits_all = np.reshape(np.array([]),(0,dim_logits))
    labels_all = np.array([])

    for i in six.moves.range(num_batch):
        print(i)
        (logits_part_nor, labels_part) = sess.run([logits_nor, tf.argmax(labels, 1)])
        logits_all = np.concatenate((logits_all, logits_part_nor), axis=0)
        labels_all = np.concatenate((labels_all, labels_part), axis=0)

    tsne_return = t_sne.tsne(logits_all, no_dims=2, initial_dims=60, perplexity=30.0)

    # Save results
    np.savetxt('nor_tsne_results_' + FLAGS.dataset + '/tSNE_' + f1, tsne_return)
    np.savetxt('nor_tsne_results_' + FLAGS.dataset + '/tSNElabels_' + f1, labels_all)
    return None


def main(_):
  if FLAGS.num_gpus == 0:
    dev = '/cpu:0'
  elif FLAGS.num_gpus == 1:
    dev = '/gpu:0'
  else:
    raise ValueError('Only support 0 or 1 gpu.')

  if FLAGS.mode == 'kernel_para':
    batch_size = 100
  elif FLAGS.mode == 'tSNE_logits':
    batch_size = 100

  hps = model_name.HParams(batch_size=batch_size,
                             num_classes=num_classes,
                             min_lrn_rate=0.0001,
                             lrn_rate=0.1,
                             num_residual_units=FLAGS.num_residual_units,
                             use_bottleneck=False,
                             weight_decay_rate=0.0002,
                             relu_leakiness=0.1,
                             optimizer=FLAGS.Optimizer,
                             RCE_train=FLAGS.RCE_train)

  with tf.device(dev):
    if FLAGS.mode == 'kernel_para':
      kernel_para(hps)
    elif FLAGS.mode == 'tSNE_logits':
      t_SNE_logits(hps, 10)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
