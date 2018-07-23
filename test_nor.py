import time
import six

import cifar_input
import resnet_model_cifar
import mnist_input
import resnet_model_mnist
import numpy as np
import tensorflow as tf

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

def evaluate(hps):
  """Eval loop."""
  images, labels = input_name.build_input(
      FLAGS.dataset, FLAGS.eval_data_path, hps.batch_size, FLAGS.mode)
  model = model_name.ResNet(hps, images, FLAGS.mode,labels=labels,Reuse=False)
  model.build_graph()
  saver = tf.train.Saver()
  summary_writer = tf.summary.FileWriter(FLAGS.eval_dir)

  sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
  tf.train.start_queue_runners(sess)

  best_precision = 0.0
  while True:
    try:
      ckpt_state = tf.train.get_checkpoint_state(FLAGS.log_root)
    except tf.errors.OutOfRangeError as e:
      tf.logging.error('Cannot restore checkpoint: %s', e)
      continue
    if not (ckpt_state and ckpt_state.model_checkpoint_path):
      tf.logging.info('No model to eval yet at %s', FLAGS.log_root)
      continue
    tf.logging.info('Loading checkpoint %s', ckpt_state.model_checkpoint_path)
    saver.restore(sess, ckpt_state.model_checkpoint_path)

    total_prediction, correct_prediction = 0, 0
    for _ in six.moves.range(FLAGS.eval_batch_count):
      (summaries, predictions, truth, train_step) = sess.run(
          [model.summaries, model.predictions,
           model.labels, model.global_step])

      truth = np.argmax(truth, axis=1)
      if FLAGS.RCE_train:
          predictions = np.argmin(predictions, axis=1)
      else:
          predictions = np.argmax(predictions, axis=1)
      correct_prediction += np.sum(truth == predictions)
      total_prediction += predictions.shape[0]

    precision = 1.0 * correct_prediction / total_prediction
    best_precision = max(precision, best_precision)

    precision_summ = tf.Summary()
    precision_summ.value.add(
        tag='Precision', simple_value=precision)
    summary_writer.add_summary(precision_summ, train_step)
    best_precision_summ = tf.Summary()
    best_precision_summ.value.add(
        tag='Best Precision', simple_value=best_precision)
    summary_writer.add_summary(best_precision_summ, train_step)
    summary_writer.add_summary(summaries, train_step)
    tf.logging.info('precision: %.5f, best precision: %.5f' %
                    (precision, best_precision))
    summary_writer.flush()

    if FLAGS.eval_once:
      break

    time.sleep(6)



def main(_):
  if FLAGS.num_gpus == 0:
    dev = '/cpu:0'
  elif FLAGS.num_gpus == 1:
    dev = '/gpu:0'
  else:
    raise ValueError('Only support 0 or 1 gpu.')


  batch_size = 100

  hps = model_name.HParams(batch_size=batch_size,
                             num_classes=num_classes,
                             min_lrn_rate=0.0001,
                             lrn_rate=0.1,
                             num_residual_units=FLAGS.num_residual_units,
                             use_bottleneck=False,
                             weight_decay_rate=0.000,
                             relu_leakiness=0.1,
                             optimizer=FLAGS.Optimizer,
                             RCE_train=FLAGS.RCE_train)

  with tf.device(dev):
      evaluate(hps)



if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
