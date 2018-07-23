import six
import sys

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
tf.app.flags.DEFINE_string('train_dir', '',
                           'Directory to keep training outputs.')
tf.app.flags.DEFINE_string('log_root', '',
                           'Directory to keep the checkpoints. Should be a '
                           'parent directory of FLAGS.train_dir/eval_dir.')
tf.app.flags.DEFINE_integer('num_gpus', 0,
                            'Number of gpus used for training. (0 or 1)')
tf.app.flags.DEFINE_integer('num_residual_units', 5,
                            'num of residual units')
tf.app.flags.DEFINE_integer('total_steps', 90000, '')
tf.app.flags.DEFINE_string('Optimizer', 'mom',
                           'The optimizer used to train the model.')
tf.app.flags.DEFINE_bool('lr_decay', False,
                           'Whether use lr_decay when training cifar100.')
tf.app.flags.DEFINE_bool('RCE_train', False,
                         'Whether use RCE to train the model.')

batchsize_test=200
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

def train(hps,hps_test):
  """Training loop."""
  images, labels = input_name.build_input(
      FLAGS.dataset, FLAGS.train_data_path, hps.batch_size, FLAGS.mode)
  images_test, labels_test = input_name.build_input(
      FLAGS.dataset, FLAGS.eval_data_path, hps_test.batch_size, 'eval')
  model = model_name.ResNet(hps, images, FLAGS.mode,labels=labels,Reuse=False)
  model.build_graph()
  model_test = model_name.ResNet(hps_test, images_test, 'eval', labels=labels_test, Reuse=True)
  model_test.build_graph()

  param_stats = tf.contrib.tfprof.model_analyzer.print_model_analysis(
      tf.get_default_graph(),
      tfprof_options=tf.contrib.tfprof.model_analyzer.
          TRAINABLE_VARS_PARAMS_STAT_OPTIONS)
  sys.stdout.write('total_params: %d\n' % param_stats.total_parameters)

  tf.contrib.tfprof.model_analyzer.print_model_analysis(
      tf.get_default_graph(),
      tfprof_options=tf.contrib.tfprof.model_analyzer.FLOAT_OPS_OPTIONS)

  truth = tf.argmax(model_test.labels, axis=1)
  if FLAGS.RCE_train:
      predictions = tf.argmin(model_test.predictions, axis=1)
  else:
      predictions = tf.argmax(model_test.predictions, axis=1)
  precision = tf.reduce_mean(tf.to_float(tf.equal(predictions, truth)))

  summary_hook = tf.train.SummarySaverHook(
      save_steps=FLAGS.total_steps//200,
      output_dir=FLAGS.train_dir,
      summary_op=tf.summary.merge([model.summaries]))

  saver = tf.train.Saver()
  ckpt_saving_hook = tf.train.CheckpointSaverHook(
      checkpoint_dir=FLAGS.log_root,
      saver=saver,
      save_steps=2500)

  class _LearningRateSetterHook(tf.train.SessionRunHook):
    """Sets learning_rate based on global step."""

    def begin(self):
      self._lrn_rate = 0.1

    def before_run(self, run_context):
      return tf.train.SessionRunArgs(
          model.global_step,  # Asks for global step value.
          feed_dict={model.lrn_rate: self._lrn_rate})  # Sets learning rate

    def after_run(self, run_context, run_values):
      train_step = run_values.results
      if FLAGS.dataset=='mnist':
          if train_step < 10000:
            self._lrn_rate = 0.1
          elif train_step < 15000:
            self._lrn_rate = 0.01
          elif train_step < 20000:
            self._lrn_rate = 0.001
          else:
            self._lrn_rate = 0.0001
      elif FLAGS.dataset=='cifar10':
          if train_step < 40000:
            self._lrn_rate = 0.1
          elif train_step < 60000:
            self._lrn_rate = 0.01
          elif train_step < 80000:
            self._lrn_rate = 0.001
          else:
            self._lrn_rate = 0.0001
      else:
          print('Wrong dataset name')

  with tf.train.MonitoredTrainingSession(
          hooks=[ckpt_saving_hook, _LearningRateSetterHook()],
          chief_only_hooks=[summary_hook],
          # Since we provide a SummarySaverHook, we need to disable default
          # SummarySaverHook. To do that we set save_summaries_steps to 0.
          save_summaries_steps=0,
          config=tf.ConfigProto(allow_soft_placement=True)) as mon_sess:
      while not mon_sess.should_stop() and mon_sess.run(model.global_step) <= FLAGS.total_steps:
          mon_sess.run(model.train_op)
          step = mon_sess.run(model.global_step)
          if step%500==0:
              precision_final = 0.0
              for _ in six.moves.range(50):
                  precision_final += mon_sess.run(precision)

              precision_final = precision_final/50
              print('Step: %d Test precision: %.5f'%(step,precision_final))




def main(_):
  if FLAGS.num_gpus == 0:
    dev = '/cpu:0'
  elif FLAGS.num_gpus == 1:
    dev = '/gpu:0'
  else:
    raise ValueError('Only support 0 or 1 gpu.')




  hps = model_name.HParams(batch_size=128,
                             num_classes=num_classes,
                             min_lrn_rate=0.0001,
                             lrn_rate=0.1,
                             num_residual_units=FLAGS.num_residual_units,
                             use_bottleneck=False,
                             weight_decay_rate=0.000,
                             relu_leakiness=0.1,
                             optimizer=FLAGS.Optimizer,
                             RCE_train=FLAGS.RCE_train)

  hps_test = model_name.HParams(batch_size=batchsize_test,
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
    if FLAGS.mode == 'train':
      train(hps,hps_test)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
