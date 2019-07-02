from __future__ import division
from __future__ import absolute_import

import six
import cifar_input
import mnist_input
import resnet_model_cifar
import resnet_model_mnist
import t_sne
import numpy as np
import tensorflow as tf
import attacks
import sys
sys.path.append('..')
sys.path.append('./../attacks')
import time
import copy
from scipy.io import loadmat
from scipy.misc import imsave

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('dataset', '', 'cifar10 or cifar100.')
tf.app.flags.DEFINE_string('mode', 'train', 'train or eval.')
tf.app.flags.DEFINE_string('train_data_path', '',
                           'Filepattern for training data.')
tf.app.flags.DEFINE_string('eval_data_path', '',
                           'Filepattern for eval data')
tf.app.flags.DEFINE_string('train_dir', '',
                           'Directory to keep training outputs.')
tf.app.flags.DEFINE_string('eval_dir', '',
                           'Directory to keep eval outputs.')
tf.app.flags.DEFINE_integer('eval_batch_count', 10,
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
tf.app.flags.DEFINE_string('Optimizer', 'mom',
                           'The optimizer used to train the model.')
tf.app.flags.DEFINE_bool('RCE_train', False,
                         'Whether use RCE to train the model.')
tf.app.flags.DEFINE_string('attack_method', 'fgsm',
                           'The attacking method used')
tf.app.flags.DEFINE_float('eps', 0.01,
                         'The eps in attacking methods.')
tf.app.flags.DEFINE_string('save_pwd', None,
                           '')

epoch_jsma = 100


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
    sigma2 = 0.1 / 0.26
    f1 = 'RCE'
else:
    sigma2 = 1.0 / 0.26
    f1 = 'CE'


def models(hps, images, RCE_train, logits=False, tsne_logits=False):
    model = model_name.ResNet(hps, images, FLAGS.mode, Reuse=True)
    model.build_graph()
    op = model.predictions.op
    logit, = op.inputs
    if RCE_train==True:
        logit = -logit
    if tsne_logits==True:
        return tf.nn.softmax(logit), model.t_SNE_logits
    if logits==True:
        return tf.nn.softmax(logit), logit
    return tf.nn.softmax(logit)

class models_carlini:
    def __init__(self,hps):
        self.image_size = image_size
        self.num_channels = num_channel############MNIST and CIFAR10 are different ar here
        self.num_labels = num_classes
        self.hps = hps

    def predict(self,images,tsne_logits=False):
        model = model_name.ResNet(self.hps, images, FLAGS.mode, Reuse=True)
        model.build_graph()
        op = model.predictions.op
        logit, = op.inputs
        if FLAGS.RCE_train==True:
            logit = -logit
        if tsne_logits==True:
            return logit,model.t_SNE_logits
        return logit

def adv_craft_func(hps, images, method, eps=0.01,RCE_train=False, target_labels=None):
    if method=='fgsm':
        print('Attacking method is fgsm')
        adversarial_sample = attacks.fgsm.fgsm(models, images, hps, RCE_train,
                                               eps=eps, epochs=1, clip_min=-0.5, clip_max=0.5)
    elif method=='random':
        print('Attacking method is random')
        adversarial_sample = tf.clip_by_value(images + tf.random_uniform((hps.batch_size,image_size,image_size,num_channel),
                                                        minval=-eps, maxval=eps), clip_value_min=-0.5, clip_value_max=0.5)
    elif method=='bim':
        print('Attacking method is bim')
        adversarial_sample = attacks.fgsm.fgsm(models, images, hps, RCE_train,
                                               eps=eps/10, epochs=10, clip_min=-0.5, clip_max=0.5)
    elif method=='tgsm':
        print('Attacking method is tgsm')
        adversarial_sample = attacks.tgsm.tgsm(models, images, hps, RCE_train, y=None,
                                               eps=eps/10, epochs=10, clip_min=-0.5, clip_max=0.5)

    elif method=='jsma':
        print('Attacking method is jsma')
        if target_labels==None:
            print('Target label is the argmin label')
            model_target_y = models(hps, images, FLAGS.RCE_train, logits=False)
            target_y64 = tf.argmin(model_target_y,axis=1)
        else:
            target_y64=target_labels
        target_y = tf.cast(target_y64, tf.int32)
        adversarial_sample = attacks.jsma.jsma(models, images, hps, RCE_train, target_y,epochs=epoch_jsma, eps=eps,
                                               clip_min=-0.5, clip_max=0.5, pair=False, min_proba=0.0)

    elif method=='smda':
        print('Attacking method is smda')
        if target_labels==None:
            print('Target label is the argmin label')
            model_target_y = models(hps, images, FLAGS.RCE_train, logits=False)
            target_y64 = tf.argmin(model_target_y,axis=1)
        else:
            target_y64=target_labels
        target_y = tf.cast(target_y64, tf.int32)
        adversarial_sample = attacks.smda.smda(models, images, hps, RCE_train, target_y, epochs=epoch_jsma, eps=eps,
                                               clip_min=-0.5, clip_max=0.5, min_proba=0.0)

    else:
        print('Not recognized method')
        adversarial_sample = None
    return adversarial_sample

def tSNE_visual(hps,num_batch):
    # Construct graph
    images, labels = input_name.build_input(
        FLAGS.dataset, FLAGS.eval_data_path, hps.batch_size, FLAGS.mode)  # FLAGS.mode='attack', batch_size=200
    Res = model_name.ResNet(hps, images, FLAGS.mode, Reuse=False)
    Res.build_graph()
    saver = tf.train.Saver()
    adv_images = adv_craft_func(hps, images, FLAGS.attack_method, eps=FLAGS.eps, RCE_train=FLAGS.RCE_train)
    model_nor = model_name.ResNet(hps, images, FLAGS.mode, Reuse=True)
    model_nor.build_graph()
    model_adv = model_name.ResNet(hps, adv_images, FLAGS.mode, Reuse=True)
    model_adv.build_graph()

    # Open session and restore checkpoint
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    tf.train.start_queue_runners(sess)
    sess.run(tf.global_variables_initializer())

    ckpt_state = tf.train.get_checkpoint_state(FLAGS.log_root)  # Choose dir according to rt
    tf.logging.info('Loading checkpoint %s', ckpt_state.model_checkpoint_path)
    saver.restore(sess, ckpt_state.model_checkpoint_path)

    logits_nor = model_nor.t_SNE_logits
    logits_adv = model_adv.t_SNE_logits

    dim_logits = logits_nor.shape[1]
    if hps.batch_size!=logits_nor.shape[0]:
        print('Error!!!!!')
        return
    logits_all = np.reshape(np.array([]),(0,dim_logits))
    labels_all = np.array([])
    is_adv_all = np.array([])


    #make the num of adv the same as per class
    if FLAGS.attack_method == 'fgsm' or FLAGS.attack_method == 'tgsm':
        num_adv = int(hps.batch_size/10)
        print('num_adv is %d'%(num_adv))
    else:
        num_adv = hps.batch_size

    for i in six.moves.range(num_batch):
        print(i)
        (logits_part_nor, logits_part_adv, labels_part) = sess.run([logits_nor, logits_adv, tf.argmax(labels, 1)])
        logits_all = np.concatenate((logits_all, logits_part_nor), axis=0)
        labels_all = np.concatenate((labels_all, labels_part), axis=0)
        is_adv_all = np.concatenate((is_adv_all, np.zeros(hps.batch_size)), axis=0)
        logits_all = np.concatenate((logits_all, logits_part_adv[:num_adv]), axis=0)
        labels_all = np.concatenate((labels_all, labels_part[:num_adv]), axis=0)
        is_adv_all = np.concatenate((is_adv_all, np.ones(num_adv)), axis=0)

    tsne_return = t_sne.tsne(logits_all, no_dims=2, initial_dims=60, perplexity=30.0)

    # Save results
    if FLAGS.RCE_train == True:
        f1 = 'RCE'
    else:
        f1 = 'CE'
    np.savetxt(FLAGS.attack_method + '_' + FLAGS.dataset + '/tSNE/tSNE_' + f1, tsne_return)
    np.savetxt(FLAGS.attack_method + '_' + FLAGS.dataset + '/tSNE/tSNElabels_' + f1, labels_all)
    np.savetxt(FLAGS.attack_method + '_' + FLAGS.dataset + '/tSNE/tSNEisadv_' + f1, is_adv_all)
    return None

def tSNE_visual_carliniLi(hps, num_batch):
    # Construct graph
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

    model_carlini = models_carlini(hps)
    if FLAGS.attack_method == 'carliniLi':
        attack_carlini = attacks.carliniLi.CarliniLi(sess, model_carlini, largest_const=10 ** -3)
    elif FLAGS.attack_method == 'carliniL2':
        attack_carlini = attacks.carliniL2.CarliniL2(sess, model_carlini, batch_size=10, max_iterations=1000, confidence=0,binary_search_steps=3)
    adv_image = tf.placeholder(tf.float32, shape=[hps.batch_size, image_size, image_size, num_channel])

    _, logits_nor = model_carlini.predict(images, tsne_logits=True)
    _, logits_adv = model_carlini.predict(adv_image, tsne_logits=True)

    dim_logits = logits_nor.shape[1]
    if hps.batch_size != logits_nor.shape[0]:
        print('Error!!!!!')
        return
    logits_all = np.reshape(np.array([]), (0, dim_logits))
    labels_all = np.array([])
    is_adv_all = np.array([])

    # make the num of adv the same as per class
    # if FLAGS.attack_method == 'fgsm' or FLAGS.attack_method == 'tgsm':
    #     num_adv = int(hps.batch_size/10)
    #     print('num_adv is %d'%(num_adv))
    # else:
    #     num_adv = hps.batch_size

    num_adv = hps.batch_size

    for i in six.moves.range(num_batch):
        print(i)
        input_data = sess.run(images)
        target_label = sess.run(labels)
        adv = attack_carlini.attack(input_data, target_label)

        (logits_part_nor, logits_part_adv, labels_part) = sess.run([logits_nor, logits_adv, tf.argmax(labels, 1)],
                                                                   feed_dict={adv_image: adv})
        logits_all = np.concatenate((logits_all, logits_part_nor), axis=0)
        labels_all = np.concatenate((labels_all, labels_part), axis=0)
        is_adv_all = np.concatenate((is_adv_all, np.zeros(hps.batch_size)), axis=0)
        logits_all = np.concatenate((logits_all, logits_part_adv[:num_adv]), axis=0)
        labels_all = np.concatenate((labels_all, labels_part[:num_adv]), axis=0)
        is_adv_all = np.concatenate((is_adv_all, np.ones(num_adv)), axis=0)

    tsne_return = t_sne.tsne(logits_all, no_dims=2, initial_dims=60, perplexity=30.0)

    # Save results
    if FLAGS.RCE_train == True:
        f1 = 'RCE'
    else:
        f1 = 'CE'
    np.savetxt(FLAGS.attack_method + '_' + FLAGS.dataset + '/tSNE/tSNE_' + f1, tsne_return)
    np.savetxt(FLAGS.attack_method + '_' + FLAGS.dataset + '/tSNE/tSNElabels_' + f1, labels_all)
    np.savetxt(FLAGS.attack_method + '_' + FLAGS.dataset + '/tSNE/tSNEisadv_' + f1, is_adv_all)
    return None

def apply_attack_carlini(hps):
    # Construct graph
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

    num_sample = hps.batch_size * FLAGS.eval_batch_count

    # Initialize results to save
    entropy_test_adv_all = np.array([])
    confidence_test_adv_all = np.array([])
    entropy_test_nor_all = np.array([])
    confidence_test_nor_all = np.array([])
    logits_adv_all = np.reshape(np.array([]), (0, 64))
    logits_nor_all = np.reshape(np.array([]), (0, 64))
    labels_adv_all = np.array([])
    labels_true_all = np.array([])
    labels_nor_all = np.array([])
    L2_distance = np.array([])
    nor_img_all = np.reshape(np.array([]), (0, image_size,image_size,num_channel))
    adv_img_all = np.reshape(np.array([]), (0, image_size,image_size,num_channel))

    print('Num of sample per eps is %d' % (num_sample))

    #Construct carlini adversarial samples
    model_carlini_adv = models_carlini(hps)

    #Construct predictions
    image = tf.placeholder(tf.float32,shape=[hps.batch_size, image_size, image_size,
                                             num_channel])############MNIST and CIFAR10 are different ar here
    adv_image = tf.placeholder(tf.float32,shape=[hps.batch_size, image_size, image_size,
                                                 num_channel])############MNIST and CIFAR10 are different ar here
    predict = tf.placeholder(tf.float32,shape=[hps.batch_size, 10])
    logit_nor,tsne_logit_nor = model_carlini_adv.predict(image,tsne_logits=True)
    logit_adv,tsne_logit_adv = model_carlini_adv.predict(adv_image,tsne_logits=True)
    predict_nor = tf.nn.softmax(logit_nor)
    predict_adv = tf.nn.softmax(logit_adv)

    # Calculate entropy
    argmax_y_onehot = tf.one_hot(tf.argmax(predict, 1), 10, on_value=0.0, off_value=1.0, axis=-1)
    normalized_y_nonmaximal = tf.reduce_sum(predict * argmax_y_onehot, 1)
    entropy = tf.reduce_sum(-tf.log(predict) * predict * argmax_y_onehot,1) / normalized_y_nonmaximal + tf.log(normalized_y_nonmaximal)

    for k in range(1):
        result_dict = loadmat('kernel_para_'+FLAGS.dataset+'/kernel1000_for_attack_' + f1 + '.mat')
        result_dict_median = loadmat('kernel_para_'+FLAGS.dataset+'/kernel1000_median_for_attack_' + f1 + '.mat')
        # e_mean = result_dict['mean_logits_' + f1]  # 10X64
        # e_invcovar = result_dict['inv_covar_' + f1]  # 64X64X10
        e_kernel_train = result_dict['kernel_'+f1+'_for_attack'] #100X64X10
        e_median = result_dict_median['median_out']  # 10X1

        if FLAGS.attack_method == 'carliniL2':
            attack1 = attacks.carliniL2.CarliniL2(sess, model_carlini_adv, batch_size=10, max_iterations=10,targeted=True,
                                                         confidence=0, initial_const=1.0,binary_search_steps=9)
            attack2 = None

        elif FLAGS.attack_method == 'carliniL2_highcon':
            attack1 = attacks.carliniL2.CarliniL2(sess, model_carlini_adv, batch_size=10, max_iterations=10000,targeted=True,
                                                         confidence=10, initial_const=1.0,binary_search_steps=9)
            attack2 = None

        elif FLAGS.attack_method == 'carliniL2_highden':
            attack1 = attacks.carliniL2.CarliniL2(sess, model_carlini_adv, batch_size=1,
                                                  max_iterations=5000,
                                                  targeted=True,
                                                  initial_const=1.0,
                                                  confidence=0, binary_search_steps=3)
            attack2 = attacks.carliniL2_specific.CarliniL2_specific(sess, model_carlini_adv, batch_size=1,
                                                                    max_iterations=10000,
                                                                    targeted=True,
                                                                    initial_const=1.0,
                                                                    confidence=0, binary_search_steps=8,
                                                                    extra_loss=True
                                                                    , e_kernel_train=e_kernel_train,
                                                                    e_median=e_median,
                                                                    sigma2=sigma2)

        elif FLAGS.attack_method == 'carliniL2_specific':
            attack1 = attacks.carliniL2.CarliniL2(sess, model_carlini_adv, batch_size=1,
                                                  max_iterations=5000,
                                                  targeted=True,
                                                  initial_const=10.0,
                                                  confidence=5, binary_search_steps=3)
            attack2 = attacks.carliniL2_specific.CarliniL2_specific(sess, model_carlini_adv, batch_size=1,
                              max_iterations=10000,
                              targeted=True,
                              initial_const=100.0,
                              confidence=5, binary_search_steps=9, extra_loss=True
                              , e_kernel_train=e_kernel_train ,
                              e_median = e_median,
                              sigma2 = sigma2)
        else:
            print('Error!!!!')
            attack1 = None
            attack2 = None

        success = 0
        efficient = 0
        L2_distance_print = 0

        for i in six.moves.range(FLAGS.eval_batch_count):
            time_start = time.time()
            (nor_img,true_label) = sess.run([images,labels])

            #Crafting target labels
            target_lab = np.zeros((hps.batch_size, 10))
            for j in range(hps.batch_size):
                r = np.random.random_integers(0, 9)
                while r == np.argmax(true_label[j]):
                    r = np.random.random_integers(0, 9)
                target_lab[j, r] = 1

            (predict_NOR, logits_part_nor) = sess.run(
                [predict_nor, tsne_logit_nor],
                feed_dict={image: nor_img}
            )
            #Attack1, craft adversarial samples in oblivious attack
            adv_img,succ = attack1.attack(nor_img, target_lab,predict_NOR)


            #Attack, craft adversarial samples in white-box attack
            if FLAGS.attack_method == 'carliniL2_specific' or FLAGS.attack_method == 'carliniL2_highden':
                if succ[0] == 1:
                    is_succ = 'Success'
                else:
                    is_succ = 'Fail'
                print('Finish attack 1. The %d batch in total %d(%f sec) %s' % (
                    i, FLAGS.eval_batch_count, time.time() - time_start,is_succ))
                time_start = time.time()
                adv_img, succ, log_density_ratio = attack2.attack(nor_img, adv_img, target_lab,predict_NOR)
                if succ[0] == 1:
                    is_succ = 'Success'
                else:
                    is_succ = 'Fail'
                print('Finish attack 2. The %d batch in total %d(%f sec) %s' % (
                    i, FLAGS.eval_batch_count, time.time() - time_start, is_succ))
            else:
                print('The %d batch in total %d, the eps = %f (%f sec)' % (
                    i, FLAGS.eval_batch_count, 0.05 * k, time.time() - time_start))

            #Local logits
            (predict_ADV,logits_part_adv) = sess.run(
                [predict_adv, tsne_logit_adv],feed_dict={adv_image:adv_img}
            )

            #Local entropy and confidence for nor_img
            (entropy_test_nor_help,labels_nor_help,confidence_test_nor_help) = sess.run(
                [entropy,tf.argmax(predict,axis=1),tf.reduce_max(predict, axis=1)],feed_dict={predict:predict_NOR}
            )

            # Local entropy and confidence for adv_img
            (entropy_test_adv_help, labels_adv_help, confidence_test_adv_help) = sess.run(
                [entropy, tf.argmax(predict, axis=1), tf.reduce_max(predict, axis=1)], feed_dict={predict: predict_ADV}
            )

            if FLAGS.attack_method == 'carliniL2_specific' or FLAGS.attack_method == 'carliniL2_highden':
                print('Log-density-ratio in attacking function of nor/adv is %f'%np.sum(log_density_ratio))
                m_tsne_logits_adv = (copy.copy(logits_part_adv)).reshape((1, 64))
                m_tsne_logits_adv = np.repeat(m_tsne_logits_adv,100,axis=0)
                kernel_train = (copy.copy(e_kernel_train[:,:,np.argmax(target_lab)])).reshape((100,64))
                log_density_ratio2 = -np.log(1e-30+np.mean(np.exp(-np.sum(np.square(m_tsne_logits_adv
                                                                      - kernel_train), axis=1) / sigma2),
                                      axis=0)) + np.log(e_median[np.argmax(target_lab)])
                # m_tsne_logits_adv = (copy.copy(logits_part_adv-e_mean[np.argmax(target_lab)])).reshape((64,1))
                # inter_mat_adv = np.matmul(e_invcovar[:,:,np.argmax(target_lab)].reshape((64,64)), m_tsne_logits_adv)
                # m_tsne_logits_nor = (copy.copy(logits_part_nor-e_mean[labels_nor_help])).reshape((64,1))
                # inter_mat_nor = np.matmul(e_invcovar[:,:,labels_nor_help].reshape((64,64)), m_tsne_logits_nor)
                # log_density_ratio2 = np.matmul(m_tsne_logits_adv.reshape((1,64)), inter_mat_adv) \
                #                          - np.matmul(m_tsne_logits_nor.reshape((1,64)), inter_mat_nor)
                #log_density_ratio2 = np.matmul(m_tsne_logits_adv.reshape((1, 64)), inter_mat_adv)+e_median[np.argmax(target_lab)]
                print('Log-density-ratio in saving results of nor/adv is %f'%np.sum(log_density_ratio2))

            entropy_test_adv_all = np.concatenate((entropy_test_adv_all,entropy_test_adv_help),axis=0)
            confidence_test_adv_all = np.concatenate((confidence_test_adv_all,confidence_test_adv_help),axis=0)
            entropy_test_nor_all = np.concatenate((entropy_test_nor_all, entropy_test_nor_help), axis=0)
            confidence_test_nor_all = np.concatenate((confidence_test_nor_all, confidence_test_nor_help), axis=0)
            logits_nor_all = np.concatenate((logits_nor_all, logits_part_nor), axis=0)
            labels_nor_all = np.concatenate((labels_nor_all, labels_nor_help), axis=0)
            logits_adv_all = np.concatenate((logits_adv_all,logits_part_adv),axis=0)
            labels_adv_all = np.concatenate((labels_adv_all, labels_adv_help), axis=0)
            labels_true_all = np.concatenate((labels_true_all, np.argmax(true_label,axis=1)), axis=0)
            L2_distance = np.concatenate((L2_distance,np.sqrt(np.mean(np.square(nor_img-adv_img),axis=(1,2,3)))), axis=0)
            nor_img_all = np.concatenate((nor_img_all,nor_img),axis=0)
            adv_img_all = np.concatenate((adv_img_all,adv_img),axis=0)

            #Efficient index refers to the indexes that are correctly classified and misclassified as adversarial samples
            efficient_index = succ*np.equal(np.argmax(true_label, axis=1),labels_nor_help)
            if FLAGS.attack_method != 'carliniL2_specific'or FLAGS.attack_method == 'carliniL2_highden':
                print('Num of attacking success is %d'%(np.sum(succ)))
            efficient += np.sum(efficient_index)
            L2_distance_print += np.sum(efficient_index*np.sqrt(np.mean(np.square(nor_img - adv_img), axis=(1, 2, 3))), axis=0)

        L2_distance_print = L2_distance_print/efficient
        k_index_begin = k*num_sample
        k_index_end = (k+1)*num_sample

        # Show local results
        precision_nor = np.mean(np.equal(labels_nor_all[k_index_begin:k_index_end],labels_true_all[k_index_begin:k_index_end]))
        precision_adv = np.mean(np.equal(labels_adv_all[k_index_begin:k_index_end],labels_true_all[k_index_begin:k_index_end]))
        mean_confidence_nor = np.mean(confidence_test_nor_all[k_index_begin:k_index_end])
        mean_confidence_adv = np.mean(confidence_test_adv_all[k_index_begin:k_index_end])
        mean_entropy_nor = np.mean(entropy_test_nor_all[k_index_begin:k_index_end])
        mean_entropy_adv = np.mean(entropy_test_adv_all[k_index_begin:k_index_end])

        print('Precision on nor images is %f, on adv images is %f' % (precision_nor, precision_adv))
        print('Confidence on nor images is %f, on adv images is %f' % (mean_confidence_nor, mean_confidence_adv))
        print('non-ME on nor images is %f, on adv images is %f' % (mean_entropy_nor, mean_entropy_adv))
        print('Average L2-distance between nor and adv imgs is %f'%(L2_distance_print))
        print('Total success num of attack 1 is %d'%(success))
        print('Total efficient num of attack 1 is %d' % (efficient))

    # # Save results
    # np.savetxt(FLAGS.attack_method + '_' + FLAGS.dataset + '/'+f1+'/entropy_nor', entropy_test_nor_all)
    # np.savetxt(FLAGS.attack_method + '_' + FLAGS.dataset + '/'+f1+ '/confidence_nor', confidence_test_nor_all)
    # np.savetxt(FLAGS.attack_method + '_' + FLAGS.dataset +  '/'+f1+'/entropy_adv', entropy_test_adv_all)
    # np.savetxt(FLAGS.attack_method + '_' + FLAGS.dataset + '/'+f1+ '/confidence_adv', confidence_test_adv_all)
    # np.savetxt(FLAGS.attack_method + '_' + FLAGS.dataset + '/'+f1+ '/logits_nor', logits_nor_all)
    # np.savetxt(FLAGS.attack_method + '_' + FLAGS.dataset + '/'+f1+ '/logits_adv', logits_adv_all)
    # np.savetxt(FLAGS.attack_method + '_' + FLAGS.dataset +  '/'+f1+'/labels_nor', labels_nor_all)
    # np.savetxt(FLAGS.attack_method + '_' + FLAGS.dataset +  '/'+f1+'/labels_adv', labels_adv_all)
    # np.savetxt(FLAGS.attack_method + '_' + FLAGS.dataset +  '/'+f1+'/labels_true', labels_true_all)
    # np.savetxt(FLAGS.attack_method + '_' + FLAGS.dataset + '/' + f1 + '/L2_distance', L2_distance)
    # np.save(FLAGS.attack_method + '_' + FLAGS.dataset + '/' + f1 + '/nor_img.npy', nor_img_all)
    # np.save(FLAGS.attack_method + '_' + FLAGS.dataset + '/' + f1 + '/adv_img.npy', adv_img_all)

    # #Save img
    # nor_img_all = nor_img_all + 0.5
    # adv_img_all = adv_img_all + 0.5
    # noise_img_all = 0.5 * (adv_img_all - nor_img_all + 1.0)

    # if FLAGS.dataset=='cifar10':
        # for i in range(nor_img_all.shape[0]):
            # imsave(FLAGS.attack_method + '_' + FLAGS.dataset + '/' + f1 + '/nor_img/nor_img_' + str(i) + '.png', nor_img_all[i])
            # imsave(FLAGS.attack_method + '_' + FLAGS.dataset + '/' + f1 + '/adv_img/adv_img_' + str(i) + '.png', adv_img_all[i])
            # imsave(FLAGS.attack_method + '_' + FLAGS.dataset + '/' + f1 + '/noise_img/noise_img_' + str(i) + '.png', noise_img_all[i])
    # elif FLAGS.dataset=='mnist':
        # for i in range(nor_img_all.shape[0]):
            # imsave(FLAGS.attack_method + '_' + FLAGS.dataset + '/' + f1 + '/nor_img/nor_img_' + str(i) + '.png', nor_img_all[i,:,:,0])
            # imsave(FLAGS.attack_method + '_' + FLAGS.dataset + '/' + f1 + '/adv_img/adv_img_' + str(i) + '.png', adv_img_all[i,:,:,0])
            # imsave(FLAGS.attack_method + '_' + FLAGS.dataset + '/' + f1 + '/noise_img/noise_img_' + str(i) + '.png', noise_img_all[i, :, :, 0])
    return None

def apply_attack_loop(hps):
    #Construct graph
    images, labels = input_name.build_input(
        FLAGS.dataset, FLAGS.eval_data_path, hps.batch_size, FLAGS.mode)#FLAGS.mode='attack', batch_size=200
    Res = model_name.ResNet(hps, images, FLAGS.mode, Reuse=False)
    Res.build_graph()
    saver = tf.train.Saver()

    #Open session and restore checkpoint
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    tf.train.start_queue_runners(sess)

    ckpt_state = tf.train.get_checkpoint_state(FLAGS.log_root)  # Choose dir according to rt
    tf.logging.info('Loading checkpoint %s', ckpt_state.model_checkpoint_path)

    num_sample = hps.batch_size*FLAGS.eval_batch_count

    # Initialize results to save
    entropy_test_adv_all = np.array([])
    confidence_test_adv_all = np.array([])
    entropy_test_nor_all = np.array([])
    confidence_test_nor_all = np.array([])
    logits_adv_all = np.reshape(np.array([]), (0, 64))
    logits_nor_all = np.reshape(np.array([]), (0, 64))
    labels_adv_all = np.array([])
    labels_true_all = np.array([])
    labels_nor_all = np.array([])
    L2_distance = np.array([])
    nor_img_all = np.reshape(np.array([]), (0, image_size, image_size, num_channel))
    adv_img_all = np.reshape(np.array([]), (0, image_size, image_size, num_channel))
    print('Num of sample per eps is %d' % (num_sample))

    # Construct predictions
    image = tf.placeholder(tf.float32, shape=[hps.batch_size, image_size, image_size,
                                              num_channel])  ############MNIST and CIFAR10 are different ar here
    adv_image = tf.placeholder(tf.float32, shape=[hps.batch_size, image_size, image_size,
                                              num_channel])  ############MNIST and CIFAR10 are different ar here
    predict = tf.placeholder(tf.float32, shape=[hps.batch_size, 10])
    predict_nor, tsne_logit_nor = models(hps, image, FLAGS.RCE_train, logits=False, tsne_logits=True)
    predict_adv, tsne_logit_adv = models(hps, adv_image, FLAGS.RCE_train, logits=False, tsne_logits=True)

    # Calculate entropy
    argmax_y_onehot = tf.one_hot(tf.argmax(predict, 1), 10, on_value=0.0, off_value=1.0, axis=-1)
    normalized_y_nonmaximal = tf.reduce_sum(predict * argmax_y_onehot, 1)
    entropy = tf.reduce_sum(-tf.log(predict) * predict * argmax_y_onehot, 1) / normalized_y_nonmaximal + tf.log(
        normalized_y_nonmaximal)

    for k in range(10):
        adv_image_craft = adv_craft_func(hps, image, FLAGS.attack_method, eps=0.02 * k + 0.02, RCE_train=FLAGS.RCE_train)
        #adv_image_craft = adv_craft_func(hps, image, FLAGS.attack_method, eps=0.04,RCE_train=FLAGS.RCE_train)
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, ckpt_state.model_checkpoint_path)

        for i in six.moves.range(FLAGS.eval_batch_count):
            time_start = time.time()
            (nor_img,true_label) = sess.run([images,labels])
            adv_img = sess.run(adv_image_craft,feed_dict={image:nor_img})

            # Local logits
            (predict_NOR, predict_ADV, logits_part_nor, logits_part_adv) = sess.run(
                [predict_nor, predict_adv, tsne_logit_nor, tsne_logit_adv],
                feed_dict={image: nor_img, adv_image: adv_img}
            )

            # Local entropy and confidence for nor_img
            (entropy_test_nor_help, labels_nor_help, confidence_test_nor_help) = sess.run(
                [entropy, tf.argmax(predict, axis=1), tf.reduce_max(predict, axis=1)], feed_dict={predict: predict_NOR}
            )

            # Local entropy and confidence for adv_img
            (entropy_test_adv_help, labels_adv_help, confidence_test_adv_help) = sess.run(
                [entropy, tf.argmax(predict, axis=1), tf.reduce_max(predict, axis=1)], feed_dict={predict: predict_ADV}
            )

            entropy_test_adv_all = np.concatenate((entropy_test_adv_all, entropy_test_adv_help), axis=0)
            confidence_test_adv_all = np.concatenate((confidence_test_adv_all, confidence_test_adv_help), axis=0)
            entropy_test_nor_all = np.concatenate((entropy_test_nor_all, entropy_test_nor_help), axis=0)
            confidence_test_nor_all = np.concatenate((confidence_test_nor_all, confidence_test_nor_help), axis=0)
            logits_nor_all = np.concatenate((logits_nor_all, logits_part_nor), axis=0)
            labels_nor_all = np.concatenate((labels_nor_all, labels_nor_help), axis=0)
            logits_adv_all = np.concatenate((logits_adv_all, logits_part_adv), axis=0)
            labels_adv_all = np.concatenate((labels_adv_all, labels_adv_help), axis=0)
            labels_true_all = np.concatenate((labels_true_all, np.argmax(true_label, axis=1)), axis=0)
            L2_distance = np.concatenate((L2_distance,np.sqrt(np.mean(np.square(nor_img-adv_img),axis=(1,2,3)))), axis=0)
            nor_img_all = np.concatenate((nor_img_all, nor_img), axis=0)
            adv_img_all = np.concatenate((adv_img_all, adv_img), axis=0)
            print('The %d batch in total %d, the eps = %f (%f sec)' % (
                i, FLAGS.eval_batch_count, 0.02 * k + 0.02, time.time() - time_start))
        k_index_begin = k * num_sample
        k_index_end = (k + 1) * num_sample

        # Show local results
        precision_nor = np.mean(
            np.equal(labels_nor_all[k_index_begin:k_index_end], labels_true_all[k_index_begin:k_index_end]))
        precision_adv = np.mean(
            np.equal(labels_adv_all[k_index_begin:k_index_end], labels_true_all[k_index_begin:k_index_end]))
        mean_confidence_nor = np.mean(confidence_test_nor_all[k_index_begin:k_index_end])
        mean_confidence_adv = np.mean(confidence_test_adv_all[k_index_begin:k_index_end])
        mean_entropy_nor = np.mean(entropy_test_nor_all[k_index_begin:k_index_end])
        mean_entropy_adv = np.mean(entropy_test_adv_all[k_index_begin:k_index_end])

        print('Precision on nor images is %f, on adv images is %f' % (precision_nor, precision_adv))
        print('Confidence on nor images is %f, on adv images is %f' % (mean_confidence_nor, mean_confidence_adv))
        print('non-ME on nor images is %f, on adv images is %f' % (mean_entropy_nor, mean_entropy_adv))
        print('Average L2-distance between nor and adv imgs is %f'%(np.mean(L2_distance)))

    # Save results
    if FLAGS.save_pwd ==None:
        np.savetxt(FLAGS.attack_method + '_' + FLAGS.dataset + '/' + f1 + '/entropy_nor', entropy_test_nor_all)
        np.savetxt(FLAGS.attack_method + '_' + FLAGS.dataset + '/' + f1 + '/confidence_nor', confidence_test_nor_all)
        np.savetxt(FLAGS.attack_method + '_' + FLAGS.dataset + '/' + f1 + '/entropy_adv', entropy_test_adv_all)
        np.savetxt(FLAGS.attack_method + '_' + FLAGS.dataset + '/' + f1 + '/confidence_adv', confidence_test_adv_all)
        np.savetxt(FLAGS.attack_method + '_' + FLAGS.dataset + '/' + f1 + '/logits_nor', logits_nor_all)
        np.savetxt(FLAGS.attack_method + '_' + FLAGS.dataset + '/' + f1 + '/logits_adv', logits_adv_all)
        np.savetxt(FLAGS.attack_method + '_' + FLAGS.dataset + '/' + f1 + '/labels_nor', labels_nor_all)
        np.savetxt(FLAGS.attack_method + '_' + FLAGS.dataset + '/' + f1 + '/labels_adv', labels_adv_all)
        np.savetxt(FLAGS.attack_method + '_' + FLAGS.dataset + '/' + f1 + '/labels_true', labels_true_all)
        np.savetxt(FLAGS.attack_method + '_' + FLAGS.dataset + '/' + f1 + '/L2_distance', L2_distance)
        np.save(FLAGS.attack_method + '_' + FLAGS.dataset + '/' + f1 + '/nor_img.npy', nor_img_all)
        np.save(FLAGS.attack_method + '_' + FLAGS.dataset + '/' + f1 + '/adv_img.npy', adv_img_all)
    else:
        np.savetxt(FLAGS.save_pwd + '/entropy_nor', entropy_test_nor_all)
        np.savetxt(FLAGS.save_pwd + '/confidence_nor', confidence_test_nor_all)
        np.savetxt(FLAGS.save_pwd + '/entropy_adv', entropy_test_adv_all)
        np.savetxt(FLAGS.save_pwd + '/confidence_adv', confidence_test_adv_all)
        np.savetxt(FLAGS.save_pwd + '/logits_nor', logits_nor_all)
        np.savetxt(FLAGS.save_pwd + '/logits_adv', logits_adv_all)
        np.savetxt(FLAGS.save_pwd + '/labels_nor', labels_nor_all)
        np.savetxt(FLAGS.save_pwd + '/labels_adv', labels_adv_all)
        np.savetxt(FLAGS.save_pwd + '/labels_true', labels_true_all)
        np.savetxt(FLAGS.save_pwd + '/L2_distance', L2_distance)
        np.save(FLAGS.save_pwd + '/nor_img.npy', nor_img_all)
        np.save(FLAGS.save_pwd + '/adv_img.npy', adv_img_all)

    return None

def main(_):
    print('attacking method is %s' % (FLAGS.attack_method))
    print('mode is %s'%(FLAGS.mode))

    if FLAGS.attack_method == 'carliniL2' or FLAGS.attack_method == 'carliniL2_highcon' \
            or FLAGS.attack_method == 'carliniL2_specific' or FLAGS.attack_method == 'carliniL2_highden':
        is_carliniL2 = True
    else:
        is_carliniL2 = False

    if FLAGS.attack_method == 'jsma' or FLAGS.attack_method == 'smda'\
            or FLAGS.attack_method == 'carliniL2_specific' or FLAGS.attack_method == 'carliniL2_highden':
        batch_size = 1
        num_batch = 1000
    elif FLAGS.attack_method == 'fgsm' or FLAGS.attack_method == 'tgsm' or FLAGS.attack_method == 'bim' or FLAGS.attack_method == 'random':
        batch_size = 200
        num_batch = 5
    elif FLAGS.attack_method == 'carliniL2'or FLAGS.attack_method == 'carliniL2_highcon':
        batch_size = 10
        num_batch = 100
    else:
        print('Undefined attacking method')
        batch_size = None
        num_batch = None

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

    if FLAGS.mode == 'attack':
        if is_carliniL2 == True:
            apply_attack_carlini(hps)
        else:
            apply_attack_loop(hps)

    elif FLAGS.mode == 'tSNE_logits':
        if is_carliniL2 == True:
            tSNE_visual_carliniLi(hps,num_batch)
        else:
            tSNE_visual(hps,num_batch)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()