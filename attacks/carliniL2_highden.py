## l2_attack.py -- attack a network optimizing for l_2 distance
##
## Copyright (C) 2016, Nicholas Carlini <nicholas@carlini.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.

import sys
import tensorflow as tf
import numpy as np


BINARY_SEARCH_STEPS = 9  # number of times to adjust the constant with binary search
MAX_ITERATIONS = 1000  # number of iterations to perform gradient descent
ABORT_EARLY = True  # if we stop improving, abort gradient descent early
LEARNING_RATE = 1e-2  # larger values converge faster to less accurate results
TARGETED = True  # should we target one specific class? or just be wrong?
CONFIDENCE = 0  # how strong the adversarial example should be
INITIAL_CONST = 1.0  # the initial constant c to pick as a first guess


class CarliniL2_highden:
    def __init__(self, sess, model, batch_size=1, confidence=CONFIDENCE,
                 targeted=TARGETED, learning_rate=LEARNING_RATE,
                 binary_search_steps=BINARY_SEARCH_STEPS, max_iterations=MAX_ITERATIONS,
                 abort_early=ABORT_EARLY,
                 initial_const=INITIAL_CONST,
                 boxmin=-0.5, boxmax=0.5, extra_loss=True,e_mean=None,e_invcovar=None,e_median=None):
        """
        The L_2 optimized attack.
        This attack is the most efficient and should be used as the primary
        attack to evaluate potential defenses.
        Returns adversarial examples for the supplied model.
        confidence: Confidence of adversarial examples: higher produces examples
          that are farther away, but more strongly classified as adversarial.
        batch_size: Number of attacks to run simultaneously.
        targeted: True if we should perform a targetted attack, False otherwise.
        learning_rate: The learning rate for the attack algorithm. Smaller values
          produce better results but are slower to converge.
        binary_search_steps: The number of times we perform binary search to
          find the optimal tradeoff-constant between distance and confidence.
        max_iterations: The maximum number of iterations. Larger values are more
          accurate; setting too small will require a large learning rate and will
          produce poor results.
        abort_early: If true, allows early aborts if gradient descent gets stuck.
        initial_const: The initial tradeoff-constant to use to tune the relative
          importance of distance and confidence. If binary_search_steps is large,
          the initial constant is not important.
        boxmin: Minimum pixel value (default -0.5).
        boxmax: Maximum pixel value (default 0.5).
        """

        image_size, num_channels, num_labels = model.image_size, model.num_channels, model.num_labels
        self.sess = sess
        self.TARGETED = targeted
        self.LEARNING_RATE = learning_rate
        self.MAX_ITERATIONS = max_iterations
        self.BINARY_SEARCH_STEPS = binary_search_steps
        self.ABORT_EARLY = abort_early
        self.CONFIDENCE = confidence
        self.initial_const = initial_const
        self.batch_size = batch_size

        self.repeat = binary_search_steps >= 10

        shape = (batch_size, image_size, image_size, num_channels)

        # the variable we're going to optimize over
        modifier = tf.Variable(np.zeros(shape, dtype=np.float32))

        # these are variables to be more efficient in sending data to tf
        self.origs = tf.Variable(np.zeros(shape), dtype=tf.float32)
        self.timg = tf.Variable(np.zeros(shape), dtype=tf.float32)
        self.tlab = tf.Variable(np.zeros((batch_size, num_labels)), dtype=tf.float32)
        self.const = tf.Variable(np.zeros(batch_size), dtype=tf.float32)
        self.const2 = tf.Variable(np.zeros(batch_size), dtype=tf.float32)
        self.mean_adv = tf.Variable(np.zeros((64,1)), dtype=tf.float32)  ##############64 is specifically for resnet
        self.inv_covar_adv = tf.Variable(np.zeros((64, 64)), dtype=tf.float32)  ##############64 is specifically for resnet
        self.mean_nor = tf.Variable(np.zeros((64,1)), dtype=tf.float32)  ##############64 is specifically for resnet
        self.inv_covar_nor = tf.Variable(np.zeros((64, 64)), dtype=tf.float32)  ##############64 is specifically for resnet
        # self.median = tf.Variable(np.zeros(1), dtype=tf.float32)  ##############64 is specifically for resnet

        # and here's what we use to assign them
        self.assign_origs = tf.placeholder(tf.float32, shape)
        self.assign_timg = tf.placeholder(tf.float32, shape)
        self.assign_tlab = tf.placeholder(tf.float32, (batch_size, num_labels))
        self.assign_const = tf.placeholder(tf.float32, [batch_size])
        self.assign_const2 = tf.placeholder(tf.float32, [batch_size])
        self.assign_mean_adv = tf.placeholder(tf.float32, [64,1])##############64 is specifically for resnet
        self.assign_inv_covar_adv = tf.placeholder(tf.float32, [64,64])##############64 is specifically for resnet
        self.assign_mean_nor = tf.placeholder(tf.float32, [64,1])  ##############64 is specifically for resnet
        self.assign_inv_covar_nor = tf.placeholder(tf.float32, [64, 64])  ##############64 is specifically for resnet
        #self.assign_median = tf.placeholder(tf.float32, [1])##############64 is specifically for resnet

        # the resulting image, tanh'd to keep bounded from boxmin to boxmax
        self.boxmul = (boxmax - boxmin) / 2.
        self.boxplus = (boxmin + boxmax) / 2.
        self.newimg = tf.tanh(modifier + self.timg) * self.boxmul + self.boxplus

        # prediction BEFORE-SOFTMAX of the model
        self.output, tsne_logit_adv = model.predict(self.newimg, tsne_logits=True)
        _, tsne_logit_nor = model.predict(self.origs, tsne_logits=True)

        self.tsne_logit_adv = tf.reshape(tsne_logit_adv, [64, 1])
        self.tsne_logit_nor = tf.reshape(tsne_logit_nor, [64, 1])

        self.softmax_output = tf.nn.softmax(self.output)

        # distance to the input data
        self.l2dist = tf.reduce_sum(tf.square(self.newimg - (tf.tanh(self.timg) * self.boxmul + self.boxplus)),
                                    [1, 2, 3])

        # compute the probability of the label class versus the maximum other
        real = tf.reduce_sum((self.tlab) * self.output, 1)
        other = tf.reduce_max((1 - self.tlab) * self.output - (self.tlab * 10000), 1)

        if self.TARGETED:
            # if targetted, optimize for making the other class most likely
            loss1 = tf.maximum(0.0, other - real + self.CONFIDENCE)
        else:
            # if untargeted, optimize for making this class least likely.
            loss1 = tf.maximum(0.0, real - other + self.CONFIDENCE)

        # sum up the losses
        self.loss2 = tf.reduce_sum(self.l2dist)
        self.loss1 = tf.reduce_sum(self.const * loss1)
        if extra_loss == True:
            self.e_mean = e_mean  # 10X64
            self.e_invcovar = e_invcovar # 64X64X10
            #self.e_median = e_median  # 10X1

            m_tsne_logits_adv = self.tsne_logit_adv-self.mean_adv
            inter_mat_adv = tf.matmul(self.inv_covar_adv,m_tsne_logits_adv)
            m_tsne_logits_nor = self.tsne_logit_nor - self.mean_nor
            inter_mat_nor = tf.matmul(self.inv_covar_nor, m_tsne_logits_nor)
            self.log_density_ratio = tf.matmul(m_tsne_logits_adv,inter_mat_adv,transpose_a=True)\
                                         -tf.matmul(m_tsne_logits_nor,inter_mat_nor,transpose_a=True)
            self.extra_loss = tf.maximum(self.log_density_ratio,0.0)
        else:
            self.extra_loss = 0
        self.loss = self.loss1 + self.loss2+self.const*tf.reduce_sum(self.extra_loss)

        # Setup the adam optimizer and keep track of variables we're creating
        start_vars = set(x.name for x in tf.global_variables())
        optimizer = tf.train.AdamOptimizer(self.LEARNING_RATE)
        self.train = optimizer.minimize(self.loss, var_list=[modifier])
        end_vars = tf.global_variables()
        new_vars = [x for x in end_vars if x.name not in start_vars]

        # these are the variables to initialize when we run
        self.setup = []
        self.setup.append(self.origs.assign(self.assign_origs))
        self.setup.append(self.timg.assign(self.assign_timg))
        self.setup.append(self.tlab.assign(self.assign_tlab))
        self.setup.append(self.const.assign(self.assign_const))
        self.setup.append(self.const2.assign(self.assign_const2))
        self.setup.append(self.mean_adv.assign(self.assign_mean_adv))
        self.setup.append(self.inv_covar_adv.assign(self.assign_inv_covar_adv))
        self.setup.append(self.mean_nor.assign(self.assign_mean_nor))
        self.setup.append(self.inv_covar_nor.assign(self.assign_inv_covar_nor))
        #self.setup.append(self.median.assign(self.assign_median))

        self.init = tf.variables_initializer(var_list=[modifier] + new_vars)

    def attack(self, origs, imgs, targets, predict):
        """
        Perform the L_2 attack on the given images for the given targets.
        If self.targeted is true, then the targets represents the target labels.
        If self.targeted is false, then targets are the original class labels.
        """
        r = []
        on = []
        ratio = []
        #print('go up to', len(imgs))
        for i in range(0, len(imgs), self.batch_size):
            #print('tick', i)
            attack,attack_on,log_ratio = self.attack_batch(origs[i:i+self.batch_size],
                                       imgs[i:i + self.batch_size],
                                       targets[i:i + self.batch_size],
                                        predict[i:i + self.batch_size])

            r.extend(attack)
            on.extend(attack_on)
            ratio.extend((log_ratio))
        return np.array(r), np.array(on),np.array(ratio)

    def attack_batch(self, origs, imgs, labs, pre_labs):
        """
        Run the attack on a batch of images and labels.
        """
        batch_size = self.batch_size

        def compare(x,y):
            if not isinstance(x, (float, int, np.int64)):
                x = np.copy(x)
                x = np.argmax(x)
            if self.TARGETED:
                return x == y
            else:
                return x != y

        # convert to tanh-space
        imgs = np.arctanh((imgs - self.boxplus) / self.boxmul * 0.999999)
        #origs = np.arctanh((origs - self.boxplus) / self.boxmul * 0.999999)

        # set the lower and upper bounds accordingly
        lower_bound = np.zeros(batch_size)
        CONST = np.ones(batch_size) * self.initial_const
        upper_bound = np.ones(batch_size) * 1e10

        CONST2 = np.ones(batch_size)*self.initial_const

        # the best l2, score, and image attack
        o_bestl2 = [1e10] * batch_size
        o_bestscore = [-1] * batch_size
        o_bestattack = [np.zeros(imgs[0].shape)] * batch_size
        o_bestattack_on = [0] * batch_size
        o_log_density_ratio = [1e10] * batch_size

        for outer_step in range(self.BINARY_SEARCH_STEPS):
            #print(o_bestl2)
            # completely reset adam's internal state.
            self.sess.run(self.init)
            batch = imgs[:batch_size]
            batchlab = labs[:batch_size]
            batchlab_nor = pre_labs[:batch_size]


            bestl2 = [1e10] * batch_size
            bestscore = [-1] * batch_size

            # The last iteration (if we run many steps) repeat the search once.
            if self.repeat == True and outer_step == self.BINARY_SEARCH_STEPS - 1:
                CONST = upper_bound

            s_index = np.argmax(batchlab)
            s_index_nor = np.argmax(batchlab_nor)

            # set the variables so that we don't have to send them over again
            self.sess.run(self.setup, {self.assign_timg: batch,
                                       self.assign_origs: origs,
                                       self.assign_tlab: batchlab,
                                       self.assign_const: CONST,
                                       self.assign_const2: CONST2,
                                       self.assign_mean_adv: np.reshape(self.e_mean[s_index],(64,1)),
                                       #self.assign_median: self.e_median[s_index],
                                       self.assign_inv_covar_adv: np.reshape(self.e_invcovar[:,:,s_index],(64,64)),
                                        self.assign_mean_nor: np.reshape(self.e_mean[s_index_nor], (64,1)),
                                       self.assign_inv_covar_nor: np.reshape(self.e_invcovar[:, :, s_index_nor], (64, 64))
                                       })

            prev = 1e20
            #print(self.sess.run(tf.reshape(self.tsne_logit_nor, (64,))))
            for iteration in range(self.MAX_ITERATIONS):
                # perform the attack
                _, l, l2s, scores, nimg, extra, log_density_ratio = self.sess.run([self.train, self.loss,
                                                                                   self.l2dist, self.softmax_output,
                                                                                   self.newimg, self.extra_loss,
                                                                                   self.log_density_ratio])

                # print out the losses every 10%
                # if iteration % (self.MAX_ITERATIONS // 10) == 0:
                #     print(iteration, self.sess.run((self.loss, self.loss1, self.loss2)))

                # check if we should abort search if we're getting nowhere.
                if self.ABORT_EARLY and iteration % (self.MAX_ITERATIONS // 10) == 0:
                    if l > prev * .9999:
                        break
                    prev = l

                # adjust the best result found so far
                for e, (l2, sc, ii) in enumerate(zip(l2s, scores, nimg)):
                    if l2 < bestl2[e] and compare(sc, np.argmax(batchlab[e])) and extra[e] <= 0:
                        bestl2[e] = l2
                        bestscore[e] = np.argmax(sc)
                    if l2 < o_bestl2[e] and compare(sc, np.argmax(batchlab[e])) and extra[e] <= 0:
                        o_bestl2[e] = l2
                        o_bestscore[e] = np.argmax(sc)
                        o_bestattack[e] = ii
                        o_bestattack_on[e] = 1
                        o_log_density_ratio[e] = log_density_ratio[e]

            # adjust the constant as needed
            for e in range(batch_size):
                if compare(bestscore[e], np.argmax(batchlab[e])) and bestscore[e] != -1:
                    # success, divide const by two
                    upper_bound[e] = min(upper_bound[e], CONST[e])
                    if upper_bound[e] < 1e9:
                        CONST[e] = (lower_bound[e] + upper_bound[e]) / 2
                else:
                    # failure, either multiply by 10 if no solution found yet
                    #          or do binary search with the known upper bound
                    lower_bound[e] = max(lower_bound[e], CONST[e])
                    if upper_bound[e] < 1e9:
                        CONST[e] = (lower_bound[e] + upper_bound[e]) / 2
                    else:
                        CONST[e] *= 10

        # return the best solution found
        o_bestl2 = np.array(o_bestl2)
        return o_bestattack,o_bestattack_on,o_log_density_ratio