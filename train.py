import tensorflow as tf
from cnn_archs import *
import cv2
import numpy as np
import os
import shutil
import argparse
import sys

def load_img(path):
    img = cv2.imread(path, -1)
    if img.shape[2] > 3:
        img = img[:, :, 0:3]
    img = img.astype(np.float32)
    img = img/127.5 - 1

    return img


def gen_data(n, scales, img, sess, all_x_hat_fake, all_x_hat_rec, z_fixed):
    H, W, C = img.shape

    h = int(round(H * pow(0.75, scales - 1)))
    w = int(round(W * pow(0.75, scales - 1)))

    curr_y = tf.image.resize_images(img, (h, w))
    curr_y = tf.expand_dims(curr_y, axis=0)
    curr_y = sess.run(curr_y)

    curr_x_fake = sess.run(tf.zeros_like(curr_y))
    curr_x_rec = sess.run(tf.zeros_like(curr_y))

    curr_z_fake = z_fixed + sess.run(tf.random_normal(tf.shape(curr_y)))
    # curr_z_rec = sess.run(tf.random_normal(tf.shape(curr_y)))
    curr_z_rec = z_fixed

    for i in range(1, n + 1):
        feed_prev = gen_feed_dict(scales - i, curr_x_fake, curr_x_rec, curr_y, curr_z_fake, curr_z_rec)

        h = int(round(H * pow(0.75, scales - i - 1)))
        w = int(round(W * pow(0.75, scales - i - 1)))

        x_hat_fake_prev = sess.run(all_x_hat_fake[i - 1], feed_dict= feed_prev)
        curr_x_fake = tf.image.resize_images(x_hat_fake_prev, (h, w))
        curr_x_fake = sess.run(curr_x_fake)

        x_hat_rec_prev = sess.run(all_x_hat_rec[i - 1], feed_dict= feed_prev)
        curr_x_rec = tf.image.resize_images(x_hat_rec_prev, (h, w))
        curr_x_rec = sess.run(curr_x_rec)

        curr_y = tf.image.resize_images(img, (h, w))
        curr_y = tf.expand_dims(curr_y, axis=0)
        curr_y = sess.run(curr_y)

        noise_level = np.sqrt(np.mean(np.square(curr_y - curr_x_rec)))

        curr_z_fake = sess.run(tf.random_normal(tf.shape(curr_x_fake)))*0.1*noise_level
        curr_z_rec = sess.run(tf.zeros_like(curr_x_rec))

    return curr_x_fake, curr_x_rec, curr_y, curr_z_fake, curr_z_rec

def gen_feed_dict(scale, x_fake, x_rec, y, z_fake, z_rec):
    feed = {'scale_' + str(scale) + '/y:0': y,
            'scale_' + str(scale) + '/x_fake:0': x_fake,
            'scale_' + str(scale) + '/x_rec:0': x_rec,
            'scale_' + str(scale) + '/z_fake:0': z_fake,
            'scale_' + str(scale) + '/z_rec:0': z_rec}

    return feed

def save_predictions(scale, scales, img, sess, all_x_hat_fake, all_x_hat_rec, z_fixed, results_dir):
    for i in range(10):
        curr_x_fake, curr_x_rec, curr_y, curr_z_fake, curr_z_rec = gen_data(scales - scale - 1, scales, img, sess, all_x_hat_fake, all_x_hat_rec, z_fixed)
        feed = gen_feed_dict(scale, curr_x_fake, curr_x_rec, curr_y, curr_z_fake, curr_z_rec)

        temp_x_rec = sess.run('scale_' + str(scale) + '/gen/x_hat_rec:0', feed_dict=feed)
        temp_x_rec = temp_x_rec[0, :, :, :] * 127.5 + 127.5
        temp_x_rec = temp_x_rec.astype(np.uint8)
        cv2.imwrite(results_dir + '/scale_' + str(scale) + '_rec_' + str(i) + '.png', temp_x_rec)

        temp_x_fake = sess.run('scale_' + str(scale) + '/gen/x_hat_fake:0', feed_dict=feed)
        temp_x_fake = temp_x_fake[0, :, :, :] * 127.5 + 127.5
        temp_x_fake = temp_x_fake.astype(np.uint8)
        cv2.imwrite(results_dir + '/scale_' + str(scale) + '_fake_' + str(i) + '.png', temp_x_fake)

def parse_argument(arg_list):
    if not arg_list:
        arg_list = ['-h']
        print('error - input required, see description below')

    parser = argparse.ArgumentParser(prog= 'train.py', description='SinGan tensorflow implementation')
    parser.add_argument('run', type= int, help= 'run number to distinguish different runs')
    args = parser.parse_args(arg_list)
    return args.run

if __name__ == '__main__':
    run = parse_argument(sys.argv[1:])

    img = load_img('D:/Shivesh/SinGAN/data/colusseum.png')
    # img = load_img('/storage/coda1/p-hl94/0/schaudhary9/testflight_data/SinGAN/data/colusseum.png')
    scales = 6

    lr = 0.0005
    training_iters = 75

    H, W, C = img.shape

    all_x_hat_fake = []
    all_x_hat_rec = []
    all_gen_loss = []
    all_disc_loss = []
    all_gen_opt = []
    all_disc_opt = []
    for i in range(scales):


        input_h = int(round(H*pow(0.75, scales - i - 1)))
        input_w = int(round(W*pow(0.75, scales - i - 1)))

        with tf.variable_scope('scale_' + str(scales - i - 1)):
            y = tf.placeholder("float", [None, input_h, input_w, img.shape[2]], name= 'y')
            x_fake = tf.placeholder("float", [None, input_h, input_w, img.shape[2]], name= 'x_fake')
            x_rec = tf.placeholder("float", [None, input_h, input_w, img.shape[2]], name= 'x_rec')
            z_fake = tf.placeholder("float", [None, input_h, input_w, img.shape[2]], name= 'z_fake')
            z_rec = tf.placeholder("float", [None, input_h, input_w, img.shape[2]], name= 'z_rec')


            with tf.variable_scope('gen', reuse= tf.AUTO_REUSE):
                x_hat_fake = generator_tf.conv_net(x_fake, z_fake)
                x_hat_fake = tf.identity(x_hat_fake, name= 'x_hat_fake')

                x_hat_rec = generator_tf.conv_net(x_rec, z_rec)
                x_hat_rec = tf.identity(x_hat_rec, name='x_hat_rec')

            alpha = tf.random_uniform(shape=[tf.shape(x_hat_fake)[0], 1, 1, 1], minval=0.0, maxval=1.0, name= 'alpha')
            interpolate = alpha * y + (1 - alpha) * x_hat_fake
            with tf.variable_scope('disc', reuse= tf.AUTO_REUSE):
                x_hat_p = discriminator_tf.conv_net(x_hat_fake)
                t_p = discriminator_tf.conv_net(y)
                dis_interpolate = discriminator_tf.conv_net(interpolate)

            rec_cost = tf.reduce_mean(tf.squared_difference(x_hat_rec, y))
            # fake_as_true = tf.ones_like(x_hat_p)
            # gen_adv_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=fake_as_true, logits=x_hat_p))
            # gen_adv_loss = tf.reduce_mean(tf.squared_difference(fake_as_true, x_hat_p))
            gen_adv_loss = -tf.reduce_mean(x_hat_p)
            gen_loss = 10*rec_cost + gen_adv_loss

            fake_as_fake = tf.zeros_like(x_hat_p)
            true_as_true = tf.ones_like(t_p)
            # disc_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.concat([fake_as_fake, true_as_true], axis=0), logits=tf.concat([x_hat_p, t_p], axis=0)))
            # disc_loss = tf.reduce_mean(tf.squared_difference(tf.concat([fake_as_fake, true_as_true], axis=0), tf.concat([x_hat_p, t_p], axis=0)))
            disc_loss = tf.reduce_mean(x_hat_p) - tf.reduce_mean(t_p)
            grad = tf.gradients(dis_interpolate, interpolate)
            slopes = tf.sqrt(tf.reduce_sum(tf.square(grad)[0], axis=[3]))
            grad_penalty = tf.reduce_mean((slopes - 1.) ** 2)
            disc_loss = disc_loss + 0.1*grad_penalty


            gen_variables = [v for v in tf.trainable_variables() if 'scale_' + str(scales - i - 1) + '/gen' in v.name]
            gen_opt = tf.train.AdamOptimizer(learning_rate= lr, beta1= 0.5).minimize(gen_loss, var_list=gen_variables)

            disc_variables = [v for v in tf.trainable_variables() if 'scale_' + str(scales - i - 1) + '/disc' in v.name]
            disc_opt = tf.train.AdamOptimizer(learning_rate=lr, beta1= 0.5).minimize(disc_loss, var_list= disc_variables)

        all_x_hat_fake.append(x_hat_fake)
        all_x_hat_rec.append(x_hat_rec)
        all_gen_loss.append(gen_loss)
        all_disc_loss.append(disc_loss)
        all_gen_opt.append(gen_opt)
        all_disc_opt.append(disc_opt)


    saver = tf.train.Saver()
    init = tf.global_variables_initializer()

    # make folder where all results will be saved
    results_dir = 'D:/Shivesh/SinGAN/Results/SinGan_train_' + str(run)
    # results_dir = '/storage/scratch1/0/schaudhary9/SinGAN/SinGan_train_' + str(run)
    if os.path.isdir(results_dir):
        shutil.rmtree(results_dir)

    os.mkdir(results_dir)

    with tf.Session() as sess:
        sess.run(init)
        summary_writer = tf.summary.FileWriter(results_dir, sess.graph)
        # file = open(results_dir + '/training_loss.txt', 'a')

        h = int(round(H * pow(0.75, scales - 1)))
        w = int(round(W * pow(0.75, scales - 1)))
        z_fixed = sess.run(tf.random_normal((1, h, w, C)))

        for i in range(scales):
            if i > 0:
                curr_weights = [v for v in tf.trainable_variables() if 'scale_' + str(scales - i - 1) in v.name]
                prev_weights = [v for v in tf.trainable_variables() if 'scale_' + str(scales - i) in v.name]
                for c_w, p_w in zip(curr_weights, prev_weights):
                    sess.run(c_w.assign(p_w))

            for n in range(training_iters):

                curr_x_fake, curr_x_rec, curr_y, curr_z_fake, curr_z_rec = gen_data(i, scales, img, sess, all_x_hat_fake, all_x_hat_rec, z_fixed)
                feed = gen_feed_dict(scales - i - 1, curr_x_fake, curr_x_rec, curr_y, curr_z_fake, curr_z_rec)

                for k in range(3):
                    do = sess.run(all_disc_opt[i], feed_dict=feed)
                for k in range(3):
                    go = sess.run(all_gen_opt[i], feed_dict=feed)

                g_cost = sess.run(all_gen_loss[i], feed_dict= feed)
                d_cost = sess.run(all_disc_loss[i], feed_dict=feed)

                print('scale_' + str(scales - i - 1) + ', iter - ' + str(n) + ' : Train gen loss == ' + "{:.6f}".format(g_cost) + ' Train disc loss == ' + "{:.6f}".format(d_cost))

            save_predictions(scales - i - 1, scales, img, sess, all_x_hat_fake, all_x_hat_rec, z_fixed, results_dir)


        summary_writer.close()





