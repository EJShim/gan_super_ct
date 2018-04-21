import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from skimage.io import imsave
import os
import shutil

img_height = 512
img_width = 512
img_size = (img_height, img_width)

to_train = True
to_restore = False
output_path = "output"

max_epoch = 500

h1_size = 150
h2_size = 300
z_size = 1
batch_size = 16

# 일반적인 GAN 의 형태 입니다.
# 라벨을 구분하지 않습니다.

# 제너레이터 (G)
def build_generator(z_prior):
    
    deconv1 = tf.layers.conv2d_transpose(inputs=z_prior, filters=8, strides=1, kernel_size=8)
    deconv1 = tf.nn.relu(deconv1)

    deconv2 = tf.layers.conv2d_transpose(inputs=deconv1, filters=16,  padding='same', strides=2, kernel_size=3)
    deconv2 = tf.nn.relu(deconv2)

    deconv3 = tf.layers.conv2d_transpose(inputs=deconv2, filters=32,  padding='same', strides=2, kernel_size = 3)
    deconv3 = tf.nn.relu(deconv3)
    
    deconv4 = tf.layers.conv2d_transpose(inputs=deconv3, filters=64, padding='same', strides=2, kernel_size=3)
    deconv4 = tf.nn.relu(deconv4)

    deconv5 = tf.layers.conv2d_transpose(inputs=deconv4, filters=128, padding='same', strides=2, kernel_size=3)
    deconv5 = tf.nn.relu(deconv5)

    deconv6 = tf.layers.conv2d_transpose(inputs=deconv5, filters=256,  padding='same', strides=2, kernel_size=3)
    deconv6 = tf.nn.relu(deconv6)

    deconv7 = tf.layers.conv2d_transpose(inputs=deconv6, filters=1,  padding='same', strides=2, kernel_size=3)
    
    x_generate = deconv7
    

    # x_generate
    

    return x_generate

# 디스크리미네이터 (D)
def build_discriminator(x_data, x_generated, keep_prob):
    # 실제 이미지와 생성된 이미지를 합침
    x_in = tf.concat([x_data, x_generated], 0) 


    conv1 = tf.layers.conv2d(inputs=x_in, filters=32, strides=2, kernel_size=7)
    conv1 = tf.nn.dropout(tf.nn.relu(conv1), keep_prob)

    conv2 = tf.layers.conv2d(inputs=conv1, filters=64, kernel_size=5)
    conv2 = tf.nn.dropout(tf.nn.relu(conv2), keep_prob)

    conv3 = tf.layers.conv2d(inputs=conv2, filters=128, kernel_size=5)
    conv3 = tf.nn.dropout(tf.nn.relu(conv3), keep_prob)

    pool1 = tf.layers.max_pooling2d(inputs=conv3, pool_size=3, strides=2)
    
    conv4 = tf.layers.conv2d(inputs=pool1, filters=256, kernel_size=3)
    conv4 = tf.nn.dropout(tf.nn.relu(conv4), keep_prob)
    
    conv5 = tf.layers.conv2d(inputs=conv4, filters=512, kernel_size=3)
    conv5 = tf.nn.dropout(tf.nn.relu(conv5), keep_prob)

    pool2 = tf.layers.max_pooling2d(inputs=conv5, pool_size=3, strides=2)
    
    conv6 = tf.layers.conv2d(inputs=pool2, filters=1024, strides=2, kernel_size=3)
    conv6 = tf.nn.dropout(tf.nn.relu(conv6), keep_prob)

    gap = tf.layers.average_pooling2d(inputs=conv6, pool_size=28, strides=1)

    fc = tf.layers.conv2d(inputs=gap, filters=1, kernel_size=1)


    #Real = y_data, #Fake = y_generated
    y_data, y_generated = tf.split(fc, num_or_size_splits=2, axis=0)    

    return y_data, y_generated



def train():

    x_data = tf.placeholder(tf.float32, [batch_size, img_height, img_width, 1], name="x_data") # (batch_size, img_size)
    z_prior = tf.placeholder(tf.float32, [batch_size, 1, 1, 1], name="z_prior") # (batch_size, z_size)
    keep_prob = tf.placeholder(tf.float32, name="keep_prob") # dropout 퍼센트
    global_step = tf.Variable(0, name="global_step", trainable=False)



    # x_generated : generator 가 생성한 이미지, g_params : generater 의 TF 변수들
    x_generated = build_generator(z_prior)

    # 실제이미지, generater 가 생성한 이미지, dropout keep_prob 를 넣고 discriminator(경찰) 이 감별
    y_data, y_generated = build_discriminator(x_data, x_generated, keep_prob)

    # loss 함수 ( D 와 G 를 따로 ) *
    d_loss = - (tf.log(y_data) + tf.log(1 - y_generated))
    g_loss = - tf.log(y_generated)

    # optimizer : AdamOptimizer 사용 *
    optimizer = tf.train.AdamOptimizer(0.0001)

    # discriminator 와 generator 의 변수로 각각의 loss 함수를 최소화시키도록 학습
    d_trainer = optimizer.minimize(d_loss)
    g_trainer = optimizer.minimize(g_loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    print("initialization complete")

    exit()

    z_sample_val = np.random.normal(0, 1, size=(batch_size, z_size)).astype(np.float32)

    for i in range(sess.run(global_step), max_epoch):
        for j in range(21870 // batch_size):
            print("epoch:%s, iter:%s" % (i, j))

            # x_value, _ = mnist.train.next_batch(batch_size)
            # x_value = 2 * x_value.astype(np.float32) - 1
            # print(x_value[0])

            batch_end = j * batch_size + batch_size
            if batch_end >= size:
                batch_end = size - 1
            x_value = phd08[ j * batch_size : batch_end ]
            x_value = x_value / 255.
            x_value = 2 * x_value - 1

            z_value = np.random.normal(0, 1, size=(batch_size, z_size)).astype(np.float32)
            sess.run(d_trainer,
                     feed_dict={x_data: x_value, z_prior: z_value, keep_prob: np.sum(0.7).astype(np.float32)})
            if j % 1 == 0:
                sess.run(g_trainer,
                         feed_dict={x_data: x_value, z_prior: z_value, keep_prob: np.sum(0.7).astype(np.float32)})
        x_gen_val = sess.run(x_generated, feed_dict={z_prior: z_sample_val})
        show_result(x_gen_val, os.path.join(output_path, "sample%s.jpg" % i))
        print(x_gen_val)
        z_random_sample_val = np.random.normal(0, 1, size=(batch_size, z_size)).astype(np.float32)
        x_gen_val = sess.run(x_generated, feed_dict={z_prior: z_random_sample_val})
        show_result(x_gen_val, os.path.join(output_path, "random_sample%s.jpg" % i))
        sess.run(tf.assign(global_step, i + 1))
        saver.save(sess, os.path.join(output_path, "model"), global_step=global_step)



if __name__ == '__main__':

    train()
