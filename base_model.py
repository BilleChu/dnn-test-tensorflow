#!/usr/bin python
import tensorflow as tf
import numpy as np

class Model(object):
    def __init__(self, v_slots, v_slot_length, v_slot_embedding):
        self.embedding_size = 128
        self.slot_num = len(v_slots)
        self.v_slots = v_slots
        self.v_slot_length = v_slot_length
        self.v_slot_embedding = v_slot_embedding
        self.v_placeholders = []
        self.v_embeds = []

    def sub_embedding(self, slot_id, slot_length, embedding_size):
        single_emb = tf.get_variable(name="embed" + str(slot_id),
                              shape=[slot_length, embedding_size],
                              initializer=tf.initializers.random_normal())
        return single_emb

    def build(self):
        for i in range(len(self.v_slots)):
            self.v_placeholders.append(tf.placeholder(dtype=tf.int32, shape=[None,]))
            self.v_embeds.append(self.sub_embedding(i, self.v_slot_length[i], self.v_slot_embedding[i]))
        emb = tf.concat(values=[tf.nn.embedding_lookup(self.v_embeds[i], self.v_placeholders[i])
                                for i in range(self.slot_num)], axis=1)

        emb_bn = tf.layers.batch_normalization(emb, name="bn")
        fc_1 = tf.layers.dense(emb_bn, 128, activation=tf.nn.relu, name="fc_1")
        fc_2 = tf.layers.dense(fc_1, 32, activation=tf.nn.relu, name="fc_2")
#        output = tf.layers.dense(fc_2, 1, activation=tf.nn.sigmoid, name="output")
        output = tf.layers.dense(fc_2, 1, name="output")
        self.v_placeholders.append(tf.placeholder(dtype=tf.float32, shape=[None,]))
        self.y = self.v_placeholders[-1]
        self.logits = output[0]
        self.pred = tf.sigmoid(self.logits)

    def train(self,sess, data):
        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                                    logits=self.logits, labels=self.y))
        self.opt = tf.train.AdamOptimizer(learning_rate=1.)
        train_op = self.opt.minimize(self.loss)
        sess.run(tf.global_variables_initializer())
        for i in range(100):
            sess.run(train_op, feed_dict={i: d for i, d in zip(self.v_placeholders, data)})
            loss, pred, y = sess.run([self.loss, self.pred, self.y], feed_dict={i: d for i, d in zip(self.v_placeholders, data)})
            print (loss,  pred, y)

    def evaluate(self):
        pass

    def predict(self, sess, data):
        for i in range(100):
            pred = sess.run(self.pred, feed_dict={i: d for i, d in zip(self.v_placeholders, data)})
            print (pred)

    def save(self, sess, path):
        saver = tf.train.Saver()
        saver.save(sess, save_path=path)

    def restore(self, sess, path):
        saver = tf.train.Saver()
        saver.restore(sess, save_path=path)

if __name__ == '__main__':
    print ("start")
    v_slots = np.arange(3)
    v_slot_size = [100 ,100 ,100]
    v_slot_embed = [5, 5, 5]
    data = [[1], [1], [1], [1]] # last one is label
    test_data = [[1], [1], [1]]
    with tf.Session() as sess:
        model = Model(v_slots, v_slot_size, v_slot_embed)
        model.build()
        model.train(sess, data)
        model.save(sess,"./model.ckpt")
        model.predict(sess, test_data)
        model.restore(sess, "./model.ckpt")
