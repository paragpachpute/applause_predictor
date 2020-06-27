#%tensorflow_version 1.x
import tensorflow as tf
import os

sess = tf.Session()
imported_meta = tf.train.import_meta_graph('model.ckpt-12207.meta') 
imported_meta.restore(sess, 'model.ckpt-12207') 
my_vars = []
for var in tf.all_variables():
    if 'adam_v' not in var.name and 'adam_m' not in var.name:
        my_vars.append(var)
saver = tf.train.Saver(my_vars)
saver.save(sess, os.path.join('/content/','model_01.ckpt'))