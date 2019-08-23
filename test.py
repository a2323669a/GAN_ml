import tensorflow as tf
tf.enable_eager_execution()

x = tf.Variable(0,trainable=False,name='k')
root = tf.train.Checkpoint(y=x)
root.save("./d.ckpt")
x = x * 12
root.restore(tf.train.latest_checkpoint("."))
x
