import tensorflow as tf
x = tf.Variable(1.0)
xnew = tf.assign(x,x+0.2)
x_summary = tf.summary.scalar('x_trace',xnew)

writer= tf.summary.FileWriter('basic_tf',graph=tf.get_default_graph())
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(20):
        _,x_smy = sess.run([xnew,x_summary])
        writer.add_summary(x_smy,global_step=i)
writer.close()