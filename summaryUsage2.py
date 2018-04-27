import tensorflow as tf

x = tf.Variable(1.0)
xnew = tf.assign(x,x+0.2)
x_summary = tf.summary.scalar('x_trace',xnew)

y = tf.Variable(tf.ones([28,28]),name='2d_value')
ynew = tf.assign(y,tf.add(y,tf.random_normal([28,28])))
y_summary = tf.summary.histogram('2d_value',ynew)

merged_summary = tf.summary.merge_all()

writer= tf.summary.FileWriter('basic_tf2',graph=tf.get_default_graph())
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(20):
        _,merged_smy = sess.run([xnew,merged_summary])
        writer.add_summary(merged_smy,global_step=i)
writer.close()