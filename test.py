import tensorflow as tf
import numpy as np
import transform

with tf.Graph().as_default(), tf.Session() as sess:
        input_batch = tf.placeholder(tf.float32, shape=(3, 160, 300, 3), name="input_batch")

        stylized_image = transform.net(input_batch/255.0)

        loss = tf.nn.l2_loss(input_batch - stylized_image)

        train_step = tf.train.AdamOptimizer(1e-3).minimize(loss)
        sess.run(tf.initialize_all_variables())
        iterations = 0
        for epoch in range(2):
            for i in range(0, 10, 3):

                batch = np.zeros((3, 160, 300, 3), dtype=np.float32)

                # for j, img_path in enumerate(content_training_images[i: i+self.batch_size]):
                #     batch[j] = load_image(img_path, img_size=self.batch_shape[1:])

                print("images loaded")
                print(batch.shape)
                train_step.run(feed_dict={input_batch:batch})
