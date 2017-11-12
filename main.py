import os.path
import time
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
	warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
	print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
	"""
	Load Pretrained VGG Model into TensorFlow.
	:param sess: TensorFlow Session
	:param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
	:return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
	"""
	# TODO: Implement function
	#   Use tf.saved_model.loader.load to load the model and weights
	vgg_tag = 'vgg16'
	vgg_input_tensor_name = 'image_input:0'
	vgg_keep_prob_tensor_name = 'keep_prob:0'
	vgg_layer3_out_tensor_name = 'layer3_out:0'
	vgg_layer4_out_tensor_name = 'layer4_out:0'
	vgg_layer7_out_tensor_name = 'layer7_out:0'

	tf.saved_model.loader.load(sess, [vgg_tag], vgg_path);

	graph = tf.get_default_graph();

	image_input = graph.get_tensor_by_name(vgg_input_tensor_name);
	keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name);
	layer3_out = graph.get_tensor_by_name(vgg_layer3_out_tensor_name);
	layer4_out = graph.get_tensor_by_name(vgg_layer4_out_tensor_name);
	layer7_out = graph.get_tensor_by_name(vgg_layer7_out_tensor_name);

	return image_input, keep_prob, layer3_out, layer4_out, layer7_out
tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
	"""
	Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
	:param vgg_layer7_out: TF Tensor for VGG Layer 3 output
	:param vgg_layer4_out: TF Tensor for VGG Layer 4 output
	:param vgg_layer3_out: TF Tensor for VGG Layer 7 output
	:param num_classes: Number of classes to classify
	:return: The Tensor for the last layer of output
	"""

	# We append a 1×1 convolution with channel  dimension 21 to predict
	# scores for each of the PASCAL classes (including background) at each
	# of the coarse output locations, followed by a deconvolution layer to bi-
	# linearly upsample the coarse outputs to pixel-dense outputs as described
	# in Section 3.3.

	KERNEL_REGULARIZATION = 1e-3;

	output = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, strides=(1,1), padding='same',
	                          kernel_regularizer=tf.contrib.layers.l2_regularizer(KERNEL_REGULARIZATION))
	output = tf.layers.batch_normalization (output);
	output = tf.layers.conv2d_transpose(output, num_classes, 4, strides=(2, 2), padding='same',
	                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(KERNEL_REGULARIZATION))

	l4_1x1 = tf.layers.conv2d(vgg_layer4_out, num_classes, 1, strides=(1,1), padding='same',
	                          kernel_regularizer=tf.contrib.layers.l2_regularizer(KERNEL_REGULARIZATION))
	l4_1x1 = tf.layers.batch_normalization (l4_1x1);
	output = tf.add(output, l4_1x1)
	output = tf.layers.conv2d_transpose(output, num_classes, 4, strides=(2, 2), padding='same',
	                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(KERNEL_REGULARIZATION))

	l3_1x1 = tf.layers.conv2d(vgg_layer3_out, num_classes, 1, strides=(1,1), padding='same',
	                          kernel_regularizer=tf.contrib.layers.l2_regularizer(KERNEL_REGULARIZATION))
	l3_1x1 = tf.layers.batch_normalization (l3_1x1);
	output = tf.add(output, l3_1x1)
	output = tf.layers.conv2d_transpose(output, num_classes, 16, strides=(8, 8), padding='same',
	                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(KERNEL_REGULARIZATION))

	return output;

tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
	"""
	Build the TensorFLow loss and optimizer operations.
	:param nn_last_layer: TF Tensor of the last layer in the neural network
	:param correct_label: TF Placeholder for the correct label image
	:param learning_rate: TF Placeholder for the learning rate
	:param num_classes: Number of classes to classify
	:return: Tuple of (logits, train_op, cross_entropy_loss)
	"""

	# Reshape 4D tensor to 2D
	logits = tf.reshape(nn_last_layer, (-1, num_classes))
	labels = tf.reshape(correct_label, (-1, num_classes))
	#labels = tf.reshape(correct_label, [-1])
	#logits = tf.Print(logits, ["Logits: ", logits, " Labels: ", labels]);

	softmax = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels);
	#softmax = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels);

	cross_entropy_loss = tf.reduce_mean(softmax)

	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
	train_op = optimizer.minimize(cross_entropy_loss)

	return logits, train_op, cross_entropy_loss
tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate, output_dir=None, saver=None):
	"""
	Train neural network and print out the loss during training.
	:param sess: TF Session
	:param epochs: Number of epochs
	:param batch_size: Batch size
	:param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
	:param train_op: TF Operation to train the neural network
	:param cross_entropy_loss: TF Tensor for the amount of loss
	:param input_image: TF Placeholder for input images
	:param correct_label: TF Placeholder for label images
	:param keep_prob: TF Placeholder for dropout keep probability
	:param learning_rate: TF Placeholder for learning rate
	:param output_dir: Directory to put all output
	:param saver: Graph saver
	"""

	#iou, iou_op = tf.metrics.mean_iou(correct_label, cross_entropy_loss, 2)
	#print (train_op);

	logfile = None;

	if (output_dir != None):
		logfile = open (os.path.join(output_dir, "session.csv"), 'w');
		logfile.write ("Epoch,Batch,Loss\n");

	for epoch in range (epochs):
		loss = 0;
		batch = 0;
		for image, label in get_batches_fn(batch_size, epoch > 0):
			# train_op and cross_entropy_loss
			_, loss = sess.run([train_op, cross_entropy_loss], feed_dict={input_image: image, correct_label: label, keep_prob: 1.0, learning_rate: 0.0001});

			if (logfile != None):
				logfile.write ("{},{},{}\n".format(epoch, batch, loss));
			batch += 1;

			#TODO: FIXME
			#sess.run(iou_op, feed_dict={input_image: image, correct_label: label})
			#print("Mean IoU =", sess.run(iou))
		if (saver != None):
			saver.save(sess, os.path.join(output_dir, "e{}".format (epoch)))

		print ("Epoch {} Loss {}".format(epoch, loss));

	if (logfile != None):
		logfile.close ();
tests.test_train_nn(train_nn)


def run():
	num_classes = 2
	image_shape = (160, 576)
	data_dir = './data'
	runs_dir = './runs'
	tests.test_for_kitti_dataset(data_dir)

	# Make folder for current run
	output_dir = os.path.join(runs_dir, str(time.time()))
	if os.path.exists(output_dir):
		shutil.rmtree(output_dir)
	os.makedirs(output_dir)

	tf.reset_default_graph();

	# Download pretrained vgg model
	helper.maybe_download_pretrained_vgg(data_dir)

	# OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
	# You'll need a GPU with at least 10 teraFLOPS to train on.
	#  https://www.cityscapes-dataset.com/

	correct_label = tf.placeholder(tf.float32, [None, None, None, num_classes], name="correct_label")
	#correct_label = tf.placeholder(tf.int32, [None, None, None, num_classes-1], name="correct_label")

	learning_rate = tf.placeholder(tf.float32, name="learning_rate")
	EPOCHS = 10;
	BATCH_SIZE = 8;

	with tf.Session() as sess:
		# Path to vgg model
		vgg_path = os.path.join(data_dir, 'vgg')
		# Create function to get batches
		get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

		# OPTIONAL: Augment Images for better results
		#  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

		# TODO: Build NN using load_vgg, layers, and optimize function
		image_input, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(sess, vgg_path)
		nn_last_layer = layers(layer3_out, layer4_out, layer7_out, num_classes);

		logits, train_op, cross_entropy_loss = optimize(nn_last_layer, correct_label, learning_rate, num_classes)

		sess.run(tf.global_variables_initializer());
		sess.run(tf.local_variables_initializer());

		saver = tf.train.Saver()

		# TODO: Train NN using the train_nn function
		train_nn(sess, EPOCHS, BATCH_SIZE, get_batches_fn, train_op, cross_entropy_loss, image_input, correct_label, keep_prob, learning_rate, output_dir, saver);

		# TODO: Save inference data using helper.save_inference_samples
		helper.save_inference_samples(output_dir, data_dir, sess, image_shape, logits, keep_prob, image_input)

		# OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
	run()
