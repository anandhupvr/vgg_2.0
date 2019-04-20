import sys
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import optimizers
from data.loader import Data
from arch.vgg import VGG16

data_dir = sys.argv[1]

data = Data(data_dir)
im = data.get()
model = VGG16([224, 224, 3])

criteon = keras.losses.CategoricalCrossentropy(from_logits=True)
metric = keras.metrics.CategoricalAccuracy()

optimizer = optimizers.Adam(learning_rate=0.0001)


for i in range(100):
	for _ in range(int(data.numbers()/4)):
		x, y = next(im)

		with tf.GradientTape() as tape:
			logits = model(x)
			loss = criteon(y, logits)
			metric.update_state(y, logits)

		grads = tape.gradient(loss, model.trainable_variables)

		grads = [ tf.clip_by_norm(g, 15) for g in grads]

		optimizer.apply_gradients(zip(grads, model.trainable_variables))

		if _ % 40 == 0:
			print (i, _, 'loss : ', float(loss), 'acc :', metric.result().numpy())
			metric.reset_states()
	if i % 20 == 0
		model.save('weights/mode_%s.h5'$(i))