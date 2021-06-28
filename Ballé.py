#pip install tensorflow-gpu==2.4.* tensorflow-compression==2.0
----------------------------------------------

import argparse
import glob
import sys
from absl import app
from absl.flags import argparse_flags
import pathlib
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_compression as tfc
from tensorflow import keras
import os
----------------------------------------------
def read_png(filename):
  """Loads a PNG image file."""
  string = tf.io.read_file(filename)
  return tf.image.decode_image(string, channels=3)


def write_png(filename, image):
  """Saves an image to a PNG file."""
  string = tf.image.encode_png(image)
  tf.io.write_file(filename, string)
----------------------------------------------
class AnalysisTransform(tf.keras.Sequential):
  """The analysis transform."""

  def __init__(self, num_filters):
    super().__init__(name="analysis")
    self.add(tf.keras.layers.Lambda(lambda x: x / 255.))
    self.add(tfc.SignalConv2D(
        num_filters, (5, 5), name="layer_0", corr=True, strides_down=4,
        padding="same_zeros", use_bias=True,
        activation=tfc.GDN(name="gdn_0")))
    self.add(tfc.SignalConv2D(
        num_filters, (5, 5), name="layer_1", corr=True, strides_down=2,
        padding="same_zeros", use_bias=True,
        activation=tfc.GDN(name="gdn_1")))
    self.add(tfc.SignalConv2D(
        num_filters, (5, 5), name="layer_2", corr=True, strides_down=2,
        padding="same_zeros", use_bias=True,
        activation=tfc.GDN(name="gdn_2")))
    self.add(tfc.SignalConv2D(
        num_filters, (5, 5), name="layer_3", corr=True, strides_down=2,
        padding="same_zeros", use_bias=True,
        activation=None))
----------------------------------------------
class SynthesisTransform(tf.keras.Sequential):
  """The synthesis transform."""

  def __init__(self, num_filters):
    super().__init__(name="synthesis")
    self.add(tfc.SignalConv2D(
        num_filters, (5, 5), name="layer_0", corr=False, strides_up=2,
        padding="same_zeros", use_bias=True,
        activation=tfc.GDN(name="igdn_0", inverse=True)))
    self.add(tfc.SignalConv2D(
        num_filters, (5, 5), name="layer_1", corr=False, strides_up=2,
        padding="same_zeros", use_bias=True,
        activation=tfc.GDN(name="igdn_1", inverse=True)))
    self.add(tfc.SignalConv2D(
        num_filters, (5, 5), name="layer_2", corr=False, strides_up=2,
        padding="same_zeros", use_bias=True,
        activation=tfc.GDN(name="igdn_2", inverse=True)))
    self.add(tfc.SignalConv2D(
        3, (5, 5), name="layer_3", corr=False, strides_up=4,
        padding="same_zeros", use_bias=True,
        activation=None))
    self.add(tf.keras.layers.Lambda(lambda x: x * 255.))
----------------------------------------------
class HyperAnalysisTransform(tf.keras.Sequential):
  """The analysis transform for the entropy model parameters."""

  def __init__(self, num_filters):
    super().__init__(name="hyper_analysis")
    self.add(tfc.SignalConv2D(
        num_filters, (3, 3), name="layer_0", corr=True, strides_down=1,
        padding="same_zeros", use_bias=True,
        activation=tf.nn.relu))
    self.add(tfc.SignalConv2D(
        num_filters, (5, 5), name="layer_1", corr=True, strides_down=2,
        padding="same_zeros", use_bias=True,
        activation=tf.nn.relu))
    self.add(tfc.SignalConv2D(
        num_filters, (5, 5), name="layer_2", corr=True, strides_down=2,
        padding="same_zeros", use_bias=False,
        activation=None))
----------------------------------------------
class HyperSynthesisTransform(tf.keras.Sequential):
  """The synthesis transform for the entropy model parameters."""

  def __init__(self, num_filters):
    super().__init__(name="hyper_synthesis")
    self.add(tfc.SignalConv2D(
        num_filters, (5, 5), name="layer_0", corr=False, strides_up=2,
        padding="same_zeros", use_bias=True, kernel_parameter="variable",
        activation=tf.nn.relu))
    self.add(tfc.SignalConv2D(
        num_filters, (5, 5), name="layer_1", corr=False, strides_up=2,
        padding="same_zeros", use_bias=True, kernel_parameter="variable",
        activation=tf.nn.relu))
    self.add(tfc.SignalConv2D(
        num_filters, (3, 3), name="layer_2", corr=False, strides_up=1,
        padding="same_zeros", use_bias=True, kernel_parameter="variable",
        activation=None))
----------------------------------------------
class BMSHJ2018Model(tf.keras.Model):
  """Main model class."""

  def __init__(self, lmbda, num_filters, num_scales, scale_min, scale_max):
    super().__init__()
    self.lmbda = lmbda
    self.num_scales = num_scales
    offset = tf.math.log(scale_min)
    factor = (tf.math.log(scale_max) - tf.math.log(scale_min)) / (
        num_scales - 1.)
    self.scale_fn = lambda i: tf.math.exp(offset + factor * i)
    self.analysis_transform = AnalysisTransform(num_filters)
    self.synthesis_transform = SynthesisTransform(num_filters)
    self.hyper_analysis_transform = HyperAnalysisTransform(num_filters)
    self.hyper_synthesis_transform = HyperSynthesisTransform(num_filters)
    self.hyperprior = tfc.NoisyDeepFactorized(batch_shape=(num_filters,))
    self.build((None, None, None, 3))
  
  @property

  def variables(self):
    return tf.Module.variables.fget(self)

  @property
  def trainable_variables(self):
    return tf.Module.trainable_variables.fget(self)

  weights = variables
  trainable_weights = trainable_variables

  # This seems to be necessary to prevent a comparison between class objects.
  _TF_MODULE_IGNORED_PROPERTIES = (
      tf.keras.Model._TF_MODULE_IGNORED_PROPERTIES.union(
          ("_compiled_trainable_state",)
      ))
  ############################################################################

  def call(self, x, training):
    """Computes rate and distortion losses."""
    entropy_model = tfc.LocationScaleIndexedEntropyModel(
        tfc.NoisyNormal, self.num_scales, self.scale_fn, coding_rank=3,
        compression=False)
    side_entropy_model = tfc.ContinuousBatchedEntropyModel(
        self.hyperprior, coding_rank=3, compression=False)

    y = self.analysis_transform(x)
    z = self.hyper_analysis_transform(abs(y))
    z_hat, side_bits = side_entropy_model(z, training=training)
    indexes = self.hyper_synthesis_transform(z_hat)
    y_hat, bits = entropy_model(y, indexes, training=training)
    x_hat = self.synthesis_transform(y_hat)

    # Total number of bits divided by total number of pixels.
    num_pixels = tf.cast(tf.reduce_prod(tf.shape(x)[:-1]), bits.dtype)
    bpp = (tf.reduce_sum(bits) + tf.reduce_sum(side_bits)) / num_pixels
    # Mean squared error across pixels.
    mse = tf.reduce_mean(tf.math.squared_difference(x, x_hat))
    ms_ssim = tf.squeeze(tf.image.ssim_multiscale(x, x_hat, 255))
    # The rate-distortion Lagrangian.
    loss = bpp + self.lmbda * mse
    return loss, bpp, mse, ms_ssim

  def train_step(self, x):
    with tf.GradientTape() as tape:
      loss, bpp, mse, ms_ssim = self(x, training=True)
    variables = self.trainable_variables
    gradients = tape.gradient(loss, variables)
    self.optimizer.apply_gradients(zip(gradients, variables))
    self.loss.update_state(loss)
    self.bpp.update_state(bpp)
    self.mse.update_state(mse)
    self.ms_ssim.update_state(ms_ssim)
    return {m.name: m.result() for m in [self.loss, self.bpp, self.mse, self.ms_ssim]}
  
  def test_step(self, x):
    loss, bpp, mse,ms_ssim = self(x, training=False)
    self.loss.update_state(loss)
    self.bpp.update_state(bpp)
    self.mse.update_state(mse)
    return {m.name: m.result() for m in [self.loss, self.bpp, self.mse, self.ms_ssim]}

  def predict_step(self, x):
    raise NotImplementedError("Prediction API is not supported.")

  def compile(self, **kwargs):
    super().compile(
        loss=None,
        metrics=None,
        loss_weights=None,
        weighted_metrics=None,
        **kwargs,
    )
    self.loss = tf.keras.metrics.Mean(name="loss")
    self.bpp = tf.keras.metrics.Mean(name="bpp")
    self.mse = tf.keras.metrics.Mean(name="mse")
    self.ms_ssim = tf.keras.metrics.Mean(name="ms_ssim")

  def fit(self, *args, **kwargs):
    retval = super().fit(*args, **kwargs)
    # After training, fix range coding tables.
    self.entropy_model = tfc.LocationScaleIndexedEntropyModel(
        tfc.NoisyNormal, self.num_scales, self.scale_fn, coding_rank=3,
        compression=True)
    self.side_entropy_model = tfc.ContinuousBatchedEntropyModel(
        self.hyperprior, coding_rank=3, compression=True)
    return retval

  @tf.function(input_signature=[
      tf.TensorSpec(shape=(None, None, 3), dtype=tf.uint8),
  ])
  def compress(self, x):
    """Compresses an image."""
    # Add batch dimension and cast to float.
    x = tf.expand_dims(x, 0)
    x = tf.cast(x, dtype=tf.float32)
    y = self.analysis_transform(x)
    z = self.hyper_analysis_transform(abs(y))
    # Preserve spatial shapes of image and latents.
    x_shape = tf.shape(x)[1:-1]
    y_shape = tf.shape(y)[1:-1]
    z_shape = tf.shape(z)[1:-1]
    z_hat, _ = self.side_entropy_model(z, training=False)
    indexes = self.hyper_synthesis_transform(z_hat)
    indexes = indexes[:, :y_shape[0], :y_shape[1], :]
    side_string = self.side_entropy_model.compress(z)
    string = self.entropy_model.compress(y, indexes)
    return string, side_string, x_shape, y_shape, z_shape

  @tf.function(input_signature=[
      tf.TensorSpec(shape=(1,), dtype=tf.string),
      tf.TensorSpec(shape=(1,), dtype=tf.string),
      tf.TensorSpec(shape=(2,), dtype=tf.int32),
      tf.TensorSpec(shape=(2,), dtype=tf.int32),
      tf.TensorSpec(shape=(2,), dtype=tf.int32),
  ])
  def decompress(self, string, side_string, x_shape, y_shape, z_shape):
    """Decompresses an image."""
    z_hat = self.side_entropy_model.decompress(side_string, z_shape)
    indexes = self.hyper_synthesis_transform(z_hat)
    indexes = indexes[:, :y_shape[0], :y_shape[1], :]
    y_hat = self.entropy_model.decompress(string, indexes)
    x_hat = self.synthesis_transform(y_hat)
    # Remove batch dimension, and crop away any extraneous padding.
    x_hat = x_hat[0, :x_shape[0], :x_shape[1], :]
    # Then cast back to 8-bit integer.
    return tf.saturate_cast(tf.round(x_hat), tf.uint8)
----------------------------------------------
def check_image_size(image, patchsize):
  shape = tf.shape(image)
  return shape[0] >= patchsize and shape[1] >= patchsize and shape[-1] == 3


def crop_image(image, patchsize):
  image = tf.image.random_crop(image, (patchsize, patchsize, 3))
  return tf.cast(image, tf.float32)


def get_dataset(name,split, patchsize, batchsize):
  """Creates input data pipeline from a TF Datasets dataset."""
  with tf.device("/cpu:0"):
    dataset = tfds.load(name,split=split, shuffle_files=True)
    #if split == "train":
      #dataset = dataset.repeat()
    dataset = dataset.filter(
        lambda x: check_image_size(x["image"], patchsize))
    dataset = dataset.map(
        lambda x: crop_image(x["image"], patchsize))
    dataset = dataset.batch(batchsize, drop_remainder=True)
  return dataset
----------------------------------------------
lmbda = 64
num_filters = 128
num_scales = 64.0
scale_min = 0.11
scale_max = 256.0


max_validation_steps = 16
epochs = 10
verbose = 1.0
batchsize = 8
patchsize = 256

model = BMSHJ2018Model(lmbda, num_filters, num_scales, scale_min,scale_max)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4)
)
----------------------------------------------
train_dataset = get_dataset("flic","train", patchsize ,batchsize)
----------------------------------------------
model.fit(
    train_dataset,
    epochs=epochs,
    callbacks=[
              tf.keras.callbacks.TerminateOnNaN(),
              #tf.keras.callbacks.TensorBoard(
                  #log_dir='/content/',
                  #histogram_freq=1, update_freq="epoch"),
              tf.keras.callbacks.experimental.BackupAndRestore('/content/'),
          ]
)
----------------------------------------------
model.summary()
----------------------------------------------
from google.colab import files
----------------------------------------------
uploaded = files.upload()
----------------------------------------------
#name of used image to test with
x = read_png('kodim03.png')
----------------------------------------------
tensors= model.compress(x)
packed = tfc.PackedTensors()
packed.pack(tensors)
----------------------------------------------
x_hat = model.decompress(*tensors)

# Cast to float in order to compute metrics.
x = tf.cast(x, tf.float32)
x_hat = tf.cast(x_hat, tf.float32)
mse = tf.reduce_mean(tf.math.squared_difference(x, x_hat))
psnr = tf.squeeze(tf.image.psnr(x, x_hat, 255))
msssim = tf.squeeze(tf.image.ssim_multiscale(x, x_hat, 255))
msssim_db = -10. * tf.math.log(1 - msssim) / tf.math.log(10.)

# The actual bits per pixel including entropy coding overhead.
num_pixels = tf.reduce_prod(tf.shape(x)[:-1])
bpp = len(packed.string) * 8 / num_pixels


print(f"Mean squared error: {mse:0.4f}")
print(f"PSNR (dB): {psnr:0.2f}")
print(f"Multiscale SSIM: {msssim:0.4f}")
print(f"Multiscale SSIM (dB): {msssim_db:0.2f}")
print(f"Bits per pixel: {bpp:0.4f}")
----------------------------------------------
#plot the original image
import matplotlib.pyplot as plt
plt.imshow(tf.dtypes.cast(x_hat, tf.int32))
----------------------------------------------
#plot the reconstruction
plt.imshow(tf.dtypes.cast(x, tf.int32))