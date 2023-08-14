import tensorflow as tf




mean = 0.0
std = 0.2
x = 1.0

gaussian_cdf = 0.5*( 1. + tf.math.erf((x-mean)/
                                (tf.cast(tf.math.sqrt(2.), tf.float32)*std))
                                )
print(gaussian_cdf)