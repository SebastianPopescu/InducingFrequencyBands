import tensorflow as tf

def update_off_diagonal(tensor, list_of_values):
    shape = tf.shape(tensor)
    rows, cols = shape[0], shape[1]
    
    #get indces of elements
    indices = tf.where(tf.ones_like(tensor, dtype=tf.bool))
    print(indices)

    desired_band = tf.cast(tf.reduce_sum(indices, axis = -1), tf.float32) * 0.5
    print(desired_band)
    desired_band = tf.cast(desired_band, tf.int32)
    print(desired_band)

    updates = [list_of_values[_] for _ in desired_band]
    print(updates)

    # Update elements with even and odd values    
    updated_tensor = tf.tensor_scatter_nd_update(tensor, indices, updates)

    return updated_tensor

# Test the function
input_tensor = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=tf.float32)
lista = [0.1, 0.2, 0.3]
updated_tensor = update_off_diagonal(input_tensor, lista)

print("Original Tensor:")
print(input_tensor)
print("Updated Tensor:")
print(updated_tensor)