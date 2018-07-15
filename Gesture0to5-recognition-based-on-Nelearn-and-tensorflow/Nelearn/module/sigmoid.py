import tensorflow as tf

__author__ = 'Neyzoter Song'

def sigmoid(z):
    """
    Computes the sigmoid of z
    
    Arguments:
    z -- input value, scalar or vector
    
    Returns: 
    results -- the sigmoid of z
    """
    
    # Create a placeholder for x. 
    x = tf.placeholder(tf.float32,name="x")
    
    # compute sigmoid(x)
    get_sigmoid = tf.sigmoid(x)
    # Create a session, and run it. Please use the method 2 explained above. 
    # Run session and call the output "result"
    with tf.Session() as sess:
        result = sess.run(get_sigmoid,feed_dict ={x:z})
    
    return result
	
	