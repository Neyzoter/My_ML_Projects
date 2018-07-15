import tensorflow as tf


def cost_using_SigmoidCrossEntropyWithLogits(logits, labels):
    """
    Computes the cost using the sigmoid cross entropy
    
    Arguments:
    logits -- vector containing z, output of the last linear unit (before the final sigmoid activation)
    labels -- vector of labels y (1 or 0) 
    
    Note: What we've been calling "z" and "y" in this class are respectively called "logits" and "labels" 
    in the TensorFlow documentation. So logits will feed into z, and labels into y. 
    
    Returns:
    cost -- runs the session of the cost (formula (2))
    """
    
    # Create the placeholders for "logits" (z) and "labels" (y) 
    lgt = tf.placeholder(tf.float32,name="lgt")
    lbl = tf.placeholder(tf.float32,name="lbl")
    
    # Use the loss function 
    # sigmoid型交叉熵和逻辑
    loss_func = tf.nn.sigmoid_cross_entropy_with_logits(logits=lgt,labels=lbl)
    # Create a session. See method 1 above.
    sess = tf.Session()
    # Run the session
    cost = sess.run(loss_func,feed_dict={lgt:logits,lbl:labels})
    # Close the session. See method 1 above.
    sess.close()
    
    return cost