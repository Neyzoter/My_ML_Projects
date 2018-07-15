
import tensorflow as tf

def getOnesMatrix(shape):
    """
    Creates an array of ones of dimension shape
    
    Arguments:
    shape -- shape of the array you want to create
        
    Returns: 
    ones -- array containing only ones
    """
    
    
    # Create "ones" tensor using tf.ones(...). 
    getOnes = tf.ones(shape)
    # Create the session 
    sess = tf.Session()
    # Run the session to compute 'ones'
    ones = sess.run(getOnes)
    # Close the session 
    sess.close()
    return ones