import tensorflow as tf
import numpy as np

def getOneHotMatrix_tf(labels, C):
    """
    Creates a matrix where the i-th row corresponds to the ith class number and the jth column
                     corresponds to the jth training example. So if example j had a label i. Then entry (i,j) 
                     will be 1. 
                     
    Arguments:
    labels -- vector containing the labels 
    C -- number of classes, the depth of the one hot dimension
    
    Returns: 
    one_hot -- one hot matrix，一列一个样本
    """
      
    # Create a tf.constant equal to C (depth), name it 'C'. 
    C_num = tf.constant(C,name="C")
    # Use tf.one_hot, be careful with the axis 
    # lables：输入标签，是以多数字形式，而不是0和1
    # depth ：深度，即一共多少种类
    # axis：横着（1）还是竖着（0）表示一个样本的类型
    get_OneHot = tf.one_hot(labels,depth = C_num,axis = 0)
    # Create the session
    sess = tf.Session()
    # Run the session 
    one_hot = sess.run(get_OneHot)
    # Close the session . 
    sess.close()
    
    return one_hot
	
	
def getOneHotMatrix_np(labels, C):	
	"""
	Creates a matrix where the i-th row corresponds to the ith class number and the jth column
				 corresponds to the jth training example. So if example j had a label i. Then entry (i,j) 
				 will be 1. 
				 
	Arguments:
	labels -- vector containing the labels 
	C -- number of classes, the depth of the one hot dimension

	Returns: 
	one_hot -- one hot matrix，一列一个样本
	"""
	#用于设置如果没有[]，eye生成一个单位矩阵。如果有[]则会根据矩阵的数字，分配1的位置，其他都是0.
	Y = np.eye(C)[labels.reshape(-1)].T  
	
	return Y
	
	