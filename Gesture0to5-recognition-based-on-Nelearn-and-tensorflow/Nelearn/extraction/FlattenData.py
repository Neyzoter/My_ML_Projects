
import numpy as np

def getFlatData(X):
	'''
	输入：(x,...)的矩阵，比如图片，x张图片、像素m*n、y个通道
	输出：(特征个数,x)的扁平化矩阵，即每一列是一个样本的工程
	'''
	
	# 让X转化为样本数*所有特征向量化.T
	X_flatten = X.reshape(X.shape[0],-1).T
	return X_flatten
	