import numpy as np
import struct

filename_train_images = './MNIST/train-images.idx3-ubyte'
filename_train_labels = './MNIST/train-labels.idx1-ubyte'
filename_test_images = './MNIST/t10k-images.idx3-ubyte'
filename_test_labels = './MNIST/t10k-labels.idx1-ubyte'

def load_images(file_name):
    ##   在读取或写入一个文件之前，你必须使用 Python 内置open()函数来打开它。##
    ##   file object = open(file_name [, access_mode][, buffering])          ##
    ##   file_name是包含您要访问的文件名的字符串值。                         ##
    ##   access_mode指定该文件已被打开，即读，写，追加等方式。               ##
    ##   0表示不使用缓冲，1表示在访问一个文件时进行缓冲。                    ##
    ##   这里rb表示只能以二进制读取的方式打开一个文件                        ##
    binfile = open(file_name, 'rb')
    ##   从一个打开的文件读取数据
    buffers = binfile.read()
    ##   读取image文件前4个整型数字
    magic, num, rows, cols = struct.unpack_from('>IIII', buffers, 0)
    ##   整个images数据大小为60000*28*28
    bits = num * rows * cols
    ##   读取images数据
    images = struct.unpack_from('>' + str(bits) + 'B', buffers, struct.calcsize('>IIII'))
    ##   关闭文件
    binfile.close()
    ##   转换为[60000,784]型数组
    images = np.reshape(images, [num, rows * cols])
    return images


def load_labels(file_name):
    ##   打开文件
    binfile = open(file_name, 'rb')
    ##   从一个打开的文件读取数据
    buffers = binfile.read()
    ##   读取label文件前2个整形数字，label的长度为num
    magic, num = struct.unpack_from('>II', buffers, 0)
    ##   读取labels数据
    labels = struct.unpack_from('>' + str(num) + "B", buffers, struct.calcsize('>II'))
    ##   关闭文件
    binfile.close()
    ##   转换为一维数组
    labels = np.reshape(labels, [num])
    return labels


def load():
    train_images = load_images(filename_train_images)
    train_labels = load_labels(filename_train_labels)
    test_images = load_images(filename_test_images)
    test_labels = load_labels(filename_test_labels)
    return train_images,train_labels,test_images,test_labels

def binary(images):
    for image in images:
        count = 0
        for pix in image:
            if pix>127:
                image[count] = 1
            else:
                image[count] = 0
            count += 1
    return images

def remove(removeList):
    train_images, train_labels, test_images, test_labels = load()
    new_train_images = []
    new_train_labels = []
    new_test_images = []
    new_test_labels = []
    for i in range(len(train_labels)):
        if train_labels[i] in removeList:
            pass
        else:
            new_train_images.append(train_images[i])
            new_train_labels.append(train_labels[i])

    for i in range(len(test_labels)):
        if test_labels[i] in removeList:
            pass
        else:
            new_test_images.append(test_images[i])
            new_test_labels.append(test_labels[i])
    new_train_images = np.array(new_train_images)
    new_train_labels = np.array(new_train_labels)
    new_test_images = np.array(new_test_images)
    new_test_labels = np.array(new_test_labels)

    return new_train_images, new_train_labels, new_test_images, new_test_labels