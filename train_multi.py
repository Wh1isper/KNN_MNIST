from rnn import classify
from utils import load
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
import datetime

train_images, train_labels, test_images, test_labels = load()
# 调整训练集大小
# train_images, train_labels = train_images[:1000],train_labels[:1000]
# 取前100快一点！
# test_images, test_labels = test_images[:100], test_labels[:100]
def multiPack(mix_args):
    return classify(mix_args[0],mix_args[1],mix_args[2],mix_args[3]) == mix_args[4]


if __name__ == '__main__':
    k = 10
    print('K:{}'.format(k))
    print("读取完成")
    test_size = test_images.shape[0]
    print("正在分类：{}".format(test_size))
    # timeA = datetime.datetime.now()
    # outY = [classify(test_images[i], train_images, train_labels, k) for i in tqdm(range(test_size))]
    # print(datetime.datetime.now()-timeA)
    # print("KNN输出:{}".format(outY))
    # print("实际标签：{}".format(test_labels))
    # acc = np.mean(outY == test_labels)
    # print('\n\n')
    # print("正确率：{}".format(acc))




    # para1 = [test_images[i] for i in range(test_size)]
    # para2 = [train_images]
    # para3 = [train_labels]
    # para4 = [k]
    # multiList = zip(para1,para2,para3,para4)
    timeA = datetime.datetime.now()
    multiList = []
    for i in range(test_size):
        multiList.append([test_images[i],train_images,train_labels,k,test_labels[i]])
    pool = Pool(8)
    outY = pool.map(multiPack,multiList)
    pool.close()
    pool.join()
    print(datetime.datetime.now() - timeA)
    count = 0
    for flag in outY:
        if flag:
            count += 1
    acc = count/test_size

    print("正确率：{}".format(acc))
    # 正确率：0.9683
