from knn import classify
from utils import load, binary, remove
from multiprocessing import Pool
import datetime
import json


def multiPack(mix_args):
    return classify(mix_args[0], mix_args[1], mix_args[2], mix_args[3]) == mix_args[4]

# 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000,
datasetList = [60000]
kList = [1, 2, 3, 4, 5]
# [1], [1, 2],[1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6, 7]
removeLists = [[]]

if __name__ == '__main__':
    for removeList in removeLists:
        for y in range(len(kList)):
            for x in range(len(datasetList)):
                # 调整训练集大小
                k = kList[y]
                train_images, train_labels, test_images, test_labels = remove(removeList)
                train_images, train_labels = train_images[:datasetList[x]], train_labels
                # 取前100快一点！
                test_images, test_labels = test_images, test_labels
                # 二值化
                # train_images = binary(train_images)
                # test_images = binary(test_images)

                print('K:{}'.format(k))
                print("读取完成")
                test_size = len(test_images)
                print("样本大小：{}".format(len(train_images)))
                print("正在分类：{}".format(test_size))

                multiList = []
                for i in range(test_size):
                    multiList.append([test_images[i], train_images, train_labels, k, test_labels[i]])
                timeA = datetime.datetime.now()
                pool = Pool(4)
                outY = pool.map(multiPack, multiList)
                pool.close()
                pool.join()
                timeB = datetime.datetime.now()
                print("用时{}".format(timeB - timeA))
                count = 0
                for flag in outY:
                    if flag:
                        count += 1
                acc = count / test_size
                print("正确率：{:.4f}".format(acc))

                writeDict = {}
                writeDict['k'] = k
                writeDict['train'] = len(train_images)
                writeDict['test'] = test_size
                writeDict['time'] = (timeB - timeA).seconds
                writeDict['acc'] = acc
                with open('./result/' + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + '.txt', 'w') as f:
                    json.dump(writeDict, f)
