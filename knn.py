import numpy as np

def classify(inX, dataSet, labels, k):
    """
    定义knn算法分类器函数
    :param inX: 测试数据
    :param dataSet: 训练数据
    :param labels: 分类类别
    :param k: k值
    :return: 所属分类
    """

    dataSetSize = dataSet.shape[0]  # shape（m, n）m列n个特征
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    # sqDistances = abs(sqDistances)  # 调整范数时使用
    distances = sqDistances ** 0.5  # 欧式距离
    sortedDistIndicies = distances.argsort()  # 排序并返回index

    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1  # default 0

    sortedClassCount = sorted(classCount.items(), key=lambda d: d[1], reverse=True)
    return sortedClassCount[0][0]


