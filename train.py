from rnn import classify
from utils import load
import numpy as np
from tqdm import tqdm

train_images, train_labels, test_images, test_labels = load()
# 取前100快一点！
# test_images, test_labels = test_images[:100], test_labels[:100]

print("读取完成")
test_size = test_images.shape[0]
print("正在分类：{}".format(test_size))
outY = [classify(test_images[i], train_images, train_labels, 10) for i in tqdm(range(test_size))]
# print("KNN输出:{}".format(outY))
# print("实际标签：{}".format(test_labels))
acc = np.mean(outY == test_labels)
print('\n\n')
print("正确率：{}".format(acc))
# 正确率：0.9683
