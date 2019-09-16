from utils import *
import matplotlib.pyplot as plt

train_images, train_labels, test_images, test_labels = load()

print(test_labels)
fig=plt.figure(figsize=(8,8))
fig.subplots_adjust(left=0,right=1,bottom=0,top=1,hspace=0.05,wspace=0.05)
for i in range(30):
    images = np.reshape(train_images[i], [28,28])
    ax=fig.add_subplot(6,5,i+1,xticks=[],yticks=[])
    ax.imshow(images,cmap=plt.cm.binary,interpolation='nearest')
    ax.text(0,7,str(train_labels[i]))
plt.show()
