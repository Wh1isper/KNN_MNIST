import os
import json
import csv

if __name__ == '__main__':

    totalList = []

    for _, _, filenames in os.walk('./result'):
        curSet = []
        for file in filenames:
            with open(os.path.join("./result", file), 'r') as f:
                cur_dict = json.load(f)
            totalList.append(cur_dict)
    with open('result.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["k", "train", "test", "time", "acc"])
        for Dict in totalList:
            writer.writerow([Dict['k'], Dict['train'], Dict['test'], Dict['time'], Dict['acc']])
