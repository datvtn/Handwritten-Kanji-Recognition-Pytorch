import os
import random
from sklearn.model_selection import train_test_split

ROOT= "ETL9G_dataset"

def data_split():
    content= []
    for root, dirs, files in os.walk(ROOT):
        for filename in files:
            path= os.path.join(root, filename)
            label= path.split('/')[-2]
            content.append(path + ' ' + label + '\n')
    random.shuffle(content)
    train= ""
    valid= ""
    train_length= int(0.8*len(content))
    for v in content[train_length:]:
        valid += v
    for v in content[:train_length]:
        train+= v
    with open("ETL9G_dataset/ETL9G_train.txt", 'w') as f:
        f.write(train)
    with open("ETL9G_dataset/ETL9G_valid.txt", 'w') as f:
        f.write(valid)
    print ("Train and Valid split...")
    
if __name__ == "__main__":
    data_split()
