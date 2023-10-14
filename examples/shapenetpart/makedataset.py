import os
import random
import time
import calendar
import shutil

def remaketxtname(path,newpath):
    filearrs = os.listdir(path)
    for index, file in enumerate(filearrs):
        newfilepath = newpath + str(int(calendar.timegm(time.gmtime())) - index -1) + ".txt"
        shutil.move(path + file,newfilepath)

def maketrain_test_val_set(txtfilepath,trainsetpath,textsetpath,valsetpath):
    filearrs = os.listdir(txtfilepath)
    filepatharrs = []
    indexmaparr = []

    trainarr = []
    testarr = []
    valarr = []


    filebegins = "shape_data/20230515/"

    for index, file in enumerate(filearrs):
        usedpath = filebegins + file[:-4]
        filepatharrs.append(usedpath)
        indexmaparr.append(index)

    random.shuffle(indexmaparr)
    limitation = int(len(indexmaparr)*0.8)

    for index in indexmaparr[:limitation]:
        trainarr.append(filepatharrs[index])
       # valarr.append(filepatharrs[index])
        #testarr.append(filepatharrs[index])


    for index in indexmaparr[limitation:]:
        valarr.append(filepatharrs[index])
        testarr.append(filepatharrs[index])

        #valarr.append(filepatharrs[index])

    with open(trainsetpath,'w') as f:
        f.write(str(trainarr).replace('\'', '\"'))

    with open(valsetpath,'w') as f:
        f.write(str(valarr).replace('\'', '\"'))

    with open(textsetpath,'w') as f:

        f.write(str(testarr).replace('\'', '\"'))




if __name__ == '__main__':

#### 重命名txt文件
    # path = "D:/pythonproject/Pointnext/data/ShapeNetPart/typeb_120_useddatasetxyz/new/"
    # newpath = "D:/pythonproject/Pointnext/data/ShapeNetPart/typeb_120_useddatasetxyz/temp/"
    # remaketxtname(path,newpath)

###分割traing and val
    txtfilepath = "D:/pythonproject/Pointnext/data/ShapeNetPart/useddatasetxyz/20230515/"
    trainsetpath = "D:/pythonproject/Pointnext/data/ShapeNetPart/useddatasetxyz/train_test_split/shuffled_train_file_list.json"
    textsetpath = "D:/pythonproject/Pointnext/data/ShapeNetPart/useddatasetxyz/train_test_split/shuffled_test_file_list.json"
    valsetpath = "D:/pythonproject/Pointnext/data/ShapeNetPart/useddatasetxyz/train_test_split/shuffled_val_file_list.json"
    maketrain_test_val_set(txtfilepath, trainsetpath, textsetpath, valsetpath)