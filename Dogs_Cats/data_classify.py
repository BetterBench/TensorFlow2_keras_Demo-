import shutil
import os


def remove_file(old_path, new_path):
    print(old_path)
    print(new_path)
    filelist = os.listdir(old_path)  # 列出该目录下的所有文件,listdir返回的文件列表是不包含路径的。
    # print(filelist)
    cat_n =0
    dog_n = 0
    for file in filelist:
        src = os.path.join(old_path, file)
        if not os.path.isfile(src):
            continue
        animal_str = str(file).split('.')[0]
        if animal_str =='cat': 
            if cat_n < 2500:
                cat_path = os.path.join('./validation/cat/', file)
                shutil.move(src, cat_path)
                cat_n +=1
            elif cat_n < 12500:
                cat_path = os.path.join('./train/cat/', file)
                shutil.move(src, cat_path)
                cat_n += 1
            else:
                os.remove(src)
        elif animal_str == 'dog':
            if dog_n < 2500:
                dog_path = os.path.join('./validation/dog/', file)
                shutil.move(src, dog_path)
                dog_n += 1
            elif dog_n < 12500:
                dog_path = os.path.join('./train/dog/', file)
                dog_n += 1
                shutil.move(src, dog_path)
            else:
                os.remove(src)
        else:
            continue
if __name__ == '__main__':
    os.makedirs('./train/cat/')
    os.makedirs('./train/dog/')
    os.makedirs('./validation/cat/')
    os.makedirs('./validation/dog/')
    remove_file(r"./train/", r"./validation/")
