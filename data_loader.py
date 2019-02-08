import scipy
from glob import glob
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2

class DataLoader():
    def __init__(self, dataset_name, img_res=(128, 128)):
        self.dataset_name = dataset_name
        self.img_res = img_res

    def load_data(self, batch_size=1, is_testing=False):
        data_type = "train" if not is_testing else "test"
        #path = glob('./datasets/%s/%s/*' % (self.dataset_name, data_type))
        path = os.listdir('/home/rd/recognition_reaserch/FACE/Dataset/lfw/')
        folder_path = '/home/rd/recognition_reaserch/FACE/Dataset/lfw/'
        batch_images = np.random.choice(path, size=batch_size)
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

        imgs_A = []
        imgs_B = []
        count = 0
        boolcheck = False

        while(1):

            if boolcheck:
                break
            for imgdir_path in batch_images:
                if boolcheck:
                    break
                idx = np.random.random_integers(0, len(glob(folder_path + imgdir_path + "/*")) - 1)
                img_path = glob(folder_path + imgdir_path + "/*")[idx]
                #for img_path in glob(folder_path + imgdir_path + "/*"):
                #img = self.imread(img_path)
                img = cv2.imread(img_path)
                #h, w, _ = img.shape
                #_w = int(w/2)
                #img_A, img_B = img[:, :_w, :], img[:, _w:, :]
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)

                for (x, y, w, h) in faces:
                    count = count + 1
                    if count > batch_size:
                        boolcheck = True
                        break

                    #sub_img = img[y:y + h, x:x + w]
                    sub_img = gray[y:y + h, x:x + w]
                    sub_img = cv2.resize(sub_img, self.img_res)
                    #sub_img = np.transpose(sub_img, (2, 0, 1))
                    img_A = np.array(sub_img)
                    img_B = img_A

                    # If training => do random flip
                    if not is_testing and np.random.random() < 0.5:
                        img_A = np.fliplr(img_A)
                        img_B = np.fliplr(img_B)

                    imgs_A.append(img_A)
                    imgs_B.append(img_B)

        imgs_A = np.array(imgs_A)/127.5 - 1.
        imgs_B = np.array(imgs_B)/127.5 - 1.

        return imgs_A, imgs_B

    def load_batch(self, batch_size=1, is_testing=False):

        data_type = "train" if not is_testing else "val"
        #path = glob('./datasets/%s/%s/*' % (self.dataset_name, data_type))
        path = os.listdir('/home/rd/recognition_reaserch/FACE/Dataset/lfw/')
        folder_path = '/home/rd/recognition_reaserch/FACE/Dataset/lfw/'
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        self.n_batches = int(len(path) / batch_size)

        for i in range(self.n_batches-1):
            batch = path[i*batch_size:(i+1)*batch_size]
            imgs_A, imgs_B = [], []
            count = 0
            boolcheck = False

            while(1):
            #for img_dir in batch:
                rd_num = np.random.random_integers(0, batch_size-1, (1, 1))
                img_dir = batch[rd_num[0][0]]

                if count > batch_size:
                    boolcheck = True
                    break

                idx = np.random.random_integers(0, len(glob(folder_path + img_dir + "/*")) - 1)
                img_list = glob(folder_path + img_dir + "/*")
                img = img_list[idx]
                img = cv2.imread(img)
                #h, w, _ = img.shape
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)

                for (x, y, w, h) in faces:
                    count = count + 1
                    if count > batch_size:
                        boolcheck = True
                        break
                    sub_img = gray[y:y + h, x:x + w]
                    #sub_img = img[y:y + h, x:x + w]
                    #sub_img = scipy.misc.imresize(sub_img, self.img_res)
                    sub_img = cv2.resize(sub_img, self.img_res)
                    #print(sub_img.shape)
                    #sub_img = np.transpose(sub_img, (2, 0, 1))
                    img_A = np.array(sub_img)
                    img_B = img_A

                    # If training => do random flip
                    if not is_testing and np.random.random() < 0.5:
                        img_A = np.fliplr(img_A)
                        img_B = np.fliplr(img_B)

                    imgs_A.append(img_A)
                    imgs_B.append(img_B)



            '''
            half_w = int(w/2)
            #img_A = img[:, :half_w, :]
            #img_B = img[:, half_w:, :]
            img_A = img
            img_B = img_A
            img_A = scipy.misc.imresize(img_A, self.img_res)
            img_B = scipy.misc.imresize(img_B, self.img_res)

            if not is_testing and np.random.random() > 0.5:
                    img_A = np.fliplr(img_A)
                    img_B = np.fliplr(img_B)

            imgs_A.append(img_A)
            imgs_B.append(img_B)
            '''

            imgs_A = np.array(imgs_A)/127.5 - 1.
            imgs_B = np.array(imgs_B)/127.5 - 1.
            yield imgs_A, imgs_B


    def imread(self, path):
        return scipy.misc.imread(path, mode='RGB').astype(np.float)
