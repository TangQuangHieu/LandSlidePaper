import os
from random import shuffle
import cv2
import numpy as np
# import tensorflow as torch
import torch
import json 
from util2 import *
import torch.nn.functional as F
from scipy import ndimage

import numpy as np
import torch
import torchvision.transforms as T
from torchvision.transforms import functional as VF
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt

class DataGenerator(object):
    def __init__(self, data_dir='../Nepal_landslide_dataset', batch_size=12, stored_folder="../results/LandSliding_Nepal_Segment"):
        self.data_dir = data_dir 
        self.batch_size = batch_size 
        self.stored_folder = stored_folder
        self.train_data=dict() # key is index, values include image, label
        self.val_data=dict() # key is index, values include image, label
        self.test_data=dict() # key is index, values include image, label
        self.n_train=0
        self.n_val=0 
        self.n_test=0
        self.train_test_split(merge_train_val=True) 
        self.train_idxs = np.arange(self.n_train)
        self.val_idxs = np.arange(self.n_val)
        self.test_idxs=np.arange(self.n_test)
        self.train_transforms = T.Compose([
            T.ColorJitter(brightness=0.2),
            # AdjustBrightness(brightness_factor=1.2),  # Adjust brightness by a factor of 1.2
            # AdjustSaturation(saturation_factor=1.5),  # Adjust saturation by a factor of 1.5
            AdjustSharpness(sharpness_factor=2.0),  # Adjust sharpness by a factor of 2.0
            # AddNoise(mean=0.0, std=0.05),  # Add Gaussian noise with mean=0 and std=0.05
            T.ToTensor(),  # Convert the image to a tensor range [0,1]
            T.Normalize(mean=[0.29858148,0.39586081,0.09152383] , 
                        std=[0.02363239,0.03083374,0.0127833 ])  # Normalize the image
        ])

        self.test_transforms = T.Compose([
            # T.ColorJitter(brightness=0.2),
            # AdjustBrightness(brightness_factor=1.2),  # Adjust brightness by a factor of 1.2
            # AdjustSaturation(saturation_factor=1.5),  # Adjust saturation by a factor of 1.5
            AdjustSharpness(sharpness_factor=2.0),  # Adjust sharpness by a factor of 2.0
            # AddNoise(mean=0.0, std=0.05),  # Add Gaussian noise with mean=0 and std=0.05
            T.ToTensor(),  # Convert the image to a tensor range [0,1]
            T.Normalize(mean=[0.29858148,0.39586081,0.09152383] , 
                        std=[0.02363239,0.03083374,0.0127833 ])  # Normalize the image
        ])

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('Device Info:',self.device)

    def train_test_split(self,merge_train_val=False):
        """
        Read data from 3 folders including:
        - Training(for train data)
        - Test(for test data)
        - Validation(for validation)
        - merge_train_val: merge train and val together
        ### Returns: 
            None but loading data into self.train_data, self.test_data, and
        self.val_data (prefecth data into RAM)
        """ 
        for idx,img_name in enumerate(os.listdir(os.path.join(self.data_dir,'Training','image'))):
            img_path = os.path.join(self.data_dir,'Training','image',img_name)
            label_path= os.path.join(self.data_dir,'Training','label',img_name.replace('.tiff','_mask.png').replace('b_COMP_','COMP_'))
            print('label_path',label_path)
            img = Image.open(img_path)
            img = img.resize((128,128))
            label=Image.open(label_path)
            label=label.resize((128,128))
            self.train_data[idx]=(img_name,img,label)
            self.n_train+=1
        n_train = self.n_train
        if merge_train_val:
            for idx,img_name in enumerate(os.listdir(os.path.join(self.data_dir,'Validation','images'))):
                img_path = os.path.join(self.data_dir,'Validation','images',img_name)
                label_path= os.path.join(self.data_dir,'Validation','masks',img_name.replace('.tiff','.png'))
                print('label_path',label_path)
                img = Image.open(img_path)
                img = img.resize((128,128))
                label=Image.open(label_path)
                label=label.resize((128,128))
                idx+=n_train
                self.train_data[idx]=(img_name,img,label)
                self.n_train+=1
        for idx,img_name in enumerate(os.listdir(os.path.join(self.data_dir,'Validation','images'))):
            img_path = os.path.join(self.data_dir,'Validation','images',img_name)
            label_path= os.path.join(self.data_dir,'Validation','masks',img_name.replace('.tiff','.png'))
            img = Image.open(img_path)
            img = img.resize((128,128))
            label=Image.open(label_path)
            label=label.resize((128,128))
            self.val_data[idx]=(img_name,img,label)
            self.n_val+=1
        for idx,img_name in enumerate(os.listdir(os.path.join(self.data_dir,'Test','image'))):
            img_path = os.path.join(self.data_dir,'Test','image',img_name)
            label_path= os.path.join(self.data_dir,'Test','mask',img_name.replace('.tiff','.png'))
            img = Image.open(img_path)
            img = img.resize((128,128))
            label=Image.open(label_path)
            label=label.resize((128,128))
            self.test_data[idx]=(img_name,img,label)
            self.n_test+=1
        
    
    def _calcDataStatistic(self):
        """
        Calculate mean, std of dataset for image
        return: mean,std of dataset for each channel 
        mean of data: [0.29858148 0.39586081 0.09152383] 
        std of data: [0.02363239 0.03083374 0.0127833 ]
        """
        self.means = np.array([0.,0.,0.]) 
        self.stds = np.array([0.,0.,0.])
        # landslide_name_list = [img for img in os.listdir(self.landslide_img_dir)]
        # non_landslide_name_list = [img for img in os.listdir(self.non_landslide_img_dir)]
        for i in range(self.n_train):
            name,img,mask = self.train_data[i]
            img = np.array(img)
            self.means += np.sum(img,axis=(0,1))/255.
        for i in range(self.n_test):
            name,img,mask = self.test_data[i]
            img = np.array(img)
            self.means += np.sum(img,axis=(0,1))/255.
        for i in range(self.n_val):
            name,img,mask = self.val_data[i]
            img = np.array(img)
            self.means += np.sum(img,axis=(0,1))/255.
        self.means /= (self.n_train+self.n_val+self.n_test)*(128**2)
        means = np.ones((128,128,3))
        means[...,0]*=self.means[0]
        means[...,1]*=self.means[1]
        means[...,2]*=self.means[2]

        for i in range(self.n_train):
            name,img,mask = self.train_data[i]
            img = np.array(img)/255.
            self.stds += np.sum((img-means)**2,axis=(0,1))

        for i in range(self.n_test):
            name,img,mask = self.test_data[i]
            img = np.array(img)/255.
            self.stds += np.sum((img-means)**2,axis=(0,1))

        for i in range(self.n_val):
            name,img,mask = self.val_data[i]
            img = np.array(img)/255.
            self.stds += np.sum((img-means)**2,axis=(0,1))

        self.stds /=(self.n_train+self.n_val+self.n_test)*(128**2)
        print('mean of data:',self.means,'std of data:',self.stds)
        return self.means,self.stds     
    
    def _shift(self, img, label,max_shift = 30, prob=0.7):
        """
        Shift an image and its label randomly in X and Y axes.
        Support morphology augmentation image and label for training 
        ### Arguments:
            img(torch.tensor): CxWxH image to be shifted 
            label(torch.tensor): WxH label to be shifted
            max_shift: maximum displament
            pro(float): probability that an image will be shifted
        ### Returns:
            img,label shifted
        """
        if np.random.uniform()>prob:
            x_shift = random.uniform(-max_shift, max_shift)
            y_shift = random.uniform(-max_shift, max_shift)
            img = VF.affine(img, angle=0, translate=(x_shift, y_shift), scale=1, shear=0)
            # torch.unsqueeze
            label = VF.affine(label, angle=0, translate=(x_shift, y_shift), scale=1, shear=0)
        return img, label
    
    def _rotate(self, img, label,degree=30,prob=0.7):
        """
        Rotate an image and its label randomly in degree.
        Support morphology augmentation image and label for training 
        ### Arguments:
            img(torch.tensor): CxWxH image to be rotated 
            label(torch.tensor): WxH label to be rotated
            pro(float): probability that an image will be rotated
            degree: degree to be rotated
        ### Returns:
            rotated img,label 
        """
        if np.random.uniform()>prob:
            angle = random.uniform(-degree, degree)
            img = VF.rotate(img, angle)
            # label = label.unsqueeze(0)
            label = VF.rotate(label, angle)
        return img, label

    def _cutmix(self,data:dict,img:torch.tensor,label:torch.tensor,prob=0.3):
        """
        Cut and mix together to increase the view of the batch 
        ### Arguments:
            data (dictionary): whole dataset 
            img(torch.tensor): An image in batch 
            label(torch.tensor): A label in batch 
            prob: probability to perform cut mix
        ### Returns:
            img and label that are cut and mix together 
        """
        n = len(data)
        rnd_idxs = np.arange(n)
        np.random.shuffle(rnd_idxs)
        if np.random.uniform() > prob: #mixing data 4 time to increase num of landslide region !!!
            for i in range(4):
                j = rnd_idxs[i]
                name2,img2,label2 = data[j]
                # print('img cut mix:',name2)
                img2 = self.train_transforms(img=img2) 
                label2 = np.array(label2)
                label2 = torch.tensor(label2,dtype=torch.float32)
                # label = label.unsqueeze(0)
                mask1 = torch.clamp(label,min=0,max=1)
                mask2 = torch.clamp(label2,min=0,max=1)
                intersect = mask2*mask1 
                exclusion = torch.clamp(mask1+mask2-2*intersect,min=0,max=1)
                mask1_only = torch.clamp(mask1-intersect,min=0,max=1) 
                mask2_only = torch.clamp(mask2-intersect,min=0,max=1)
                background_mask = (1-(mask1+mask2-intersect))
                # Intersect region of landslide will be treated as 50% for img1, 50% for img2
                # Other regions will be treated as 80% for img1, 20% for img2 
                img=0.5*(img+img2)*intersect+(0.8*img+0.2*img2)*(mask1_only+background_mask)+(0.2*img+0.8*img2)*mask2_only
                label = torch.clamp(label+label2,min=0,max=1)
            # not mix with label of imgs, maybe we need to do it
        return img,label

    def morph_transforms(self,data:dict,img:torch.tensor,label:torch.tensor):
        """
        Performce morphology transformations including cutmix, rotate, and shift
        ### Arguments:
            data(dictionary): whole dataset 
            img(torch.tensor): CxWxH image to be augumented 
            label(torch.tensor):WxH image to be augmented
        """
        img,label = self._cutmix(data=data,img=img,label=label)
        # print('after cut mix:img shape, label shape:',img.shape,label.shape)
        img,label = self._rotate(img=img,label=label)
        # print('after rotate:img shape, label shape:',img.shape,label.shape)

        img,label = self._shift(img=img,label=label)
        # print('after shift:img shape, label shape:',img.shape,label.shape)

        return img,label

    def getBatch(self,batch_id,mode=0,is_aug=True):
        """
        Get batch of image:
        ### Arguments:
            batch_id: id of batch 
            mode:0 (train),1(val),2(test)
            is_aug: augmentation input image 
        """
        idxs = None
        data = None
        transforms = None
        if mode==0: #train
            data = self.train_data
            idxs=self.train_idxs
            transforms = self.train_transforms
        elif mode==1: #val
            data = self.val_data
            idxs=self.val_idxs
            transforms = self.test_transforms
        else: #test 
            data = self.test_data
            idxs=np.arange(self.n_test)
            transforms = self.test_transforms
        n = len(idxs)
        # np.random.shuffle(idxs)
        start_idx = batch_id*self.batch_size 
        stop_idx = (batch_id+1)*self.batch_size if (batch_id+1)*self.batch_size<n else n 
        input_128 = None 

        mask_list_128 = None 
        mask_list_256 = None 
        mask_list_64 = None
        for i in range(start_idx,stop_idx):
            name,img_src,label=data[idxs[i]]
            # Create a new image with the same size and mode as the source image
            img = Image.new(img_src.mode, img_src.size)

            # Copy the source image into the new image
            img.paste(img_src)
            label = torch.tensor(np.array(label))
            label = torch.clamp(label,min=0,max=1).unsqueeze(0)

            if is_aug:
                img = transforms(img)
                if mode==0:
                    # apply morphology transforms for training only
                    img,label= self.morph_transforms(data=data,img=img,label=label)
            else:
                img = self.test_transforms(img)
            img = img.unsqueeze(0)
            input_128 = img if input_128 is None else torch.concat((input_128,img))
            mask_list_128 = label if mask_list_128 is None else torch.concat((mask_list_128,label))

        # mask_list_128 = mask_list_128.unsqueeze(dim=1)
        mask_list_128 = mask_list_128.unsqueeze(1)
        # print('mask_list_128',mask_list_128.shape)
        mask_list_256 = F.interpolate(mask_list_128, size=(256, 256), mode='nearest')
        mask_list_64 = F.interpolate(mask_list_128, size=(64, 64), mode='nearest')
        mask_list_128 = torch.squeeze(mask_list_128,dim=1)
        mask_list_256 =torch.squeeze(mask_list_256,dim=1)
        mask_list_64 = torch.squeeze(mask_list_64,dim=1)

        # mask_list_128 = torch.clamp(mask_list_128,min=0,max=1).to(torch.float32)
        mask_list_256 = torch.clamp(mask_list_256,min=0,max=1).to(torch.int32)
        mask_list_64 = torch.clamp(mask_list_64,min=0,max=1).to(torch.int32)
        mask_list_128 = mask_list_128.to(torch.int32)
        return input_128,mask_list_64,mask_list_128,mask_list_256

    def getNumBatch(self,mode=0)->int:
        """
        Get total batch numbers and last batch image number
        ### Arguments:
            mode: 0 for training, 1 for validating, 2 for testing 
        ### Returns:
            num_batch,num_img_in_last_batch
        """
        num_batch=0 
        num_img_last_batch = 0 
        n = 0 
        if mode == 0 :
            n = self.n_train 
        elif mode == 1:
            n = self.n_val 
        else:
            n = self.n_test 
        num_batch = n//self.batch_size 
        num_img_last_batch=n%self.batch_size 
        if num_batch==0 and num_img_last_batch>0: num_batch=1 # Test size is smaller than batch_num
        return num_batch,num_img_last_batch  

    def getNumImg(self,mode=0)->int:
        """
        Get number of image
        ### Arguments:
            mode: 0 train, 1 val, 2 test 
        ### Returns:
            Number of corresponse mode
        """
        if mode==0:return  self.n_train 
        elif mode==1:return self.n_val 
        else: return self.n_test

    def shuffle_train(self):
        """
        Shuffle training set 
        ### Returns:
            self.train_data is shuffled
        """ 
        np.random.shuffle(self.train_idxs)
        np.random.shuffle(self.val_idxs)
        np.random.shuffle(self.test_idxs)
def main():
    print('>>>>>>>>>>>>>>>>>1-TESTING PREFECT<<<<<<<<<<<<<<<<<<<<<<<')
    generator = DataGenerator()
    input('Wait here until enter any key ... please check RAM usage to make sure there is no overload problem')
    print('>>>>>>>>>>>>>>>>>>DONE 1<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    print('>>>>>>>>>>>>>>>>>>2- TESTING GET BATCH<<<<<<<<<<<<<<<<<<<<<<<')
    input_128,mask_list_64,mask_list_128,mask_list_256=generator.getBatch(batch_id=0,mode=0)
    print('input shape:',input_128.shape,'output shape:',mask_list_128.shape) 
    i = 0 
    img = np.array(input_128[i].permute(1,2,0))
    mask_128=np.array(mask_list_128[i].squeeze(0))
    mask_64 =np.array(mask_list_64[i].squeeze(0)) 
    mask_256=np.array(mask_list_256[i].squeeze(0)) 
    # Recover image from normalization
    mean=np.array([0.29858148,0.39586081,0.09152383])
    std=np.array([0.02363239,0.03083374,0.0127833])
    means= np.ones_like(img)
    stds = np.ones_like(img)
    means[...,0]*=mean[0]
    means[...,1]*=mean[1]
    means[...,2]*=mean[2]
    stds[...,0]*=std[0]
    stds[...,1]*=std[1]
    stds[...,2]*=std[2]    
    img = (img*stds+means)*255
    img = img.astype(np.uint8)
    img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    img_seg = draw_segment_on_image(img,mask_128)
    draw_image(img,'image.png')
    draw_image(img_seg,'image_segment.png')
    draw_image(mask_128*255,'mask128.png')
    draw_image(mask_64*255,'mask64.png')
    print('>>>>>>>>>>>>>>>>>>DONE 2<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    print('>>>>>>>>>>>>>>>>>>3- Testing image<<<<<<<<<<<<<<<<<<<<<<')
    name,img,label = generator.train_data[0]
    label = np.array(label)*255
    label = label.astype(np.uint8)
    print('Name of image:',name)
    img = np.array(img)
    img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    
    # print(label)
    draw_image(img,'image_original.png')
    draw_image(label,'label_original.png')
    img_seg = draw_segment_on_image(img,label)
    draw_image(img_seg,'image_original_segment.png')
    print('>>>>>>>>>>>>>>>>>>DONE 3<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')

if __name__=="__main__":
    main()
