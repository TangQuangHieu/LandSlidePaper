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

    def __calcDataStatistic(self):
        """
        Calculate mean, std of dataset for image
        return: mean,std of dataset for each channel 
        """
        self.means = np.array([0.,0.,0.]) 
        self.stds = np.array([0.,0.,0.])
        landslide_name_list = [img for img in os.listdir(self.landslide_img_dir)]
        non_landslide_name_list = [img for img in os.listdir(self.non_landslide_img_dir)]
        for img_name in landslide_name_list:
            img = Image.open(os.path.join(self.landslide_img_dir,img_name))
            img = img.resize((128,128))
            img = np.array(img)/255. 
            self.means +=np.sum(img,axis=(0,1))
        for img_name in non_landslide_name_list:
            img = Image.open(os.path.join(self.non_landslide_img_dir,img_name))
            img = img.resize((128,128))
            img = np.array(img)/255. 
            self.means +=np.sum(img,axis=(0,1))
        n = (len(landslide_name_list)+len(non_landslide_name_list))
        self.means = self.means/n/(128**2)
        for img_name in landslide_name_list:
            img = Image.open(os.path.join(self.landslide_img_dir,img_name))
            img = img.resize((128,128))
            img = np.array(img)/255. 
            self.stds+=(np.sum(img,axis=(0,1))/(128**2)-self.means)**2
        for img_name in non_landslide_name_list:
            img = Image.open(os.path.join(self.non_landslide_img_dir,img_name))
            img = img.resize((128,128))
            img = np.array(img)/255. 
            self.stds+=(np.sum(img,axis=(0,1))/(128**2)-self.means)**2
        self.stds /= n
        self.stds = np.sqrt(self.stds)
        print('means:',self.means,'std:',self.stds) 
        return self.means,self.stds

    def __init__(self, data_dir, batch_size,  test_ratio=0.2, test_fold=1, stored_folder="../results",read_file=False):
        self.stored_folder = stored_folder
        self.data_dir        = data_dir # ../Bijie-landslide-dataset
        self.landslide_dir   = os.path.join(data_dir,"landslide")
        self.landslide_img_dir = os.path.join(self.landslide_dir,"image")
        self.landslide_mask_dir = os.path.join(self.landslide_dir,"mask")
        self.landslide_dem_dir = os.path.join(self.landslide_dir,"dem")
        self.non_landslide_dir = os.path.join(data_dir,"non-landslide")
        self.non_landslide_img_dir = os.path.join(self.non_landslide_dir,"image")
        self.non_landslide_mask_dir = os.path.join(self.non_landslide_dir,"mask")
        self.non_landslide_dem_dir = os.path.join(self.non_landslide_dir,"dem")
        self.test_fold       = test_fold
        self.n_img           = len(os.listdir(self.landslide_img_dir)) + len(os.listdir(self.non_landslide_img_dir))
        self.test_ratio = test_ratio
        self.is_read_file = read_file
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if read_file:
            self.read_train_test_data(stored_folder) #Continue training
        else:
            self.train_test_split(test_ratio,over_sample=True)
        self.batch_size      = batch_size

        self.transforms = T.Compose([
    T.Resize((128, 128)),  # Resize the image to 256x256
    T.ColorJitter(brightness=0.2),
    # AdjustBrightness(brightness_factor=1.2),  # Adjust brightness by a factor of 1.2
    # AdjustSaturation(saturation_factor=1.5),  # Adjust saturation by a factor of 1.5
    AdjustSharpness(sharpness_factor=2.0),  # Adjust sharpness by a factor of 2.0
    # AddNoise(mean=0.0, std=0.05),  # Add Gaussian noise with mean=0 and std=0.05
    T.ToTensor(),  # Convert the image to a tensor range [0,1]
    T.Normalize(mean=[0.3028171,0.37409051,0.31993082], std=[0.0922879,0.08401628,0.09078308])  # Normalize the image
])
    
    def write_train_test_data(self,path:str):
        """
        write_train_test_data(self,path:str): write train image, test image for 
        2nd stage using classification
        ### Arguments:
            path(str): path to result folder (Contains model, train, test history as well)
        ### Returns:
            None: train data and test data are saved 
        """
        with open(os.path.join(path,"train_data_path.json"),'w') as f:
            data={'train_img':self.train_img,
                  'train_dem':self.train_dem,
                  'train_mask':self.train_mask}
            json.dump(data,f)
        with open(os.path.join(path,'test_data_path.json'),'w') as f:
            data={'test_img':self.test_img,
                  'test_dem':self.test_dem,
                  'test_mask':self.test_mask
            }
            json.dump(data,f)
    
    def read_train_test_data(self,path:str):
        """
        read_train_test_data(self,path:str): Read back previous data stored in path 
        ### Arguments: 
            path(str): path to folder that contains train_data_path.json and test_data_path.json
        ### Returns:
            None but load data into member variables base on the contain of data file (ignore train test split)
        """
        try:
            with open(os.path.join(path,"train_data_path.json"),'r') as f:
                data = json.load(f)
                print('before unique',len(data['train_img']),'after unique',len(set(data['train_img'])))
                unique_img = list(data['train_img'])
                unique_dem = [img.replace('image','dem') for img in unique_img]
                unique_mask= [img.replace('image','mask') for img in unique_img]
                self.train_img = unique_img
                self.train_dem = unique_dem
                self.train_mask = unique_mask
            with open(os.path.join(path,'test_data_path.json'),'r') as f:
                data = json.load(f)
                self.test_img = data['test_img']
                self.test_dem = data['test_dem']
                self.test_mask = data['test_mask']
            # Calculate num of train, val data 
            self.n_img_train = len(self.train_img)
            self.n_img_val = len(self.test_img)
        except FileNotFoundError:
            print('The file does not exist.')
        except json.JSONDecoderError:
            print("File is not a valid JSON.")
    
    def train_test_split(self,test_ratio,over_sample=True):
        """
        train_test_split(self,test_ratio): split data using test_ratio
        ### Parameters:
            test_ratio(float): test ratio
        ### Returns:
            None but train and test paths are stored inside class member variables 
        """
        # Prepare landslide images 
        img_name_list = [img for img in os.listdir(self.landslide_img_dir)]
        n = len(img_name_list)
        # print("img_name_list",img_name_list) # OK
        shuffle_idxs = np.arange(n)
        np.random.shuffle(shuffle_idxs)
        img_name_list_shuffle = [None]*n
        for i,j in zip(range(n),shuffle_idxs):
            img_name_list_shuffle[i] = img_name_list[j]
        # print("img_name_list_shuffle",img_name_list_shuffle) # OK
        self.landslide_img = [os.path.join(self.landslide_img_dir,img_name) for img_name in img_name_list_shuffle]
        self.landslide_dem = [os.path.join(self.landslide_dem_dir,dem_name) for dem_name in img_name_list_shuffle]
        self.landslide_mask = [os.path.join(self.landslide_mask_dir,mask_name) for mask_name in img_name_list_shuffle]
        # Lay random index tu tap landslide de chia train va test
        # print(self.landslide_img) # OK
        landslide_idxs = [i for i in range(len(self.landslide_img))]
        shuffle(landslide_idxs)
        landslide_test_num = int(test_ratio * len(landslide_idxs))
        self.test_landslide_idxs = landslide_idxs[:landslide_test_num]
        self.train_landslide_idxs = landslide_idxs[landslide_test_num:]
        # print(train_landslide_idxs) # OK

        # Prepare paths for non landslide data
        img_name_list = [img for img in os.listdir(self.non_landslide_img_dir)]
        n = len(img_name_list)
        shuffle_idxs = np.arange(n)
        np.random.shuffle(shuffle_idxs)
        img_name_list_shuffle = [None]*n
        for i,j in zip(range(n),shuffle_idxs):
            img_name_list_shuffle[i] = img_name_list[j]
        self.non_landslide_img = [os.path.join(self.non_landslide_img_dir,img_name) for img_name in img_name_list_shuffle]
        self.non_landslide_dem  = [os.path.join(self.non_landslide_dem_dir ,dem_name) for dem_name in img_name_list_shuffle]
        self.non_landslide_mask = [os.path.join(self.non_landslide_mask_dir,mask_name) for mask_name in img_name_list_shuffle]

        # Lay random index tu tap landslide de chia train va test
        non_landslide_idxs = [i for i in range(len(self.non_landslide_img))]
        shuffle(non_landslide_idxs)
        non_landslide_test_num = int(test_ratio * len(non_landslide_idxs))
        self.test_non_landslide_idxs = non_landslide_idxs[:non_landslide_test_num]
        self.train_non_landslide_idxs = non_landslide_idxs[non_landslide_test_num:]


        # Get list train data of landslide
        self.train_img=[]
        self.train_dem=[]
        self.train_mask=[]
        self.landslide_train_imgs = [] # Contain all landslide train images 
        # used to support cutmix
        self.landslide_train_masks = [] # Contain all landslide train mask 
        for idx in range(len(self.train_landslide_idxs)):
            train_landslide_idx = self.train_landslide_idxs[idx]
            train_non_landslide_idx = self.train_non_landslide_idxs[idx]
            self.landslide_train_imgs.append(self.landslide_img[train_landslide_idx])
            self.train_img.append(self.landslide_img[train_landslide_idx])
            self.train_img.append(self.non_landslide_img[train_non_landslide_idx])
            self.train_dem.append(self.landslide_dem[train_landslide_idx])
            self.train_dem.append(self.non_landslide_dem[train_non_landslide_idx])
            self.landslide_train_masks.append(self.landslide_mask[train_landslide_idx])
            self.train_mask.append(self.landslide_mask[train_landslide_idx])
            self.train_mask.append(self.non_landslide_mask[train_non_landslide_idx])
        # print(len(self.train_img)) # OK
        #Over sampling landslide image to make it equal to non landslide image 
        for idx in range(len(self.train_landslide_idxs),len(self.train_non_landslide_idxs)):
            train_landslide_idx = self.train_landslide_idxs[idx%len(self.train_landslide_idxs)]
            train_non_landslide_idx = self.train_non_landslide_idxs[idx]
            self.train_img.append(self.non_landslide_img[train_non_landslide_idx])
            self.train_dem.append(self.non_landslide_dem[train_non_landslide_idx])
            self.train_mask.append(self.non_landslide_mask[train_non_landslide_idx])
            if over_sample:
                self.train_img.append(self.landslide_img[train_landslide_idx])
                self.train_dem.append(self.landslide_dem[train_landslide_idx])
                self.train_mask.append(self.landslide_mask[train_landslide_idx])
            
        # Get list test data of landslide
        self.test_img=[]
        self.test_dem=[]
        self.test_mask=[]
        for test_idx in self.test_landslide_idxs:
            self.test_img.append(self.landslide_img[test_idx])
            self.test_dem.append(self.landslide_dem[test_idx])
            self.test_mask.append(self.landslide_mask[test_idx])
        for test_idx in self.test_non_landslide_idxs:
            self.test_img.append(self.non_landslide_img[test_idx])
            self.test_dem.append(self.non_landslide_dem[test_idx])
            self.test_mask.append(self.non_landslide_mask[test_idx])        
        # Calculate num of train, val data 
        self.n_img_train = len(self.train_img)
        self.n_img_val = len(self.test_img)
        self.write_train_test_data(self.stored_folder) # Write train_test data into file
        # self.calcDataStatistic()
    
    def shuffle_train(self):
        """
        shuffle_train_data(self): Shuffle all train data after each epoch 
        # Arguments: 
        # Returns:
            self.train_img, self.train_mask, elf.train_dem will be shuffled 
        """
        # Get list train data of landslide
        odd_idxs = [i for i in range(1,len(self.train_img),2)]
        even_idxs = [i for i in range(0,len(self.train_img),2)]
        # print(odd_idxs,even_idxs)
        shuffle(odd_idxs)
        shuffle(even_idxs)
        n = len(self.train_img)
        train_img=[None]*n
        train_dem=[None]*n
        train_mask=[None]*n

        for i,j in zip(even_idxs,range(0,n,2)):
            # print(i,j)
            train_img[j] = self.train_img[i]
            train_dem[j] = self.train_dem[i]
            train_mask[j] = self.train_mask[i]

        for i,j in zip(odd_idxs,range(1,n,2)):
            # print(i,j)
            train_img[j] = self.train_img[i]
            train_dem[j] = self.train_dem[i]
            train_mask[j] = self.train_mask[i]

        # print('Test shuffle',len(self.train_img),train_img,self.n_img_train)
        self.train_img = train_img
        self.train_dem = train_dem 
        self.train_mask= train_mask 

        # self.train_img=[]
        # self.train_dem=[]
        # self.train_mask=[]
        # shuffle(self.train_landslide_idxs)
        # shuffle(self.train_non_landslide_idxs)
        # for idx in range(len(self.train_landslide_idxs)):
        #     train_landslide_idx = self.train_landslide_idxs[idx]
        #     train_non_landslide_idx = self.train_non_landslide_idxs[idx]
        #     self.train_img.append(self.landslide_img[train_landslide_idx])
        #     self.train_img.append(self.non_landslide_img[train_non_landslide_idx])
        #     self.train_dem.append(self.landslide_dem[train_landslide_idx])
        #     self.train_dem.append(self.non_landslide_dem[train_non_landslide_idx])
        #     self.train_mask.append(self.landslide_mask[train_landslide_idx])
        #     self.train_mask.append(self.non_landslide_mask[train_non_landslide_idx])
        # # print(len(self.train_img)) # OK
        # #Over sampling landslide image to make it equal to non landslide image 
        # for idx in range(len(self.train_landslide_idxs),len(self.train_non_landslide_idxs)):
        #     train_landslide_idx = self.train_landslide_idxs[idx%len(self.train_landslide_idxs)]
        #     train_non_landslide_idx = self.train_non_landslide_idxs[idx]
        #     self.train_img.append(self.landslide_img[train_landslide_idx])
        #     self.train_img.append(self.non_landslide_img[train_non_landslide_idx])
        #     self.train_dem.append(self.landslide_dem[train_landslide_idx])
        #     self.train_dem.append(self.non_landslide_dem[train_non_landslide_idx])
        #     self.train_mask.append(self.landslide_mask[train_landslide_idx])
        #     self.train_mask.append(self.non_landslide_mask[train_non_landslide_idx]) 
    
    def getNumBatch(self, op='train'):
        batch_total = self.getNumImg(op)
        if batch_total % self.batch_size == 0:
            return (int(batch_total/self.batch_size), 0)
        return (int(batch_total/self.batch_size) + 1, batch_total % self.batch_size)
    
    def getNumImg(self, op='train'):
        if op == 'train':
            return self.n_img_train
        return self.n_img_val
    def getBatch(self,batch_num,is_aug=False,is_train=True,is_cutmix=True,is_test=False):
        if is_train:
            img_list = self.train_img 
            mask_list = self.train_mask 
            dem_list  = self.train_dem 
            op = "train"
        else: 
            img_list = self.test_img 
            mask_list = self.test_mask 
            dem_list = self.test_dem 
            is_aug=False
            op = "val"
        ## handle last batch
        if (batch_num == (self.getNumBatch(op=op)[0]-1)) and (self.getNumBatch(op=op)[1] != 0):
            start_ind = batch_num*self.batch_size
            end_ind   = start_ind + self.getNumBatch(op=op)[1]
        else:
            start_ind = batch_num*self.batch_size
            end_ind   = (batch_num+1)*self.batch_size
        # Split into land slide and non land slide images 
        input_seq = None
        mask_list_128 = None
        n_images = 0
        SIZE=128
        hog_img=None
        edges=None
        for idx in range(start_ind,end_ind):
            n_images+=1
            # Get Image 
            img = Image.open(img_list[idx])
            # add edge 
            # img1 = np.array(img)
            # edges  = cv2.Canny(np.mean(img1,axis=2).astype(np.uint8),90,255)/255.
            # edge_features = torch.tensor(edges,dtype=torch.float32).unsqueeze(0)
            # print('edge_features.shape',edge_features.shape)
            # add hog 
            # hog_features,hog_img = hog_feature(img1)
            # hog_features = torch.tensor(hog_features,dtype=torch.float32).unsqueeze(0)
            # print('hog_features',hog_features.shape)
            # img = cv2.resize(img,(SIZE,SIZE),interpolation=cv2.INTER_NEAREST)
            # img = torch.tensor(img,dtype=torch.float32,device=self.device)
            if is_aug:
                img = self.transforms(img=img)
            else:
                img = T.Resize((SIZE, SIZE))(img)
                img = AdjustSharpness(sharpness_factor=2.0)(img)
                img = T.ToTensor()(img)
                img = T.Normalize(mean=[0.3028171,0.37409051,0.31993082], std=[0.0922879,0.08401628,0.09078308])(img)
            # print('img.shape',img.shape) 3x128x128
            
            mask = Image.open(mask_list[idx])
            mask = mask.resize(size=(SIZE,SIZE))
            mask =  T.ToTensor()(mask) # (mask,dtype=torch.float32,device=self.device)
            mask = mask.unsqueeze(0) #1x128x128
            if is_aug and np.random.uniform()>0.7:
                # rotate left-right image and mask 
                img = T.RandomVerticalFlip(p=1.0)(img)
                mask= T.RandomVerticalFlip(p=1.0)(mask)
            img = img.unsqueeze(0) #1x3x128x128
            
            input_seq = img if input_seq is None else torch.concatenate((input_seq,img))
            # Get masks
            # print('mask.shape',mask.shape)
            mask_list_128 = mask if mask_list_128 is None else torch.concatenate((mask_list_128,mask))
        if is_cutmix:
            non_land_slide_idxs=[]
            land_slide_idxs=[]
            for i in range(len(mask_list_128)):
                if torch.sum(mask_list_128[i]) < 1.:
                    non_land_slide_idxs.append(i)
                else:  land_slide_idxs.append(i)
            # if len(non_land_slide_idxs)>0 and len(land_slide_idxs)==0:
            #     print("The batch contains only non landslide image")
            # else:
            for i in non_land_slide_idxs:
                if np.random.uniform()>0.5:continue
                j = np.random.randint(0,len(self.landslide_train_imgs))
                img_land_slide = Image.open(img_list[j])
                if is_aug:
                    img_land_slide = self.transforms(img_land_slide)
                else:
                    img_land_slide = T.Resize((SIZE, SIZE))(img_land_slide)
                    img_land_slide = AdjustSharpness(sharpness_factor=2.0)(img_land_slide)
                    img_land_slide = T.ToTensor()(img_land_slide) # 3x128x128
                    img_land_slide = T.Normalize(mean=[0.3028171,0.37409051,0.31993082], std=[0.0922879,0.08401628,0.09078308])(img_land_slide)
                # img_land_slide = img_land_slide.unsqueeze(0) #1x3x128x128
                mask_land_slide = Image.open(self.landslide_train_masks[j])
                mask_land_slide = mask_land_slide.resize(size=(SIZE,SIZE))
                mask_land_slide =  T.ToTensor()(mask_land_slide) #1x128x128

                input_seq[i],mask_list_128[i] = self.cutmixPytorch(
                    img_land_slide=img_land_slide,
                            img_non_land_slide=input_seq[i],
                            mask_land_slide=mask_land_slide)

        # Swap axes to get (batch_size, num_classes, height, width)
        # mask_list_128 = torch.unsqueeze(mask_list_128,dim=1) # BxCxWxH
        mask_list_256 = F.interpolate(mask_list_128, size=(256, 256), mode='nearest')
        mask_list_64 = F.interpolate(mask_list_128, size=(64, 64), mode='nearest')
        mask_list_128 = torch.squeeze(mask_list_128,dim=1)
        mask_list_256 =torch.squeeze(mask_list_256,dim=1)
        mask_list_64 = torch.squeeze(mask_list_64,dim=1)

        mask_list_128 = torch.where(mask_list_128>0,1,0).to(torch.int32)
        mask_list_256 = torch.where(mask_list_256>0,1,0).to(torch.int32)
        mask_list_64 = torch.where(mask_list_64>0,1,0).to(torch.int32)
        if not is_test:
            return input_seq,mask_list_256,mask_list_128,mask_list_64,n_images #,img_list,mask_list
        else:
            return input_seq,mask_list_256,mask_list_128,mask_list_64,n_images ,img_list[start_ind:end_ind],mask_list[start_ind:end_ind],hog_img,edges

    def checkTrainTestDupplicate(self):
        """
        After loading data, let make sure there is no dupplication 
        of self.train_img and self.test_img
        """
        is_dup = False 
        for test_path in self.test_img:
            if test_path in self.train_img:
                print('There is a dupplication between train and test at:',
                      test_path)
                is_dup=True 
        if not is_dup: print('[Good] There is no overlap between train and test data')
    # def getBatch(self,batch_num,is_aug=False,is_train=True,is_cutmix=True,is_test=False):
    #     if is_train:
    #         img_list = self.train_img 
    #         mask_list = self.train_mask 
    #         dem_list  = self.train_dem 
    #         op = "train"
    #     else: 
    #         img_list = self.test_img 
    #         mask_list = self.test_mask 
    #         dem_list = self.test_dem 
    #         is_aug=False
    #         op = "val"
    #     ## handle last batch
    #     if (batch_num == (self.getNumBatch(op=op)[0]-1)) and (self.getNumBatch(op=op)[1] != 0):
    #         start_ind = batch_num*self.batch_size
    #         end_ind   = start_ind + self.getNumBatch(op=op)[1]
    #     else:
    #         start_ind = batch_num*self.batch_size
    #         end_ind   = (batch_num+1)*self.batch_size
    #     # Split into land slide and non land slide images 
    #     input_seq = None
    #     mask_list_128 = None
    #     n_images = 0
    #     SIZE=128
    #     for idx in range(start_ind,end_ind):
    #         n_images+=1
    #         # Get Image 
    #         # img = cv2.imread(img_list[idx])

    #         img = cv2.imread(img_list[idx])
    #         img = cv2.resize(img,(SIZE,SIZE),interpolation=cv2.INTER_NEAREST)
    #         # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #         img = img.astype(np.float32)
    #         img = addGray(img)
    #         img = addEdge(img)
    #         img = addBlur(img)
    #         img = addGradient(img)
    #         h,w,c = img.shape
    #         dem = cv2.imread(dem_list[idx],cv2.IMREAD_GRAYSCALE)
    #         dem = cv2.resize(dem,(w,h),interpolation=cv2.INTER_NEAREST)
    #         dem = dem.astype(np.float32)
    #         dem = np.expand_dims(dem,axis=-1)
    #         # print(img.shape,dem.shape)
    #         img = np.concatenate((img,dem),axis=-1)

    #         # Get masks
    #         mask = cv2.imread(mask_list[idx],cv2.IMREAD_GRAYSCALE)
    #         mask = cv2.resize(mask,(SIZE,SIZE),interpolation=cv2.INTER_NEAREST)
    #         # mask = np.where(mask>0,1.,0.)
    #         # mask = np.array([1]) if np.sum(mask)>0 else np.array([0]) 
    #         # Preprocess
    #         # Convert img
            
    #         img = np.expand_dims(img,axis=0) # 1xHxWx6
    #         # img_t = torch.convert_to_tensor(img,dtype=torch.float32)
    #         input_seq = img if input_seq is None else  np.concatenate((input_seq,img),axis=0) # BxHxWx6
    #         # del img # Reduce burden for CPU 

    #         # Convert mask
    #         mask = np.expand_dims(mask,axis=0) # 1x2
    #         # mask_t = torch.convert_to_tensor(mask,dtype=torch.float32)
    #         mask_list_128 = mask if mask_list_128 is None else np.concatenate((mask_list_128,mask),axis=0) #BxHxW
    #         # del mask # Reduce burden for CPU 
        
    #     if is_cutmix:
    #         non_land_slide_idxs=[]
    #         land_slide_idxs=[]
    #         for i in range(len(mask_list_128)):
    #             if np.sum(mask_list_128[i]) < 1.:
    #                 non_land_slide_idxs.append(i)
    #             else:  land_slide_idxs.append(i)
    #         if len(non_land_slide_idxs)>0 and len(land_slide_idxs)==0:
    #             print("The batch contains only non landslide image")
    #         else:
    #             for i in non_land_slide_idxs:
    #                 j = np.random.randint(0,len(land_slide_idxs))
    #                 j = land_slide_idxs[j]
    #                 input_seq[i],mask_list_128[i] = self.cutmix(
    #                     img_land_slide=input_seq[j],
    #                             img_non_land_slide=input_seq[i],
    #                             mask_land_slide=mask_list_128[j])
                    
    #     for i in range(len(mask_list_128)):
    #         if is_aug:
    #             input_seq[i],mask_list_128[i] = self.augmentateImage(image=input_seq[i],mask=mask_list_128[i])
    #             print('min:',np.max(input_seq[i],axis=(0,1)),'max:',np.min(input_seq[i],axis=(0,1)))
    #         if not is_test:    
    #             input_seq[i] = (input_seq[i]-np.min(input_seq[i],axis=(0,1)))/(np.max(input_seq[i],axis=(0,1))-np.min(input_seq[i],axis=(0,1))+1e-9)
    #     input_seq = torch.tensor(input_seq,dtype=torch.float32)
    #     input_seq = input_seq.permute(0,3,1,2) # BxCxWxH
    #     mask_list_128 = torch.tensor(mask_list_128,dtype=torch.float32)    
         
    #     # print(mask_list_128.shape)      
    #     # Swap axes to get (batch_size, num_classes, height, width)
    #     mask_list_128 = torch.unsqueeze(mask_list_128,dim=1) # BxCxWxH
    #     mask_list_256 = F.interpolate(mask_list_128, size=(256, 256), mode='nearest')
    #     mask_list_64 = F.interpolate(mask_list_128, size=(64, 64), mode='nearest')
    #     mask_list_128 = torch.squeeze(mask_list_128,dim=1)
    #     mask_list_256 =torch.squeeze(mask_list_256,dim=1)
    #     mask_list_64 = torch.squeeze(mask_list_64,dim=1)

    #     mask_list_128 = torch.where(mask_list_128>0,1,0).to(torch.int32)
    #     mask_list_256 = torch.where(mask_list_256>0,1,0).to(torch.int32)
    #     mask_list_64 = torch.where(mask_list_64>0,1,0).to(torch.int32)
    #     if not is_test:
    #         return input_seq,mask_list_256,mask_list_128,mask_list_64,n_images #,img_list,mask_list
    #     else:
    #         return input_seq,mask_list_256,mask_list_128,mask_list_64,n_images ,img_list[start_ind:end_ind],mask_list[start_ind:end_ind]

    
    def augmentateImage(self, image, mask):
        ## rotate both image and mask
        if np.random.uniform() >= 0.6:
            # Flip left-right
            #1. Shift x, y
            # shift_range=(np.random.randint(-30,30),np.random.randint(-30,30))
            image = self.flip_augmentataion(image)
            mask = self.flip_augmentataion(mask)
        # if np.random.uniform() >=0.7:
        #     #2. rotate image
        #     angle = np.random.randint(-90,90) 
        #     image = self.rotateImgMask(image,angle)
        #     mask = self.rotateImgMask(mask,angle)
        if np.random.uniform() >=0.8:  
            #3. hue shift
            hue_shift = np.random.randint(-10,10) 
            image[...,:3] = self.hue_augmentation(image[...,:3],hue_shift)
        # if np.random.uniform() >=0.7:    
        #     #4. shear 
        #     shear_angle =  np.random.randint(-30,30)
        #     image = self.shear_augmentation(image,shear_angle)
        #     mask = self.shear_augmentation(mask,shear_angle)
        if np.random.uniform()>=0.7:
            scale = np.random.randint(8,12)/10. # From 0.8 to 1.2 
            shift = np.random.randint(-30,30)
            image[...,:3] = self.expose_augmentation(image[...,:3],scale,shift)
        if np.random.uniform()>0.7:
            h,w = image.shape[0],image.shape[1]
            width = np.random.randint(low=w*4//5,high=w)
            height = np.random.randint(low=h*4//5,high=h)
            # print(image.shape,'h',h,'w',w,'width',width,'height',height)
            image,mask = self.random_crop_augmentation(image,mask,width=width,height=height)
        return image,mask
    
    def expose_augmentation(self,image:np.ndarray,scale:float,shift:float)->np.ndarray:
        """
        expose_augmentation(self,image:np.ndarray,scale:float,shift:float): change exposer of an image 
        """
        image = image*scale+shift 
        
        image = np.clip(image,0,255)
        image = image.astype(np.float32)
        return image 
    
    def flip_augmentataion(self,image):
        """
        Flip left right 
        """
        image = image[:,::-1,:]
        return image
    
    def shear_augmentation(self,image, shear_angle):
        """
        Apply shear transformation to a 2D array with multiple channels.
        
        Args:
        image (np.ndarray): Input 2D array (image with multiple channels).
        shear_angle (float): Shear angle in degrees.
        
        Returns:
        np.ndarray: Sheared image.
        """
        is_gray = False
        if len(image.shape)==3:
            # print(image.shape)
            # image = np.expand_dims(image,-1)
            height, width,channels = image.shape
        else:
            is_gray = True
            height,width = image.shape 
            image_zeros = np.zeros((height,width,3),dtype=np.uint8)
            image_zeros[:,:,0] = image 
            channels = 3
            image = image_zeros
        # Define shear angle range
        # shear_angle = np.random.uniform(-shear_range, shear_range)
        
        # Compute the center point of the image
        center_x = image.shape[1] / 2
        center_y = image.shape[0] / 2
        
        # Define the shear matrix
        shear_matrix = np.array([[1, np.tan(np.deg2rad(shear_angle)), 0],
                                [0, 1, 0],
                                [0, 0, 1]], dtype=np.float32)
        
        # Compute the translation matrix to move the center point back to the origin
        translate_matrix1 = np.array([[1, 0, -center_x],
                                    [0, 1, -center_y],
                                    [0, 0, 1]])
        
        # Compute the translation matrix to move the center point back to its original position
        translate_matrix2 = np.array([[1, 0, center_x],
                                    [0, 1, center_y],
                                    [0, 0, 1]])
        
        # Combine the shear and translation matrices
        affine_matrix = translate_matrix2 @ shear_matrix @ translate_matrix1
        
        # Extract the first two rows of the affine matrix to make it a 2x3 matrix
        affine_matrix_2x3 = affine_matrix[:2,:]
        
        # Apply shear transformation to each channel separately
        sheared_image = np.zeros_like(image)
        if not is_gray:
            for c in range(image.shape[2]):
                sheared_image[:,:,c] = cv2.warpAffine(image[:,:,c], affine_matrix_2x3, (image.shape[1], image.shape[0]))
            return sheared_image
        else:
            sheared_image[:,:,0] = cv2.warpAffine(image[:,:,0], affine_matrix_2x3, (image.shape[1], image.shape[0]))
            return sheared_image[:,:,0]
    
    def hue_augmentation(self,image, hue_shift):
        """
        Apply Hue augmentation to an image.
        
        Args:
        image (np.ndarray): Input image (H x W x C, where C is the number of channels).
        hue_shift (int or float): Amount of hue shift in degrees (-180 to 180).
        
        Returns:
        np.ndarray: Augmented image.
        """
        # Convert image to HSV color space
        hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Shift hue channel
        hsv_image[:, :, 0] = (hsv_image[:, :, 0].astype(int) + hue_shift) % 180
        
        # Convert image back to RGB color space
        augmented_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)
    
        return augmented_image

    def saturation_augmentation(self,image, hue_shift):
        """
        Apply Hue augmentation to an image.
        
        Args:
        image (np.ndarray): Input image (H x W x C, where C is the number of channels).
        hue_shift (int or float): Amount of hue shift in degrees (-180 to 180).
        
        Returns:
        np.ndarray: Augmented image.
        """
        # Convert image to HSV color space
        hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Shift hue channel
        hsv_image[:, :, 0] = (hsv_image[:, :, 0].astype(int) + hue_shift) % 180
        
        # Convert image back to RGB color space
        augmented_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)
    
        return augmented_image
    
    def shift_augmentation(self,image, shift_range):
        """
        Apply shift augmentation to an image.
        
        Args:
        image (np.ndarray): Input image.
        shift_range (tuple): Range of shift along x and y axes.
        
        Returns:
        np.ndarray: Shifted image.
        """
        # Randomly generate shift values within the specified range
        shift_x = shift_range[0]#np.random.randint(shift_range[0], shift_range[1])
        shift_y = shift_range[1]#np.random.randint(shift_range[0], shift_range[1])
        
        # Shift the image using numpy's roll function
        shifted_image = np.roll(image, shift_x, axis=1)
        shifted_image = np.roll(shifted_image, shift_y, axis=0)
        
        return shifted_image

    def rotateImgMask(self, one_image,angle):
        # 1,2,3 -> 90,180,270
        # rotated_image = np.rot90(one_image, angle)
        # rotated_mask  = np.rot90(one_mask, angle)
        # Rotate the 2D NumPy array by the specified angle
        rotated_image = ndimage.rotate(one_image, angle, reshape=False)
        # rotated_mask  = ndimage.rotate(one_mask, angle, reshape=False)
        return rotated_image

    def image_enhancement(self,image:np.ndarray):
        """
        image_enhancement(self,image): Enhance image quality 
        ### Argurments:
            image(np.ndarray): Image to be enhanced 
        ### Returns:
            enhanced image 
        """
        # Define a sharpening kernel
        sharpening_kernel = np.array([[-1, -1, -1],
                                    [-1,  9, -1],
                                    [-1, -1, -1]])

        # Apply the sharpening kernel to the image
        sharpened_image = cv2.filter2D(image, -1, sharpening_kernel)

        # Use histogram equalization 
        # Convert the image to HSV color space
        hsv_image = cv2.cvtColor(sharpened_image, cv2.COLOR_BGR2HSV)

        # Equalize the histogram of the V channel
        hsv_image[:, :, 2] = cv2.equalizeHist(hsv_image[:, :, 2])

        # Convert the image back to BGR color space
        equalized_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
        return equalized_image 

    def cutmix(self, img_non_land_slide:np.ndarray, 
               img_land_slide:np.ndarray,mask_land_slide:np.ndarray):
        """
            Args:
                Inputs:
                    img_non_land_slide(np.ndarray): Image to be mixed
                    mask_non_land_slide(np.ndarray): mask to be mixed
                    img_land_slide(np.ndarray): image with land slide
                    mask_land_slide(np.ndarray): mask with land slide 
                outputs:
                    images and mask with land slide
        """
        mask = np.expand_dims(mask_land_slide,axis=-1)
        mask = np.where(mask>0,.8,0.2)
        result = (1.-mask)*img_non_land_slide+mask*img_land_slide
        return result, mask_land_slide.copy()
    
    def cutmixPytorch(self, img_non_land_slide:torch.Tensor, 
               img_land_slide:torch.Tensor,mask_land_slide:torch.Tensor):
        """
            Args:
                Inputs:
                    img_non_land_slide(torch.Tensor): Image to be mixed
                    mask_non_land_slide(torch.Tensor): mask to be mixed
                    img_land_slide(torch.Tensor): image with land slide
                    mask_land_slide(torch.Tensor): mask with land slide 
                outputs:
                    images and mask with land slide
        """
        # mask = np.expand_dims(mask_land_slide,axis=-1)
        # mask = mask_land_slide.unsqueeze(dim=0) #CxWxH
        # print(mask.shape,img_non_land_slide.shape,img_land_slide.shape)
        mask = mask_land_slide.expand_as(img_non_land_slide)  # Expand to shape (C, H, W)
        mask = torch.where(mask>0,0.8,0.2)
        result = (1.-mask)*img_non_land_slide+mask*img_land_slide
        return result, mask_land_slide
 
    def Mosaic_Images(self,img1,img2):
        """
        Args:
        img_4: 4 input images
        Original images:
        |1,1|1,2| |2,1|2,2| 
        |1,3|1,4| |2,3|2,4| 
        swap 4 parts of 4 images to 4 images with index like this
        |1,1|2,2| |2,1|1,2| 
        |2,3|4,1|,|1,3|2,4|
        """
        shapes = img1.shape  #HxWx[C]
        h2,w2 = shapes[0]//2,shapes[1]//2
        part_11 = img1[:h2,:w2]
        part_12 = img1[:h2,w2:]
        part_13 = img1[h2:,:w2]
        part_14 = img1[h2:,w2:]

        part_21 = img2[:h2,:w2]
        part_22 = img2[:h2,w2:]
        part_23 = img2[h2:,:w2]
        part_24 = img2[h2:,w2:]

        img1_out_up = np.concatenate((part_11,part_22),axis=1)
        img1_out_down = np.concatenate((part_23,part_14),axis=1)
        img1_out = np.concatenate((img1_out_up,img1_out_down),axis=0)

        img2_out_up = np.concatenate((part_21,part_12),axis=1)
        img2_out_down = np.concatenate((part_13,part_24),axis=1)
        img2_out = np.concatenate((img2_out_up,img2_out_down),axis=0)

        return img1_out,img2_out 

    def Mosaic_Batch(self,seq_x:np.ndarray, 
                     seq_y_256:np.ndarray, seq_y_128:np.ndarray, 
                     seq_y_64:np.ndarray):
        """
        Args:
            seq_x: input image (Bx128x128xC)
            seq_y_256: output 256 (Bx256x256)
            seq_y_128: output 128 (Bx128x128)
            seq_y_64: output 64 (Bx64x64)
        Ouput: Mosaic versions of seq_x, seq_y_256, seq_y_128, seq_y_64, n_image
        """
        shapes = seq_x.shape #BxHxWxC 
        B = shapes[0]
        num_pairs = B//2 
        remain = B%2
        seq_x_out= None 
        seq_y_256_out = None 
        seq_y_128_out = None 
        seq_y_64_out = None 
        for i in range(num_pairs):
            im1,im2 = self.Mosaic_Images(seq_x[i*2],seq_x[i*2+1])
            y_256_1,y_256_2 = self.Mosaic_Images(seq_y_256[i*2],seq_y_256[i*2+1])
            y_128_1,y_128_2 = self.Mosaic_Images(seq_y_128[i*2],seq_y_128[i*2+1])
            y_64_1,y_64_2 = self.Mosaic_Images(seq_y_64[i*2],seq_y_64[i*2+1])
            im1 = np.expand_dims(im1,axis=0)
            im2 = np.expand_dims(im2,axis=0)
            y_256_1 = np.expand_dims(y_256_1,axis=0)
            y_256_2 = np.expand_dims(y_256_2,axis=0)
            y_128_1 = np.expand_dims(y_128_1,axis=0)
            y_128_2 = np.expand_dims(y_128_2,axis=0)
            y_64_1 = np.expand_dims(y_64_1,axis=0)
            y_64_2 = np.expand_dims(y_64_2,axis=0)
            if seq_x_out is None: 
                seq_x_out = np.concatenate((im1,im2),axis = 0)
                seq_y_256_out = np.concatenate((y_256_1,y_256_2),axis = 0)
                seq_y_128_out = np.concatenate((y_128_1,y_128_2),axis = 0)
                seq_y_64_out = np.concatenate((y_64_1,y_64_2),axis = 0)
            else:
                seq_x_out = np.concatenate((seq_x_out,im1,im2),axis = 0)
                seq_y_256_out = np.concatenate((seq_y_256_out,y_256_1,y_256_2),axis = 0)
                seq_y_128_out = np.concatenate((seq_y_128_out,y_128_1,y_128_2),axis = 0)
                seq_y_64_out = np.concatenate((seq_y_64_out,y_64_1,y_64_2),axis = 0)
        if remain==1:
            # last image in batch, just simply add to batch 
            seq_x_out = np.concatenate((seq_x_out,seq_x[-1]),axis = 0)
            seq_y_256_out = np.concatenate((seq_y_256_out,seq_y_256[-1]),axis = 0)
            seq_y_128_out = np.concatenate((seq_y_128_out,seq_y_128[-1]),axis = 0)
            seq_y_64_out = np.concatenate((seq_y_64_out,seq_y_64[-1]),axis = 0)
        return seq_x_out,seq_y_256_out,seq_y_128_out,seq_y_64_out 

    def mix(self,images,masks):
        """
            # To mix batch of images and mask together
            Args:
                Inputs:
                    image(4D torch.tensor): Images to be mixed B x W x H x C
                    mask(3 D torch.tensor): masks to be mixed  B x W x H
                outputs:
                    images and masks with multiple landslide
        """
        if np.random.uniform()>0.7:
            B,W,H,C = images.shape
            # idxs = np.random.shuffle(np.arange(B))
            indices = torch.range(start=0, limit=B, dtype=torch.int32)
            shuffled_indices = torch.random.shuffle(indices)
            images = 0.5*(images + torch.gather(images, shuffled_indices))
            
            masks = masks+torch.gather(masks,shuffled_indices)
            masks = torch.clip_by_value(masks,clip_value_min=0,clip_value_max=1.)
            # print(images.shape,masks.shape)
        return images,masks
    
    def random_crop_augmentation(self,img, mask, width, height):
        assert img.shape[0] >= height
        assert img.shape[1] >= width
        assert img.shape[0] == mask.shape[0]
        assert img.shape[1] == mask.shape[1]
        h,w = img.shape[0],img.shape[1]
        x = np.random.randint(0, img.shape[1] - width)
        y = np.random.randint(0, img.shape[0] - height)
        img = img[y:y+height, x:x+width]
        mask = mask[y:y+height, x:x+width]
        img = cv2.resize(img,(w,h),interpolation=cv2.INTER_NEAREST)
        mask = cv2.resize(mask,(w,h),interpolation=cv2.INTER_NEAREST)
        # print(img.shape,mask.shape,w,h)
        return img, mask

if __name__ == '__main__':
    ### TEST GET BATCH ###
    # def __init__(self, data_dir, batch_size, test_ratio=0.2, test_fold=1, band_opt=1, is_multi_resolution=False):
    generator = DataGenerator(data_dir='../Bijie-landslide-dataset',
                              batch_size=8,test_ratio=0.3,stored_folder='../results/Bijie_RAU_Store_Train_Test_backup')
    print(generator.n_img_train,generator.n_img_val)
    for i in range(10):
        generator.shuffle_train()
    input_seq,mask_list_256,mask_list_128,mask_list_64,n_images,img_list,mask_list,hog_img,edges = generator.getBatch(10,is_aug=True,is_train=True,is_test=True,is_cutmix=True)
    img = input_seq.permute(0,2,3,1)[1].cpu().numpy() 
    mask= mask_list_128[1].cpu().numpy()
    print(img_list[1],img.shape,mask_list[1],mask.shape)


    mean=np.array([0.3028171,0.37409051,0.31993082])
    std=np.array([0.0922879,0.08401628,0.09078308]) 
    img = (img*std+mean)*255.
    img = img.astype(np.uint8)
    draw_image(img[...,:3],"image.png")
    draw_image(mask*255,'mask128.png')
    # draw_image(hog_img,'hog_img.png')
    # draw_image(edges*255.,'edges.png')
    mask256=mask_list_256[1].cpu().numpy()
    draw_image(mask256*255,'mask256.png')
    mask64=mask_list_64[1].cpu().numpy()
    draw_image(mask64*255,'mask64.png')
    print('n_train:',generator.getNumImg('train'),'\tn_val:',generator.getNumImg('val'))



