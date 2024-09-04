import os
import cv2
import numpy as np
import torch
from util2 import *
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.transforms import functional as VF
import h5py

class DataGenerator(object):
    def __init__(self, data_dir='../LandSlide4Sense_dataset', batch_size=12, stored_folder="../results/LandSliding_LandSlide4Sense",test_ratio=0.2):
        self.data_dir = data_dir 
        self.batch_size = batch_size 
        self.stored_folder = stored_folder
        self.test_ratio=test_ratio 
        self.train_test_split() 
       
        self.train_transforms = T.Compose([
            # T.ColorJitter(brightness=0.2),
            # AdjustBrightness(brightness_factor=1.2),  # Adjust brightness by a factor of 1.2
            # AdjustSaturation(saturation_factor=1.5),  # Adjust saturation by a factor of 1.5
            # AdjustSharpness(sharpness_factor=2.0),  # Adjust sharpness by a factor of 2.0
            # AddNoise(mean=0.0, std=0.05),  # Add Gaussian noise with mean=0 and std=0.05
            Numpy2Tensor(),  # Convert the image to a tensor range [0,1]
            # T.Normalize(mean=[0.29858148,0.39586081,0.09152383] , 
            #             std=[0.02363239,0.03083374,0.0127833 ])  # Normalize the image
        ])

        self.test_transforms = T.Compose([
            # T.ColorJitter(brightness=0.2),
            # AdjustBrightness(brightness_factor=1.2),  # Adjust brightness by a factor of 1.2
            # AdjustSaturation(saturation_factor=1.5),  # Adjust saturation by a factor of 1.5
            # AdjustSharpness(sharpness_factor=2.0),  # Adjust sharpness by a factor of 2.0
            # AddNoise(mean=0.0, std=0.05),  # Add Gaussian noise with mean=0 and std=0.05
            Numpy2Tensor(),  # Convert the image to a tensor range [0,1]
            # T.Normalize(mean=[0.29858148,0.39586081,0.09152383] , 
            #             std=[0.02363239,0.03083374,0.0127833 ])  # Normalize the image
        ])

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('Device Info:',self.device)

    def train_test_split(self):
        """
        Read data from training folder:
        ### Returns: 
            None but loading data into self.train_data, self.test_data, and
        self.val_data (prefecth data into RAM)
        """ 
        data_path = os.path.join(self.data_dir,'train')
        img_paths = os.path.join(data_path,'img')
        label_paths = os.path.join(data_path,'mask')
        
        ############ Read all data path into data############################
        self.data=list() 
        landslide_idxs = [] 
        nonslide_idxs = []
        idxs = [] 
        for idx,img_name in enumerate(os.listdir(img_paths)):
            img_full_path = os.path.join(img_paths,img_name)
            label_full_path = os.path.join(label_paths,img_name.replace('image','mask'))
            self.data.append((img_full_path,label_full_path))
            # add into index # 
            idxs.append(idx)
            mask = self._read_image(label_full_path)
            if np.sum(mask)>0: landslide_idxs.append(idx)
            else:nonslide_idxs.append(idx) 
        #####################################################################

        ########## Create random index for training and testing #############
        n_landslide = len(landslide_idxs)
        n_nonslide = len(nonslide_idxs)
        print('num of landslide:',n_landslide,';num of nonslide:',n_nonslide)
        random.shuffle(landslide_idxs)
        random.shuffle(nonslide_idxs)
        test_num = int(n_landslide*self.test_ratio) 
        self.test_idxs = landslide_idxs[:test_num]
        self.train_idxs=landslide_idxs[test_num:]
        test_num = int(n_nonslide*self.test_ratio)
        self.test_idxs = self.test_idxs+nonslide_idxs[:test_num]
        self.train_idxs = self.train_idxs+nonslide_idxs[test_num:]
        self.val_idxs = None # NOT USE
        
        self.n_train = len(self.train_idxs)
        self.n_test = len(self.test_idxs)
        self.n_val = 0 # NOT USE   

        ##################################################################### 
        ############### Check integrity of train and test ###################
        n_overlap = 0 
        for i in self.test_idxs:
            if i in self.train_idxs: n_overlap+=1 
        if n_overlap==0:print('[Good] No overlap between train and test')
        else: print(f'[Bad] there [is/are] {n_overlap} indexes between train and test')
        #####################################################################
        
    def _read_image(self,img_path:str)->np.ndarray:
        """
        Read image from file_path 
        ### Arguments:
            img_path(str): string to file 
        ### Returns:
            image 
        """
        with h5py.File(img_path) as f_img:
            # Access the dataset 
            # print(f.keys())
            image = f_img[list(f_img.keys())[0]]
            image = np.asarray(image) # (128,128,14)
        return image
    
    def read_data(self,idx:int):
        """
        Read image and label given index
        ### Arguments:
            idx(int): index of image and label in self.data 
        ### Returns:
            image, label
        """
        img_path,label_path = self.data[idx]
        
        img = self._read_image(img_path)
        label=self._read_image(label_path)
        return img,label
        
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

    def _cutmix(self,idxs:list,img:torch.tensor,label:torch.tensor,prob=0.7):
        """
        Cut and mix together to increase the view of the batch 
        ### Arguments:
            idxs(list): index list of train or test data which will be used to mixed with img and label
            img(torch.tensor): An image in batch 
            label(torch.tensor): A label in batch 
            prob: probability to perform cut mix
        ### Returns:
            img and label that are cut and mix together 
        """
        rnd_idxs = np.copy(idxs)
        np.random.shuffle(rnd_idxs)
        if np.random.uniform() > prob: #mixing data 4 time to increase num of landslide region !!!
            print('Run cut mix here')
            for i in range(4):
                j = rnd_idxs[i]
                img2,label2 = self.read_data(idx=j)
                # while np.sum(label2)==0: # Make sure that there is landslide image to be mixed
                #     np.random.shuffle(rnd_idxs)
                #     j = rnd_idxs[i]
                #     img2,label2 = self.read_data(idx=j)
                # print('img cut mix:',name2)
                img2 = self.train_transforms(img=img2) 
                label2 = np.array(label2)
                label2 = torch.tensor(label2,dtype=torch.float32)
                # label = label.unsqueeze(0)
                mask1 = torch.clamp(label,min=0,max=1)
                mask2 = torch.clamp(label2,min=0,max=1)
                intersect = mask2*mask1 
                # exclusion = torch.clamp(mask1+mask2-2*intersect,min=0,max=1)
                mask1_only = torch.clamp(mask1-intersect,min=0,max=1) 
                mask2_only = torch.clamp(mask2-intersect,min=0,max=1)
                background_mask = (1-(mask1+mask2-intersect))
                # Intersect region of landslide will be treated as 50% for img1, 50% for img2
                # Other regions will be treated as 80% for img1, 20% for img2 
                img=0.5*(img+img2)*intersect+(0.8*img+0.2*img2)*(mask1_only+background_mask)+(0.2*img+0.8*img2)*mask2_only
                label = torch.clamp(label+label2,min=0,max=1)
            # not mix with label of imgs, maybe we need to do it
        return img,label

    def morph_transforms(self,idxs:list,img:torch.tensor,label:torch.tensor):
        """
        Performce morphology transformations including cutmix, rotate, and shift
        ### Arguments:
            idxs(list): index list of train or test data which will be used to mixed with img and label
            img(torch.tensor): CxWxH image to be augumented 
            label(torch.tensor):WxH image to be augmented
        """
        img,label = self._cutmix(idxs=idxs,img=img,label=label)
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
        data = self.data
        transforms = None
        if mode==0: #train
            # data = self.train_data
            idxs=self.train_idxs
            transforms = self.train_transforms
        elif mode==1: #val
            # data = self.val_data
            idxs=self.val_idxs
            transforms = self.test_transforms
        else: #test 
            # data = self.test_data
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
            img,label=self.read_data(idxs[i])
            label = torch.tensor(np.array(label))
            label = torch.clamp(label,min=0,max=1).unsqueeze(0)

            if is_aug:
                img = transforms(img)
                if mode==0:
                    # apply morphology transforms for training only
                    img,label= self.morph_transforms(idxs=idxs,img=img,label=label)
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
        random.shuffle(self.train_idxs)
        # np.random.shuffle(self.val_idxs)
        # np.random.shuffle(self.test_idxs)
def main():
    print('>>>>>>>>>>>>>>>>>1-TESTING PREFECT<<<<<<<<<<<<<<<<<<<<<<<')
    generator = DataGenerator()
    input('Wait here until enter any key ... please check RAM usage to make sure there is no overload problem')
    print('>>>>>>>>>>>>>>>>>>DONE 1<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    print('>>>>>>>>>>>>>>>>>>2- TESTING GET BATCH<<<<<<<<<<<<<<<<<<<<<<<')
    input_128,mask_list_64,mask_list_128,mask_list_256=generator.getBatch(batch_id=0,mode=0)
    print('input shape:',input_128.shape,'output shape:',mask_list_128.shape) 
    i = 0 
    img = np.array(input_128[i].permute(1,2,0)[...,1:4]) # Band 2: B, Band 3: G, Band 4: R
    mask_128=np.array(mask_list_128[i].squeeze(0))
    mask_64 =np.array(mask_list_64[i].squeeze(0)) 
    mask_256=np.array(mask_list_256[i].squeeze(0)) 
    # Recover image from normalization
    img = (img)*255
    img = img.astype(np.uint8)
    img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    img_seg = draw_segment_on_image(img,mask_128)
    draw_image(img,'image.png')
    draw_image(img_seg,'image_segment.png')
    draw_image(mask_128*255,'mask128.png')
    draw_image(mask_64*255,'mask64.png')
    print('>>>>>>>>>>>>>>>>>>DONE 2<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    # print('>>>>>>>>>>>>>>>>>>3- Testing image<<<<<<<<<<<<<<<<<<<<<<')
    # name,img,label = generator.train_data[0]
    # label = np.array(label)*255
    # label = label.astype(np.uint8)
    # print('Name of image:',name)
    # img = np.array(img)
    # img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    
    # # print(label)
    # draw_image(img,'image_original.png')
    # draw_image(label,'label_original.png')
    # img_seg = draw_segment_on_image(img,label)
    # draw_image(img_seg,'image_original_segment.png')
    # print('>>>>>>>>>>>>>>>>>>DONE 3<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')

if __name__=="__main__":
    main()
