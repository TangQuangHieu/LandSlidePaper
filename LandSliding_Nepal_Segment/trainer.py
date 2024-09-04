from util2 import remove_small_region_by_contour
from RAU_Net import * 
from Nepal_data_generator import DataGenerator
from losses import *
from metrics import * 
from loggers import Logger
from tqdm import tqdm 
import torch.optim.lr_scheduler as lr_scheduler
class Trainer:
    def __init__(self,generator:DataGenerator,model:RauNet,epoch:int,path:str):
        self.generator = generator 
        self.model = model 
        self.epochs = epoch 
        self.criterions = []
        # self.criterions.append(IOULoss())
        self.device =  'cuda' if torch.cuda.is_available() else 'cpu'
        alpha = torch.tensor([1., 5.5], dtype=torch.float32).to(self.device)  # Example class weights
        self.criterions=CombinedLoss(alpha=alpha)
        self.class_criter=nn.BCELoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(),lr=0.003,momentum=0.99)
        self.scheduler = lr_scheduler.ExponentialLR(optimizer=self.optimizer,gamma=0.9)
        self.path = path 
        self.logger = Logger(path=path)
        self.best_f1 = 0.
        
        self.model.to(self.device)

    def backward(self,X_train,y_256,y_128,y_64):
        """
        Calulate loss 
        """
        y_preds = self.model(X_train)
        loss_256 = self.criterions(y_preds[0],y_256)
        loss = self.criterions(y_preds[1],y_128)
        loss_64 = self.criterions(y_preds[2],y_64)
        y_class = torch.where(torch.sum(y_128,dim=(1,2))>0,1.,0.)
        c_loss = self.class_criter(y_preds[3].squeeze(dim=1),y_class)
        loss = 0.65*loss_256+0.3*loss+0.05*loss_64+c_loss
        loss.backward()
        return loss.item() 

    def step(self): 
        self.optimizer.step()
        # self.scheduler.step() 

    def train(self):
        """
        Training data 
        """
        for epoch in range(self.epochs):
            num_batch,num_img_last_batch  = self.generator.getNumBatch()
            loss = 0.
            with tqdm(total=num_batch, desc=f"Training", unit="Mini Batch") as pbar:
                
                for batch_id in range(num_batch):   
                    self.model.train()  
                    if epoch<70:
                        X_train,y_64,y_128,y_256 = self.generator.getBatch(batch_id=batch_id,mode=0,is_aug=True)
                    else:
                        X_train,y_64,y_128,y_256 = self.generator.getBatch(batch_id=batch_id,mode=0,is_aug=False)
                    X_train = X_train.to(self.device)
                    y_256 = y_256.to(self.device)
                    y_128 = y_128.to(self.device)
                    y_64  = y_64.to(self.device)
                    self.optimizer.zero_grad()
                    loss+=self.backward(X_train=X_train,
                                y_256=y_256,
                                y_128=y_128,
                                y_64=y_64) # Accumulate gradient
                    pbar.update(1)
                    pbar.set_description(f"Batch ID -Total Batch {batch_id+1}/{num_batch}")
                    self.step() # Update weights
                    
            train_info = f'Train >>>> Epoch:{epoch}|Loss:{loss/num_batch:.4f}'
            self.logger.write(train_info)
            # Testing
            if epoch>0:
                self.test(epoch=epoch)
            self.generator.shuffle_train() 
            if epoch%20==0 and epoch>0:
                self.scheduler.step() 
        # Close log file
        self.logger.close()
            
    
    def test(self,epoch):
        self.model.eval() 
        test_num,last_batch_size = self.generator.getNumBatch(mode=2)
        # print('test_num',test_num)
        test_tp = 0.
        test_fp = 0.
        test_fn = 0.
        test_acc = 0.
        with torch.no_grad():
            for batch_id in range(test_num):
                X_test,y_64,y_128,y_256 = self.generator.getBatch(batch_id=batch_id,mode=2,is_aug=False)
                # print('X_test.shape',X_test.shape,y_64.shape)
                X_test = X_test.to(self.device)
                y_256 = y_256.to(self.device)
                y_128 = y_128.to(self.device)
                y_64  = y_64.to(self.device)
                y_preds = self.model(X_test)
                # print('sum y_preds',torch.sum(torch.argmax(y_preds[1],dim=1)))
                # print('sum y_128',torch.sum(y_128))
                tp,fp,fn,acc = calculate_metrics(y_true=y_256,
                                y_pred=y_preds[0],y_pred_class=y_preds[3])
                test_tp += tp 
                test_fp += fp 
                test_fn += fn 
                test_acc+= acc 
            print('test_tp',test_tp,'test_fp',test_fp,'test_fn',test_fn,'test_acc',test_acc)
            f1,precision,recall = calculate_f1(tp=test_tp,fp=test_fp,fn=test_fn)
            SIZE=256
            test_acc /= ((SIZE**2)*self.generator.getNumImg(mode=2))
            test_info = f'Test >>>> Epoch:{epoch}|Accuracy:{test_acc:.4f}|F1:{f1:.4f}|Precision:{precision:.4f}|Recall:{recall:.4f}'
            self.logger.write(test_info)
            if f1>self.best_f1:
                self.best_f1 = f1
                torch.save(self.model,os.path.join(self.path,'best_model.pth'))
                self.logger.write(f'Saved best model:f1:{f1:.4f}')

    def analyse(self,):
        """
        use after training for analyzing result
        """         
        self.model.eval() 
        test_num,last_batch_size = self.generator.getNumBatch(mode=3)
        segment_metrics = np.zeros(7) # tp,fp,fn,acc,f1,precision,recall
        classify_metrics=np.zeros(47) # tp,fp,fn,acc,f1,precision,recall
        print(f'>>>>Testing>>>>')
        # self.generator.checkTrainTestDupplicate()
        with torch.no_grad():
            for batch_id in range(test_num):
                X_test,y_64,y_128,y_256 = self.generator.getBatch(batch_id=batch_id,mode=2,is_aug=False)
                X_test = X_test.to(self.device)
                y_256 = y_256.to(self.device)
                y_128 = y_128.to(self.device)
                y_64  = y_64.to(self.device)
                y_preds = self.model(X_test)
                
                # y_pred = torch.tensor(y_pred).unsqueeze(dim=0)
                tp,fp,fn,acc = calculate_metrics_with_postprocess(y_true=y_256,
                                y_pred=y_preds[0])
                segment_metrics[0]+=tp
                segment_metrics[1]+=fp
                segment_metrics[2]+=fn
                segment_metrics[3]+=acc

                tp,fp,fn,acc= calculate_metrics_classifier(y_true=y_256,y_pred=y_preds[3])
                classify_metrics[0]+=tp
                classify_metrics[1]+=fp
                classify_metrics[2]+=fn
                classify_metrics[3]+=acc

            # print('test_tp',test_tp,'test_fp',test_fp,'test_fn',test_fn,'test_acc',test_acc)
            f1,pre,rec = calculate_f1(tp=segment_metrics[0],
                                               fp=segment_metrics[1],
                                               fn=segment_metrics[2])
            segment_metrics[4]=f1
            segment_metrics[5]=pre
            segment_metrics[6]=rec

            f1,pre,rec = calculate_f1(tp=classify_metrics[0],
                                               fp=classify_metrics[1],
                                               fn=classify_metrics[2])
            classify_metrics[4]=f1
            classify_metrics[5]=pre
            classify_metrics[6]=rec
            
            SIZE=256
            n_test = self.generator.getNumImg(mode=2)
            n_train = self.generator.getNumImg(mode=0)
            segment_metrics[3] /= ((SIZE**2)*n_test)
            classify_metrics[3] /= n_test
            info = f'''>>>>>>>>>REPORT INFO<<<<<<<<<<<
            num of test:{n_test} vs num of train:{n_train}
            1. Segment Report:
            True Positive:{segment_metrics[0]:.4f}
            False Positive:{segment_metrics[1]:.4f}
            False Negative:{segment_metrics[2]:.4f}
            Accuracy:{segment_metrics[3]:.4f}
            F1 Score:{segment_metrics[4]:.4f}
            Precision:{segment_metrics[5]:.4f}
            Recall:{segment_metrics[6]:.4f}
            2. Classification Report:
            True Positive:{classify_metrics[0]:.4f}
            False Positive:{classify_metrics[1]:.4f}
            False Negative:{classify_metrics[2]:.4f}
            Accuracy:{classify_metrics[3]:.4f}
            F1 Score:{classify_metrics[4]:.4f}
            Precision:{classify_metrics[5]:.4f}
            Recall:{classify_metrics[6]:.4f} 
            >>>>>>DONE<<<<<<<
'''
        print(info)
        print('>>>>>>>>>>...Predicting...<<<<<<<<<<<<<<<<<')
        self.predict()
        print('>>>>>>>>>>>>>>>Done<<<<<<<<<<<<<<<<<<<<<')
            
    def predict(self,saved_folder="../results/LandSliding_Nepal_Segment/test_imgs"):
        """
        Predict and save data for image prediction 
        ### Arguments:
            saved_folder(str): folder to store data
        ### Returns:
            Predict and save data 
        """
        self.model.eval()
        if not os.path.exists(saved_folder):
            os.mkdir(saved_folder)
        n_test = len(self.generator.test_data)
        with tqdm(total=n_test, desc=f"Prediciting", unit="Image") as pbar:
            for key,values in self.generator.test_data.items():
                img_name,img,label = values
                img_gpu = self.generator.test_transforms(img)
                label = np.array(label)
                # label = torch.tensor(label,dtype=torch.float32)
                # label = torch.clamp(label,min=0,max=1)
                img_gpu = img_gpu.to(self.device).unsqueeze(0)
                with torch.no_grad():
                    y_preds = self.model(img_gpu)
                    # y_pred = torch.squeeze(y_preds[0],dim=0)
                    # y_preds = y_preds.cpu().to_numpy()
                # Draw image and store in saved_folder, convert to numpy to use opencv 
                y_pred = torch.argmax(y_preds[0],dim=1).squeeze(dim=0).cpu().numpy().astype(np.uint8)
                y_pred = remove_small_region_by_contour(y_pred)
                img = cv2.resize(cv2.cvtColor(np.array(img),cv2.COLOR_RGB2BGR),dsize=(256,256))
                label = cv2.resize(label,dsize=(256,256))
                # img = np.where(label==1,(0, 200, 200),img)
                img_draw = np.zeros_like(img)
                img_draw[...,0]=label*255 
                img_draw[...,1]=y_pred*255 
                img_name = img_name.replace('tiff','png')
                img_name = os.path.join(saved_folder,img_name)
                cv2.imwrite(img_name,img)
                cv2.waitKey(10)
                cv2.imwrite(img_name.replace('.png','_masks.png'),img_draw)
                cv2.waitKey(10)
                pbar.update(1)
                pbar.set_description(f"Num Img -Total Img {key}/{n_test}")
            # Label is white, predict is yellow
             


                

        
            
                    

                


            