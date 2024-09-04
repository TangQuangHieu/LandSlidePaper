from RAU_Net import * 
from Bijie_data_generator import DataGenerator
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
        alpha = torch.tensor([1., 3.5], dtype=torch.float32).to(self.device)  # Example class weights
        self.criterions=CombinedLoss(alpha=alpha)
        self.class_criter=nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=0.001)
        # self.scheduler = lr_scheduler.StepLR(optimizer=self.optimizer,step_size=15,gamma=0.1)
        self.path = path 
        self.logger = Logger(path=path)
        self.best_f1 = 0.
        
        self.model.to(self.device)

    def backward(self,X_train,y_256,y_128,y_64):
        """
        Calulate loss 
        """
        y_preds = self.model(X_train)
        # print('y_128',y_preds[1],'y_128.shape',y_preds[1].shape)
        loss_256 = self.criterions(y_preds[0],y_256)
        
        # loss_256 += self.criterions[1](y_256,y_preds[0])

        loss = self.criterions(y_preds[1],y_128)
        # print('IOULoss',loss)
        # focal_loss= self.criterions[1](y_128,y_preds[1])
        
        # loss+=focal_loss
        loss_64 = self.criterions(y_preds[2],y_64)
        # loss_64+= self.criterions[1](y_64,y_preds[2])
        # print('loss128:',loss,'loss64:',loss_64,'loss_256:',loss_256)
        y_class = torch.where(torch.sum(y_128,dim=(1,2))>0,1.,0.)
        c_loss = self.class_criter(y_preds[3].squeeze(dim=1),y_class)

        loss = 0.25*loss_256+0.5*loss+0.25*loss_64+c_loss
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
            
            
            num_batch,last_batch_size = self.generator.getNumBatch()
            loss = 0.
            with tqdm(total=num_batch, desc=f"Training", unit="Mini Batch") as pbar:
                self.optimizer.zero_grad()
                for batch_id in range(num_batch):   
                    self.model.train()  
                    
                    if epoch<60:
                        X_train,y_256,y_128,y_64,n = self.generator.getBatch(batch_num=batch_id,is_aug=True,
                                                                is_train=True,is_cutmix=True,)
                    else:
                        X_train,y_256,y_128,y_64,n = self.generator.getBatch(batch_num=batch_id,is_aug=False,
                                                                            is_train=True,is_cutmix=False,)
                    X_train = X_train.to(self.device)
                    y_256 = y_256.to(self.device)
                    y_128 = y_128.to(self.device)
                    y_64  = y_64.to(self.device)
                    
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
        
        # Close log file
        self.logger.close()
            
    
    def test(self,epoch):
        self.model.eval() 
        test_num,last_batch_size = self.generator.getNumBatch('test')
        test_tp = 0.
        test_fp = 0.
        test_fn = 0.
        test_acc = 0.
        with torch.no_grad():
            for batch_id in range(test_num):
                X_test,y_256,y_128,y_64,n = self.generator.getBatch(batch_num=batch_id,
                                                                is_aug=False,is_train=False,
                                                                is_cutmix=False)
                X_test = X_test.to(self.device)
                y_256 = y_256.to(self.device)
                y_128 = y_128.to(self.device)
                y_64  = y_64.to(self.device)
                y_preds = self.model(X_test)
                tp,fp,fn,acc = calculate_metrics(y_true=y_128,
                                y_pred=y_preds[1],y_pred_class=y_preds[3])
                test_tp += tp 
                test_fp += fp 
                test_fn += fn 
                test_acc+= acc 
            print('test_tp',test_tp,'test_fp',test_fp,'test_fn',test_fn,'test_acc',test_acc)
            f1,precision,recall = calculate_f1(tp=test_tp,fp=test_fp,fn=test_fn)
            SIZE=128
            test_acc /= ((SIZE**2)*self.generator.getNumImg('val'))
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
        test_num,last_batch_size = self.generator.getNumBatch('test')
        segment_metrics = np.zeros(7) # tp,fp,fn,acc,f1,precision,recall
        classify_metrics=np.zeros(47) # tp,fp,fn,acc,f1,precision,recall
        print(f'>>>>Testing>>>>')
        self.generator.checkTrainTestDupplicate()
        with torch.no_grad():
            for batch_id in range(test_num):
                X_test,y_256,y_128,y_64,n = self.generator.getBatch(batch_num=batch_id,
                                                                is_aug=False,is_train=False,
                                                                is_cutmix=False)
                X_test = X_test.to(self.device)
                y_256 = y_256.to(self.device)
                y_128 = y_128.to(self.device)
                y_64  = y_64.to(self.device)
                y_preds = self.model(X_test)
                tp,fp,fn,acc = calculate_metrics(y_true=y_128,
                                y_pred=y_preds[1],y_pred_class=y_preds[3])
                segment_metrics[0]+=tp
                segment_metrics[1]+=fp
                segment_metrics[2]+=fn
                segment_metrics[3]+=acc

                tp,fp,fn,acc= calculate_metrics_classifier(y_true=y_128,y_pred=y_preds[3])
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
            
            SIZE=128
            n_test = self.generator.getNumImg('val')
            n_train = self.generator.getNumImg()
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
        self.predict()

               
    def predict(self,saved_folder="../results/LandSliding_Bijie_Segment_30_70/test_imgs"):
        """
        use after training for analyzing result
        """         
        self.model.eval() 
        print(f'>>>>>>>....Predicting.....<<<<<<<')
        # self.generator.checkTrainTestDupplicate()
        if not os.path.exists(saved_folder):
            os.mkdir(saved_folder)
        n_batch_test,_ = self.generator.getNumBatch(op='val')
        with tqdm(total=n_batch_test, desc=f"Prediciting", unit="Image") as pbar:
            for batch_id in range(n_batch_test):
                X_test,y_256,y_128,y_64,n = self.generator.getBatch(batch_num=batch_id,
                                                                is_aug=False,is_train=False,
                                                                is_cutmix=False)
                X_test = X_test.to(self.device)
                y_256 = y_256.to(self.device)
                y_128 = y_128.to(self.device)
                y_64  = y_64.to(self.device)
                with torch.no_grad():
                    y_preds = self.model(X_test) 
                    i=0
                    for y_pred,y_class in zip(y_preds[1],y_preds[3]):
                        idx = batch_id * self.generator.batch_size + i
                        img_name = self.generator.test_img[idx].split('/')[-1].replace('.png','_img.png')
                        label_name=self.generator.test_mask[idx].split('/')[-1].replace('.png','_mask.png')
                        # print('img_name',self.generator.test_img[idx],'label_name',self.generator.test_mask[idx])
                        img = cv2.imread(self.generator.test_img[idx])
                        img = cv2.resize(img,(128,128))
                        label=cv2.imread(self.generator.test_mask[idx],cv2.IMREAD_GRAYSCALE)
                        label=cv2.resize(label,(128,128))
                        # print('y_pred.shape',y_pred.shape)
                        y_pred = torch.argmax(y_pred,dim=0).squeeze(dim=0).cpu().numpy().astype(np.uint8) #WxH  
                        if y_class<0.5:
                            y_pred = np.zeros_like(y_pred)
                        # print('y_pred.shape',y_pred.shape)
                        output = np.zeros_like(img)
                        # print('output.shape',output.shape,'label.shape',label.shape,'y_pred.shape',y_pred.shape)
                        # label = label[...,np.newaxis]
                        # y_pred = y_pred[...,np.newaxis]
                        # print('output.shape',output.shape,'label.shape',label.shape,'y_pred.shape',y_pred.shape)
                        output[...,0]=label # Blue 
                        output[...,1]=y_pred*255# Green
                        # output*=255
                        out_img_path= os.path.join(saved_folder,img_name)
                        out_label_path=os.path.join(saved_folder,label_name)
                        cv2.imwrite(out_img_path,img=img) 
                        cv2.waitKey(10)
                        cv2.imwrite(out_label_path,img=output)  
                        cv2.waitKey(10)
                        i+=1
                pbar.update(1)
                pbar.set_description(f"Num Batch -Total Batch {batch_id+1}/{n_batch_test}")
                # break
        print(f'>>>>>>>DONE<<<<<<<')
     

        
            
                    

                


            