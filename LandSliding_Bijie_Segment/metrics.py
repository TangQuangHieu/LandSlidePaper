
import torch 
import cv2 
import numpy as np
def calculate_metrics_classifier(y_true,y_pred):
    """
    Calculate metric for binary classfication
    ### Arguments:
        y_true (BxWxH): Ground truth, need to be converted to Bx1
        y_pred(Bx1): Predictions in probabilities
    ### Returns:
        tp,fp,fn,acc for testing
    """
    y_true = torch.where(torch.sum(y_true,dim=(1,2))>0,1.,0.)
    y_pred = torch.where(y_pred<0.5,0,1).squeeze(dim=-1)
    # print(y_true.shape,y_pred.shape)
    tp = torch.sum((y_pred == 1) & (y_true == 1)).item()
    fp = torch.sum((y_pred == 1) & (y_true == 0)).item()
    fn = torch.sum((y_pred == 0) & (y_true == 1)).item()
    acc = torch.sum(y_pred == y_true).item()
    return tp,fp,fn,acc
def calculate_metrics(y_true,y_pred,y_pred_class=None):
    """
    calculate_metrics(y_true,y_pred): Calculate metrics 
    tp,fp,fn,acc
    ### Arguments:
        y_true (BxWxH): ground truth
        y_pred (BxCxWxH):Prediction
    ### Returns:
        tp,fp,fn,acc
    """
    # Calculate TP, FP, FN, and accuracy
    y_pred=torch.argmax(y_pred,dim=1) #Bx1xWxH
    y_pred=y_pred.squeeze(dim=1) #BxWxH
    for i,y_c in enumerate(y_pred_class):
        if y_c<0.5: y_pred[i]=torch.zeros_like(y_pred[i])
    # y_pred= torch.where(y_pred_class<0.5,torch.zeros_like(y_pred),y_pred)
    # print(y_pred.shape,y_pred,y_true.shape,y_true)
    tp = torch.sum((y_pred == 1) & (y_true == 1)).item()
    fp = torch.sum((y_pred == 1) & (y_true == 0)).item()
    fn = torch.sum((y_pred == 0) & (y_true == 1)).item()
    acc = torch.sum(y_pred == y_true).item()
    return tp,fp,fn,acc
def calculate_metrics_test(y_true:torch.tensor,y_pred:torch.tensor,thresh = 10):
    """
    calculate_metrics_test(y_true:torch.tensor,y_pred:torch.tensor): 
    Test output with post process method ( erase small area of segmentation map using finding contour)
    ### Arguments: 
        y_true(torch.tensor): groundtruth tensor (BxWxH)
        y_pred(torch.tensor): output of model(BxCxWxH)
    ### Returns:
        true positive, false positive, false negative, accuracy 
    """

    y_true = y_true.cpu().numpy() 
    y_pred=torch.argmax(y_pred,dim=1) #Bx1xWxH
    y_pred=y_pred.squeeze(dim=1) #BxWxH
    y_pred = y_pred.cpu().numpy()
    tp = fp = fn = acc = 0.
    for y in y_pred:
        # print(y.shape)
        # Find contours
        # y = np.expand_dims(y,axis=-1)
        y = y.astype('uint8')
        contours, hierarchy = cv2.findContours(y, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # Filter out small contours
        filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) <= thresh]
        # Draw the filtered contours with the background color (e.g., black for a white background)
        for cnt in filtered_contours:
            cv2.drawContours(y, [cnt], -1, (0, 0, 0), -1)  # -1 fill the contour with the color
        # print(y_pred.shape,y_pred,y_true.shape,y_true)
        tp += np.sum((y_pred == 1) & (y_true == 1))
        fp += np.sum((y_pred == 1) & (y_true == 0))
        fn += np.sum((y_pred == 0) & (y_true == 1))
        acc += np.sum(y_pred == y_true)
    return tp,fp,fn,acc

def calculate_f1(tp:float,fp:float,fn:float)->tuple:
    """
    calculate_f1(tp:float,fp:float,fn:float): Calculate F1,
    precision, recall 
    ### Returns: 
        F1, precision, recall 

    """
    precision = tp/(tp+fp+1e-9)
    recall = tp/(tp+fn+1e-9)
    f1 = 2*precision*recall/(precision+recall+1e-9)
    return f1,precision, recall 
