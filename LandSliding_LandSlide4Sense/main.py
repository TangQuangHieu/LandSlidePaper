from trainer import Trainer 
from LandSlide4Sense_data_generator import DataGenerator 
from RAU_Net import RauNet,weights_init,RauNet12
import argparse 
import os 
import platform 
import torch
import numpy as np 
def init_argparse():
    parser = argparse.ArgumentParser(
        usage="%(prog)s --is_train XXX --stored_folder XXX --batch_size XXX",
        description='Settings'
    )
    parser.add_argument('--is_train',required=True,help=' --is_train 1')
    parser.add_argument('--stored_folder',required=True,help=' --stored_folder ../result/segment_pytorch')
    parser.add_argument('--batch_size',required=True,help=' --batch_size 16')
    return parser 
def main():
    parser = init_argparse() 
    args = parser.parse_args() 

    ## Check libraries version 
    print('Python version:',platform.python_version())
    print('Pytorch version',torch.__version__)
    print('Numpy version',np.__version__)
    print('args params',args)

    ## Assign parameters 
    is_train = int(args.is_train)
    stored_folder = os.path.join('../results',args.stored_folder)
    batch_size = int(args.batch_size)
    if not os.path.exists(stored_folder):
        os.makedirs(stored_folder)
    generator = DataGenerator(data_dir='../LandSlide4Sense_dataset', 
                                batch_size=batch_size, 
                                stored_folder=stored_folder)
    if is_train:
        # Training
        model = RauNet12(input_channels=25)
        model.apply(weights_init)
        trainer= Trainer(generator=generator,model=model,epoch=150,path=stored_folder)
        trainer.train()
    else:
        # Testing
        model = torch.load(os.path.join(stored_folder,'best_model.pth'))
        trainer= Trainer(generator=generator,model=model,epoch=300,path=stored_folder)
        trainer.analyse()

if __name__=='__main__':
    main() 

