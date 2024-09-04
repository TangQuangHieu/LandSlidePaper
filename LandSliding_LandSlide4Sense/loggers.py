import os 

class Logger:
    def __init__(self,path:str):
        self.path = path
        self.f = open(os.path.join(path,"log.txt"),'w')
    def write(self,line):
        self.f.write(line+'\n')
        print(line)
    def close(self):
        self.f.close() 
    