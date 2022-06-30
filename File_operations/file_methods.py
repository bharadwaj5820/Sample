import os
import shutil
import pickle
class File_methods:
    def __init__(self):
        self.directory="Models"
    def Save_model(self,model,filename):
        path=os.path.join(self.directory,filename)
        if os.path.isdir(path):
            shutil.rmtree(self.directory+"/")
            os.makedirs(path)
        else:
            os.makedirs(path)
        with open(path+'/'+filename+'.sav','wb') as f:
            pickle.dump(model,f)
        return 'success'
    def load_model(self,filename):
        with open(self.directory+'/'+filename+'/'+filename+".sav") as f:
            return pickle.load(f)

class right_model:
    def __init__(self,cluster_number):
        self.cluster_number=cluster_number
    def split1(self):
        self.dir="Models"
        self.list_dir=os.listdir(self.dir)
        print(self.list_dir)
        for dir1 in self.list_dir:
            k=int(dir1.split("_")[-1])
            if k==self.cluster_number:
                return dir1

