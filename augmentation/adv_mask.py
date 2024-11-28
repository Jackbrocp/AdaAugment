import numpy as np
from numpy.lib.polynomial import poly
from PIL import Image
import cv2
import os
from tqdm import tqdm
import random
import torchvision.transforms as transforms
import pickle as cPickle
import scipy.misc
transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

class adv_mask():
    def __init__(self,adv_path ,save_path,ratio, transform) :
        self.adv_path=adv_path
        self.save_path=save_path
        self.ratio = ratio
        self.transform = transform
    def read_adv_mask(self):
        # 读取adv_mask
        files= os.listdir(self.adv_path)
        del_num=[]
        for i in tqdm(range(len(files))):
            file = files[i]
        # for file in tqdm(files):
            if 'mask' in file:
                mask = Image.open(self.adv_path+file)
                point_list = self.get_point_pos(mask)
                ##### 保存point_list
                np.save('/nas/yangsuorong/adv_Aug/point_list/'+file[:-4],point_list)
                mask_points = self.cal_mask_point(list(point_list))
                ##### 计算删除点的比例
                del_num.append((len(point_list)-len(mask_points))/len(point_list) )
                img = Image.open(self.adv_path+file[:file.index('_')]+'.png')
                for pos in mask_points:
                    img[pos[0],pos[1]]=[0,0,0]
                # cv2.imwrite(self.save_path+file,img)                
                img.save(self.save_path+file)
                # break
        print(np.mean(del_num))
    def find_point_in_list(self,pos,point_list):
        # 在point_list中删去相邻点
        x,y=pos
        x_off=[-1,1,0,0]
        y_off=[0,0,-1,1]
        illegal_value=[]
        for i in range(4):
            x_=min(31,max(0,x+x_off[i]))
            y_=min(31,max(0,y+y_off[i]))
            illegal_value.append(np.array([x_,y_]))
            # illegal_value.append(np.array([min(31,max(0,x+x_off[i])),min(31,max(0,y+y_off[i]))]))
        for i in range(len(point_list)):
            
            for j in range(4):
                # print(i,len(point_list) )
                x_ill=illegal_value[j][0]
                y_ill=illegal_value[j][1]
                # print(i,len(point_list) )
                if i>len(point_list)-1:
                    return point_list
                if  point_list[i][0]==x_ill and point_list[i][1]==y_ill:
                    del point_list[i]
            
        return point_list
    def cal_mask_point(self,point_list):
        # 根据point_list中的点，选出待mask的点
        # 标准：不连续选点,mask点数限制在50%以内
        # 策略：随机抽取点，查上下左右是否在list内，若是，删去。
        # print('cal_mask_point')
        length=len(point_list)
        mask_points=[]
        while 1:
            pos_index = random.randint(0,len(point_list)-1)
            # print(len(point_list),' ',pos_index)
            point_pos = point_list[pos_index]
            del point_list[pos_index]
            mask_points.append(point_pos)
            point_list = self.find_point_in_list(point_pos,point_list)
            if len(point_list)==0 or len(mask_points)>self.ratio*length:
                break
            
        # 可对mask_points中的点做mask
        # print('target:',len(mask_points))
        return mask_points
    def read_adv_mask_v3(self):
        '''
        用来制作point_list文件,
        读取mask图片,得到mask点的坐标.
        '''
        # files= os.listdir(self.adv_path)
        for i in tqdm(range(50000)):
            # 读取一个mask图片
            file = str(i)+'_mask.png'
            # print(self.adv_path+file)
            mask = cv2.imread(self.adv_path+file)
            # mask = cv2.imread(file)
            ##### 读取point_list
            point_list = self.get_point_pos(mask)
            ############# 测试导入npy文件 #############
            point_list = np.load('./CIFAR100_Dataset/point_list/'+str(i)+'.npy')
            cv2.imwrite('standard.png',mask)
            table = np.zeros((32,32,3))
            for pos in point_list:
                table[pos[0],pos[1]] = [255,255,255]
            cv2.imwrite('table.png',table)
            return
            ##### 保存point_list #####
            np.save('./CIFAR100_Dataset/point_list/'+str(i),point_list)
            # mask_points = self.cal_mask_point(list(point_list))
            ##### 计算删除点的比例
            # img = cv2.imread(self.adv_path+file[:file.index('_')]+'.png')
            # img = self.transform(img).permute(1,2,0).numpy()
            # for pos in mask_points:
            #     img[pos[0],pos[1]]=[0,0,0]
                    # print(pos[0],pos[1])
                    # print(pos[0]+1,pos[1]+1)
                    # print(pos[0]+2,pos[1]+2)
                    # print(img.shape)
                    # print(img[pos[0]:pos[0]+2,pos[1]:pos[1]+2,:])
                    # img[pos[0],pos[1]]=[0,0,0]
                    # img[pos[0],min(pos[1]+1,31)]=[0,0,0]
                    # img[min(pos[0]+1,31),pos[1]]=[0,0,0]
                    # img[min(pos[0]+1,31),min(pos[1]+1,31)]=[0,0,0]
                     
            # cv2.imwrite(self.save_path+file,img)                
                # print(img.shape)
                # cv2.imwrite('./deleted/'+file,img)
                # return
    def load_pointlist(self):
        print('==>load point list')
        savepath = './adv_mask/myDataset/'
        dict = {}
        for i in tqdm(range(25000)):
            path = '/nas/yangsuorong/adv_Aug/point_list/'+str(i)+'_mask.npy'
            if not os.path.exists(path):
                continue
            points = np.load(path)
            dict[str(i)+'.png']=points
        with open(os.path.join(savepath, 'point_list'), 'wb') as fi:
            cPickle.dump(dict, fi)
        return dict
    def get_point_pos(self,mask):
        # 获取mask中扰动点的位置
        # print(mask)
        point_list = np.argwhere(mask[:,:,0] == 255)
        # print(mask[point_list[0][0],point_list[0][1]])
        return point_list
    def main(self):
        # print('success')
        self.read_adv_mask_v3()
    
if __name__=='__main__':
    # adv_path='/nas/lijinqiao/project/AutoAdversarial/result/cifar10/ours/1638880967.3780322/adv_image/'
    adv_path = '/nas/lijinqiao/project/AutoAdversarial/result/cifar100/ours/1640674941.7526093/adv_image/'
    save_path='./CIFAR100_Dataset/'
    advMask=adv_mask(adv_path,save_path,0.5,transform)
    advMask.main()
