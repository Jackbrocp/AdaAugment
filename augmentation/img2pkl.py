import numpy as np
# import chardet
from PIL import Image
import operator
import os
import cv2
from tqdm import tqdm
import torch
import sys
import pickle
import pickle as cPickle
import torchvision.transforms as transforms
 
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

def imread(im_path, shape=None, color="RGB", mode=cv2.IMREAD_UNCHANGED):
    im = cv2.imread(im_path, cv2.IMREAD_UNCHANGED)
    if color == "RGB":
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    # im = np.transpose(im, [2, 1, 0])
    if shape != None:
        assert isinstance(shape, int) 
        im = cv2.resize(im, (shape, shape))
    return im
def read_pkl():
    # path='/nas/lijinqiao/project/AutoAdversarial/result/cifar10/ours/1638880967.3780322/process_result/labels.pkl'
    path = '/nas/lijinqiao/project/AutoAdversarial/result/cifar100/ours/1640674941.7526093/process_result/labels.pkl'
    with open(path,'rb') as file:
        label_data = pickle.load(file,encoding='bytes')
    return label_data
def read_img(filename,path,color='RGB'):
    im = imread(path)
def get_data_and_lst():
    DATA_LEN = 3072
    CHANNEL_LEN = 1024
    SHAPE = 32
    total_num = 50000 
    # folder = '/nas/lijinqiao/project/AutoAdversarial/result/cifar10/ours/1638880967.3780322/adv_image/'
    folder = '/nas/lijinqiao/project/AutoAdversarial/result/cifar100/ours/1640674941.7526093/adv_image/'
    idx=0
    data=np.zeros((total_num, DATA_LEN), dtype=np.uint8)
    s, c = SHAPE, CHANNEL_LEN
    filename_list=[]
    for i in tqdm(range(total_num)):
        img_path=folder+str(i)+'.png'
        # if not os.path.exists(img_path):
        #     continue
        # 存在图片
        filename_list.append(str(i)+'.png')
        im = imread(img_path, shape=s, color='RGB')
        data[idx,:c]=np.reshape(im[:,:,0], c)
        data[idx,c:c*2]=np.reshape(im[:,:,1], c)
        data[idx, 2*c:] = np.reshape(im[:,:,2], c)
        idx+=1
    return data,filename_list
def save_pickle(savepath,data,labels,fnames_list,batch_label,idx):
    assert os.path.isdir(savepath)
    dict = {'data':data,
            'labels':labels,
            'filenames':fnames_list,
            'batch_label':batch_label}
    with open(os.path.join(savepath, 'adv_data_batch_'+str(idx)), 'wb') as fi:
        cPickle.dump(dict, fi)
def pickled(savepath, data, label, fnames, bin_num=1, mode="train"):
  '''
    savepath (str): save path
    data (array): image data, a nx3072 array
    label (list): image label, a list with length n
    fnames (str list): image names, a list with length n
    bin_num (int): save data in several files
    mode (str): {'train', 'test'}
  '''
  assert os.path.isdir(savepath)
  total_num = len(fnames)
  samples_per_bin = int(total_num / bin_num)
  assert samples_per_bin > 0
  idx = 0
  for i in range(bin_num): 
    start = i*samples_per_bin
    end = (i+1)*samples_per_bin
    print('end=',end)
    
    if end <= total_num:
      dict = {'data': data[start:end, :],
              'labels': label[start:end],
              'filenames': fnames[start:end]}
    else:
      dict = {'data': data[start:, :],
              'labels': label[start:],
              'filenames': fnames[start:]}
    if mode == "train":
      dict['batch_aug_label'] = "training batch {} of {}".format(idx, bin_num)
    else:
      dict['batch_aug_label'] = "testing batch {} of {}".format(idx, bin_num)
      
    with open(os.path.join(savepath, 'data_batch_'+str(idx)), 'wb') as fi:
      cPickle.dump(dict, fi)
    idx = idx + 1
def re_save_cifar_batch(data_1,labels_1,lst_1,data_2,labels_2,lst_2 ):
    i=1
    total_batch=7
 
    for i in range(1,6):
        path='../data/cifar-10-batches-py/data_batch_'+str(i)
        with open(path,'rb') as file:
            label_data = pickle.load(file,encoding='bytes')
        data = label_data[b'data']
        labels = label_data[b'labels']
        filenames = label_data[b'filenames']

        batch_label='training batch {} of {}'.format(i,total_batch)
        save_pickle('./myDataset',data,labels,filenames,batch_label,i)
def save_variable(data,filename):
    f=open(filename,'wb')
    pickle.dump(data,f)
    f.close()
    return 
def load_variable(filename):
    f=open(filename,'rb')
    r=pickle.load(f)
    f.close()
    return r
def test_pickle_img():
    
    for j in range(1,6):
        j=1
        path = './advMask_Dataset/adv_data_batch_1'
        with open(path,'rb') as file:
            label_data = pickle.load(file,encoding='bytes')

        # print(label_data.keys())
        data = label_data['data']
        data = np.vstack(data).reshape(-1, 3, 32, 32)
        data = data.transpose((0, 2, 3, 1))#.astype(np.float32)
        # print('data shape:',data.shape)
        labels = label_data['labels']
        # filename_list = label_data['filenames']
        # print(len(filename_list))
        # print(filename_list[:10])
        img = data[500]
        print(labels[500])
        img = Image.fromarray(img)
        img.save('./test.png')
        return
        batch_label = label_data['batch_label']
        for i in range(len(data)):
            print(data[i][0][0])
            t=transform(data[i]).permute(1,2,0)
            print('t.shape:',t.shape)
            data_tensor[i]=t
            print(t[0][0])
            print(data_tensor[0][0])
            data[i]=transform(data[i]).permute(1,2,0)
            print(data[i][0][0])
            return
        print('loop:',type(data))
        data = torch.tensor(data).view(len(data),-1)
        print(data.shape)
        return
        save_pickle('./myDataset',data,labels,filename_list,batch_label,1)
    print(type(data))
    print(type(labels))
    print(type(filename_list))
    return
    num=np.zeros(11)
    for i in labels:
        num[i]+=1
    print(num)
    return
    filename = filename_list[0]

    ori_img = Image.open('./data/adv_image/'+filename)
    ori_img = np.array(ori_img)
    img = data[0,:]
    r=img[:1024].reshape(32,32)
    g=img[1024:2048].reshape(32,32)
    b=img[2048:3072].reshape(32,32)
    img = cv2.merge((b,g,r))
    cv2.imwrite('./ot.png',img)
    print(labels[0])
    img = img.reshape(32,32,3)
    img = Image.fromarray(img)
    img.save('./test.png')
def shuffle():
    path = './myDataset/'
    for i in range(1,8):
        path='./myDataset/data_aug_batch_'+str(i)
        with open(path,'rb') as file:
            label_data = pickle.load(file,encoding='bytes')
        print('i=',i)
        if i==1:
            data = label_data['data']
            labels = label_data['labels']
            filenames = label_data['filenames']
        else:
            data = np.r_[data,label_data['data']]
            if i==7:
                labels = labels + label_data['labels'].numpy().tolist()
            else:
                labels = labels + label_data['labels']
            filenames = filenames + label_data['filenames']
    data=data.tolist()
    state = np.random.get_state()
    np.random.shuffle(data)
    np.random.set_state(state)
    np.random.shuffle(labels)
    np.random.set_state(state)
    np.random.shuffle(filenames)
    for i in range(7):
        data_ = data[i*10000:(i+1)*10000]
        label_ = labels[i*10000:(i+1)*10000]
        filename_ = filenames[i*10000:(i+1)*10000]
        batch_label='training batch {} of {}'.format(i+1,7)
        
        save_pickle('./myDataset',np.array(data_),label_,filename_,batch_label,i+1)

        # print(type(data_),np.array(data_).shape)
        # print(type(label_),np.array(label_).shape )
        # print(type(filename_),np.array(filename_).shape)
        print(batch_label)
    return
    # data = label_data['data']
    # labels = label_data['labels']
    # print(len(data)==len(labels))
    # # label_names=label_data[b'batch_labels']
    # # print(label_names)
    # lst  = label_data[b'filenames']
    # batch_label = label_data[b'batch_label']
    # data_1,labels_1,lst_1 = data[:10000], labels[:10000], lst[:10000]
    # labels_1 = labels_1.numpy().tolist()

    # # save_pickle('./myDataset',data_1,labels_1,lst_1,6,7)
    # data_2,labels_2,lst_2 = data[10000:20000], labels[10000:20000],lst[10000:20000]
    # labels_2 = labels_2.numpy().tolist()
def main():
    # data,filename_list=get_data_and_lst()
    # labels=read_pkl()
    save_path='./CIFAR100_Dataset/'
    # save_variable(data,save_path+'data.pkl')
    # save_variable(filename_list,save_path+'lst.pkl')
    # save_variable(labels,save_path+'labels.pkl')
    # print(data.shape)
    data=load_variable(save_path+'data.pkl')
    filename_list=load_variable(save_path+'lst.pkl')
    labels=load_variable(save_path+'labels.pkl')
    batch_label = 'training batch 1 of 1'
    labels=labels.numpy().tolist()
    save_pickle(save_path,data,labels,filename_list,batch_label,0)
    # for i in range(5):
    #     batch_label = "training batch {} of {}".format(i, 5)
    #     print(batch_label)
    #     single_len = 10000
    #     d,l,lst = data[single_len*i:single_len*(i+1)],labels[single_len*i:single_len*(i+1)],filename_list[single_len*i:single_len*(i+1)]
    #     l  = l .numpy().tolist()
    #     save_pickle(save_path,d ,l ,lst,batch_label,i+1)
     
if __name__=='__main__':
    # test_pickle_img()
    # exit()
    # shuffle()
    main()


    
    # data = open('data.pkl','rb')
    # data = pickle.load(data)
    # filename_list=open('lst.pkl','rb')
    # filename_list = pickle.load(filename_list)
    # labels = open('labels.pkl','rb')
    # labels = pickle.load(labels)
    # exit()

    # data=load_variable('./data.pkl')
    # lst=load_variable('lst.pkl')
    # labels=load_variable('labels.pkl')
    # print(data.shape)
    # print(len(filename_list),len(labels))

    # re_save_cifar_batch()
    # path='../data/cifar-10-batches-py/data_batch_1'
    
 

