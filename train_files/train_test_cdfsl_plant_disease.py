import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import pandas as pd
import torchvision.transforms as transforms
from torch.nn import functional as F
#import datasets.additional_transforms as add_transforms
# import data.additional_transforms as add_transforms
from torch.utils.data import Dataset, DataLoader, TensorDataset
from abc import abstractmethod
from torchvision.datasets import ImageFolder
from scipy.spatial.distance import cdist
from clip.clip import _MODELS, _download, _transform, available_models
from clip.model import CLIP, convert_weights
from clip import clip
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import time



def load_clip_to_cpu():
    backbone_name = 'ViT-B/16'
    url = _MODELS[backbone_name]
    model_path = _download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    design_details = {"trainer": 'MaPLe',
                      "vision_depth": 0,
                      "language_depth": 0, "vision_ctx": 0,
                      "language_ctx": 0,
                      "maple_length": 2}
    model = clip.build_model(state_dict or model.state_dict(), design_details)
    
    return model


CropDisease_path = '.../datasets_for_bscdfsl/Plant_disease'

identity = lambda x:x
class SimpleDataset:
    def __init__(self, transform, target_transform=identity):
        self.transform = transform
        self.target_transform = target_transform

        self.meta = {}

        self.meta['image_names'] = []
        self.meta['image_labels'] = []


        d = ImageFolder(CropDisease_path + "/dataset/train/")

        for i, (data, label) in enumerate(d):
            self.meta['image_names'].append(data)
            self.meta['image_labels'].append(label)  

    def __getitem__(self, i):

        img = self.transform(self.meta['image_names'][i])
        target = self.target_transform(self.meta['image_labels'][i])

        return img, target

    def __len__(self):
        return len(self.meta['image_names'])
    
class SetDataset:
    def __init__(self, batch_size, transform):

        self.sub_meta = {}
        self.cl_list = range(38)

        for cl in self.cl_list:
            self.sub_meta[cl] = []

        d = ImageFolder(CropDisease_path + "/dataset/train/")


        for i, (data, label) in enumerate(d):
            self.sub_meta[label].append(data)
    

        for key, item in self.sub_meta.items():
            print (len(self.sub_meta[key]))

        self.sub_dataloader = [] 
        sub_data_loader_params = dict(batch_size = batch_size,
                                  shuffle = True,
                                  num_workers = 0, #use main thread only or may receive multiple batches
                                  pin_memory = False)        
        for cl in self.cl_list:
            sub_dataset = SubDataset(self.sub_meta[cl], cl, transform = transform )
            self.sub_dataloader.append( torch.utils.data.DataLoader(sub_dataset, **sub_data_loader_params) )

    def __getitem__(self, i):
        return next(iter(self.sub_dataloader[i]))

    def __len__(self):
        return len(self.sub_dataloader)

class SubDataset:
    def __init__(self, sub_meta, cl, transform=transforms.ToTensor(), target_transform=identity):
        self.sub_meta = sub_meta
        self.cl = cl 
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self,i):

        img = self.transform(self.sub_meta[i])
        target = self.target_transform(self.cl)
        return img, target

    def __len__(self):
        return len(self.sub_meta)
    
class EpisodicBatchSampler(object):
    def __init__(self, n_classes, n_way, n_episodes, seed):
        self.n_classes = n_classes
        self.n_way = n_way
        self.n_episodes = n_episodes
        self.seed = seed
        torch.manual_seed(seed)

    def __len__(self):
        return self.n_episodes

    def __iter__(self):
        for i in range(self.n_episodes):
            yield torch.randperm(self.n_classes)[:self.n_way]
                
from PIL import ImageEnhance

transformtypedict=dict(Brightness=ImageEnhance.Brightness, Contrast=ImageEnhance.Contrast, Sharpness=ImageEnhance.Sharpness, Color=ImageEnhance.Color)



class ImageJitter(object):
    def __init__(self, transformdict):
        self.transforms = [(transformtypedict[k], transformdict[k]) for k in transformdict]


    def __call__(self, img):
        out = img
        randtensor = torch.rand(len(self.transforms))

        for i, (transformer, alpha) in enumerate(self.transforms):
            r = alpha*(randtensor[i]*2.0 -1.0) + 1
            out = transformer(out).enhance(r).convert('RGB')

        return out

class TransformLoader:
    def __init__(self, image_size, 
                 normalize_param    = dict(mean= [0.485, 0.456, 0.406] , std=[0.229, 0.224, 0.225]),
                 jitter_param       = dict(Brightness=0.4, Contrast=0.4, Color=0.4)):
        self.image_size = image_size
        self.normalize_param = normalize_param
        self.jitter_param = jitter_param
    
    def parse_transform(self, transform_type):
        if transform_type=='ImageJitter':
            method = ImageJitter( self.jitter_param )
            return method
        method = getattr(transforms, transform_type)
        if transform_type=='RandomSizedCrop':
            return method(self.image_size) 
        elif transform_type=='CenterCrop':
            return method(self.image_size) 
        elif transform_type=='Resize':
            return method([int(self.image_size*1.15), int(self.image_size*1.15)])
        elif transform_type=='Normalize':
            return method(**self.normalize_param )
        else:
            return method()

    def get_composed_transform(self, aug = False):
        if aug:
            transform_list = ['RandomSizedCrop', 'ImageJitter', 'RandomHorizontalFlip', 'ToTensor', 'Normalize']
        else:
            # transform_list = ['Resize','CenterCrop', 'ToTensor', 'Normalize']
            transform_list = ["ToTensor"]

        transform_funcs = [ self.parse_transform(x) for x in transform_list]
        transform = transforms.Compose(transform_funcs)
        return transform

from abc import abstractmethod
    
class DataManager(object):
    @abstractmethod
    def get_data_loader(self, data_file, aug):
        pass 

class SimpleDataManager(DataManager):
    def __init__(self, image_size, batch_size):        
        super(SimpleDataManager, self).__init__()
        self.batch_size = batch_size
        self.trans_loader = TransformLoader(image_size)

    def get_data_loader(self, aug): #parameters that would change on train/val set
        transform = self.trans_loader.get_composed_transform(aug)
        dataset = SimpleDataset(transform)

        data_loader_params = dict(batch_size = self.batch_size, shuffle = True, num_workers = 12, pin_memory = True)       
        data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)

        return data_loader

class SetDataManager(DataManager):
    def __init__(self, image_size, n_way=5, n_support=1, n_query=15, n_eposide = 100):        
        super(SetDataManager, self).__init__()
        self.image_size = image_size
        self.n_way = n_way
        self.batch_size = n_support + n_query
        self.n_eposide = n_eposide

        self.trans_loader = TransformLoader(image_size)

    def get_data_loader(self, aug): #parameters that would change on train/val set
        transform = self.trans_loader.get_composed_transform(aug)
        dataset = SetDataset(self.batch_size, transform)
        sampler = EpisodicBatchSampler(len(dataset), self.n_way, self.n_eposide, seed = 0 )  
        data_loader_params = dict(batch_sampler = sampler,  num_workers = 12, pin_memory = True)       
        data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)
        return data_loader

few_shot_params = dict(n_way = 5 , n_support = 1) # n_support = n_shot
print(few_shot_params)
datamgr = SetDataManager(224, n_eposide = 600, n_query = 15, **few_shot_params)
novel_loader = datamgr.get_data_loader(aug = False )

n_support = 1 # n_support = n_shot
n_query = 15
n_eposide = 600

from models.Maple_cdfsl import Maple_Model

def Orig_Augmenter(query_set):
    transform1 = transforms.Compose([
                        transforms.ToPILImage(),
                        transforms.Resize((int(224*1.15), int(224*1.15))),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406) , (0.229, 0.224, 0.225))
                    ])
    orig_query_set = []
    for imgs in query_set:
        orig_query_set.append(transform1(imgs))
    orig_query_set = torch.stack(orig_query_set)
    
    return orig_query_set.cuda()



def Augmenter(support_set, num_augs):
    support_set = support_set.cpu()
    
    transform1 = transforms.Compose([
                        transforms.ToPILImage(),
                        transforms.Resize((int(224*1.15), int(224*1.15))),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406) , (0.229, 0.224, 0.225))
                    ])
    
    transform2 = transforms.Compose([
                        transforms.ToPILImage(),
                        transforms.RandomResizedCrop(224),
                        ImageJitter(dict(Brightness=0.4, Contrast=0.4, Color=0.4)),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406) , (0.229, 0.224, 0.225))
                    ])
    
    orig_support_set = []
    for imgs in support_set:
        orig_support_set.append(transform1(imgs))
    orig_support_set = torch.stack(orig_support_set)
    
    if num_augs == 0:
        return orig_support_set.cuda()
    
    aug_support_set = []
    for _ in range(num_augs):
        aug_set = torch.stack([transform2(imgs) for imgs in support_set])
        aug_support_set.append(aug_set)
    
    aug_support_set = torch.stack(aug_support_set)    
    aug_support_set = aug_support_set.reshape((aug_support_set.shape[0] * aug_support_set.shape[1]), 3, 224, 224)
    
    result_support_set = torch.cat((orig_support_set, aug_support_set), dim=0)
    
    return result_support_set.cuda()


def check_cos_sim(idx, support_set, model):
    logits, txtf, imgf = model(support_set)
    print(f"cosine_similarity before training for episode {idx+1}: \n", cosine_similarity(txtf.cpu().detach().numpy(), imgf.cpu().detach().numpy()))


class Custom_CE_Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.cross_entropy_loss = nn.CrossEntropyLoss()
    
    def upper_triangle(self, matrix):
        upper = torch.triu(matrix, diagonal=0)
        #diagonal = torch.mm(matrix, torch.eye(matrix.shape[0]))
        diagonal_mask = torch.eye(matrix.shape[0]).cuda()
        return upper * (1.0 - diagonal_mask)

    def forward(self, logits, support_set_gt, txf, imf):
        ce_loss = self.cross_entropy_loss(logits, support_set_gt)
        
        n_classes = txf.shape[0]
        
        txf_expand1 = txf.unsqueeze(0)
        txf_expand2 = txf.unsqueeze(1)
        # txfx = (txf_expand2 - txf_expand1)**2
        txf_norm_mat = torch.sum((txf_expand2 - txf_expand1)**2, dim=-1)
        txf_norm_upper = self.upper_triangle(txf_norm_mat)
        mu = (2.0 / (n_classes**2 - n_classes)) * torch.sum(txf_norm_upper)
        residuals = self.upper_triangle((txf_norm_upper - mu)**2)
        rw1 = (2.0 / (n_classes**2 - n_classes)) * torch.sum(residuals)
        
        
        num_features = imf.shape[1]
        num_images = imf.shape[0]
        class_sums = torch.zeros((n_classes, num_features)).cuda()
        class_counts = torch.zeros(n_classes).cuda()
        
        for i in range(num_images):
            class_idx = support_set_gt[i]
            class_sums[class_idx] += imf[i]
            class_counts[class_idx] += 1
        
        cls_protos = class_sums / class_counts.view(-1, 1)
        
        
        imf_expand1 = cls_protos.unsqueeze(0)
        imf_expand2 = cls_protos.unsqueeze(1)
        # imfx = (imf_expand2 - imf_expand1)**2
        imf_norm_mat = torch.sum((imf_expand2 - imf_expand1)**2, dim=-1)
        imf_norm_upper = self.upper_triangle(imf_norm_mat)
        mu1 = (2.0 / (n_classes**2 - n_classes)) * torch.sum(imf_norm_upper)
        residuals1 = self.upper_triangle((imf_norm_upper - mu)**2)
        rw2 = (2.0 / (n_classes**2 - n_classes)) * torch.sum(residuals1)
        
        custom_loss = ce_loss + rw1 + rw2
        return custom_loss


def Choose_Augmentations(support_set, new_support_set_gt, model):
    _, txf, imf = model(support_set)
    
    class_imfs = []
    support_set_clses = []
    for i in range(5):
        cls_imf = imf[new_support_set_gt == i]
        support_set_cls = support_set[new_support_set_gt == i]
        class_imfs.append(cls_imf)
        support_set_clses.append(support_set_cls)
    
    
    class1_imf = class_imfs[0]
    class2_imf = class_imfs[1]
    class3_imf = class_imfs[2]
    class4_imf = class_imfs[3]
    class5_imf = class_imfs[4]
    
    class1_logits = (class1_imf @ txf[0, :].t())
    class2_logits = (class2_imf @ txf[1, :].t())
    class3_logits = (class3_imf @ txf[2, :].t())
    class4_logits = (class4_imf @ txf[3, :].t())
    class5_logits = (class5_imf @ txf[4, :].t())
    
    _, top_indices1 = torch.topk(class1_logits, k=15)
    _, top_indices2 = torch.topk(class2_logits, k=15)
    _, top_indices3 = torch.topk(class3_logits, k=15)
    _, top_indices4 = torch.topk(class4_logits, k=15)
    _, top_indices5 = torch.topk(class5_logits, k=15)
    
    support_set_cls1 = support_set_clses[0][top_indices1]
    support_set_cls2 = support_set_clses[1][top_indices2]
    support_set_cls3 = support_set_clses[2][top_indices3]
    support_set_cls4 = support_set_clses[3][top_indices4]
    support_set_cls5 = support_set_clses[4][top_indices5]
    
    support_set_new = []
    for i in range(15):
        a = torch.cat((support_set_cls1[i].unsqueeze(0), support_set_cls2[i].unsqueeze(0), support_set_cls3[i].unsqueeze(0), support_set_cls4[i].unsqueeze(0), support_set_cls5[i].unsqueeze(0)), dim=0)
        support_set_new.append(a)
    support_set_new = torch.cat(support_set_new, dim=0)
    
    return support_set_new



def episode_loop():
    for idx, (x, y) in enumerate(novel_loader):
        print('episode:', idx + 1)
        support_set = x[:, 0:n_support, :,:,:]
        query_set = x[:, n_support:, :,:,:]
        
        support_set_gt = y[:, 0: n_support]
        query_set_gt = y[:, n_support:]
        
        
        support_set = support_set.reshape((support_set.shape[0] * support_set.shape[1]), 3, 256, 256).cuda() # [5,3,224,224] for 5 way-1 shot, [25, 3, 224, 224] for 5 way-5 shot
        query_set = query_set.reshape((query_set.shape[0] * query_set.shape[1]), 3, 256, 256).cuda() # [80, 3, 224, 224] for both settings
    
        num_augments = 20
        support_set = Augmenter(support_set, (num_augments - 1))
        query_set = Orig_Augmenter(query_set)
        
        
        support_set_gt = support_set_gt.flatten() # 5 if 5 way-1 shot, 25 if 5 way-5 shot
        query_set_gt = query_set_gt.flatten()
        
        class_names = ([labels2classnames[labels.item()] for labels in support_set_gt]) 
        unique_classnames = list(dict.fromkeys(class_names)) # list of 5 classnames for the current episode
        remap_labels = {i: cn for i, cn in enumerate(unique_classnames)} # remap labels in 0 to 4 format
        
        episode_labels = [0, 1, 2, 3, 4]
        
        new_support_set_gt = torch.tensor([item for item in episode_labels for _ in range(n_support)]).cuda() # support_set_gt but reformatted to 0-4 range
        new_support_set_gt = new_support_set_gt.repeat(num_augments)
        
        new_query_set_gt = torch.tensor([item for item in episode_labels for _ in range(n_query)]).cuda() # query_set_gt but reformatted to 0-4 range
        
        model = Maple_Model(unique_classnames, clip_model).cuda()
        
        # Make the text and image encoders non-trainable
        for name, param in model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)
        
        
        support_set=Choose_Augmentations(support_set,new_support_set_gt, model)
        new_support_set_gt=new_support_set_gt[:75]
        
        learning_rate = 0.01
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
        loss_fn = Custom_CE_Loss()

        num_epochs = 150
        train_losses = []
        val_losses = []

        progress_bar = tqdm(range(num_epochs))
        
        
        for epoch in progress_bar:
            shuffle = torch.randperm(support_set.shape[0]).cuda()
            support_set = support_set[shuffle, :]
            new_support_set_gt = new_support_set_gt[shuffle]
            
            model.train()
            optimizer.zero_grad()
            logits, txf, imf = model(support_set)
            loss = loss_fn(logits, new_support_set_gt, txf, imf)
            
            train_losses.append(loss.item())
            loss.backward()
            optimizer.step()
            
            model.eval()
            with torch.no_grad():
                s_logits, s_tf, s_if = model(support_set)
                eval_logits, q_tf, q_if  = model(query_set)
                val_loss = loss_fn(eval_logits, new_query_set_gt, q_tf, q_if)
                val_losses.append(val_loss.item())
                _, eval_predict = torch.max(eval_logits, dim=1)
                correct_predictions = (eval_predict == new_query_set_gt).sum().item()
                
                accuracy = correct_predictions / eval_logits.shape[0]
                
            progress_bar.set_postfix({"Accuracy" : accuracy})
        
        print(f'episode {idx + 1} accuracy: {accuracy}')
        episode_accuracies.append(accuracy)
        
        
        print(f'episode {idx + 1} ends...')

    
from clip_modules.interface import *
from clip_modules.model_loader import *

start_time = time.time()

clip_model = load_clip_to_cpu()

labels2classnames = {0: "Apple___Apple_scab", 1: "Apple___Black_rot", 2: "Apple___Cedar_apple_rust", 3: "Apple___healthy", 4: "Blueberry___healthy",
                     5: "Cherry_(including_sour)___Powdery_mildew", 6: "Cherry_(including_sour)___healthy", 7: "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
                     8: "Corn_(maize)___Common_rust_", 9: "Corn_(maize)___Northern_Leaf_Blight",
                     10: "Corn_(maize)___healthy", 11: "Grape___Black_rot", 12: "Grape___Esca_(Black_Measles)", 13: "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
                     14: "Grape___healthy", 15: "Orange___Haunglongbing_(Citrus_greening)", 16: "Peach___Bacterial_spot", 17: "Peach___healthy",
                     18: "Pepper,_bell___Bacterial_spot", 19: "Pepper,_bell___healthy", 20: "Potato___Early_blight", 21: "Potato___Late_blight",
                     22: "Potato___healthy", 23: "Raspberry___healthy", 24: "Soybean___healthy", 25: "Squash___Powdery_mildew",
                     26: "Strawberry___Leaf_scorch", 27: "Strawberry___healthy", 28: "Tomato___Bacterial_spot", 29: "Tomato___Early_blight", 30: "Tomato___Late_blight",
                     31: "Tomato___Leaf_Mold", 32: "Tomato___Septoria_leaf_spot", 33: "Tomato___Spider_mites Two-spotted_spider_mite", 34: "Tomato___Target_Spot",
                     35: "Tomato___Tomato_Yellow_Leaf_Curl_Virus", 36: "Tomato___Tomato_mosaic_virus", 37: "Tomato___healthy"}


episode_accuracies = []
episode_loop()

total_accuracy = (sum(episode_accuracies)) / n_eposide

var = np.sum(((np.array(episode_accuracies) - total_accuracy) ** 2) / n_eposide)
std = np.sqrt(var)

print(f'after {len(episode_accuracies)} episodes, total accuracy is: {total_accuracy} +- {std}')


end_time = time.time()

elapsed_time = end_time - start_time

# Print the elapsed time
print("Time taken:", elapsed_time, "seconds")