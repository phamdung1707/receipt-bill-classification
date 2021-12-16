#
# You can modify this files
#
import os
import torch
import torch.nn as nn
from torchvision import models, transforms,datasets
from PIL import Image
from torch.utils.data import DataLoader
class HoadonOCR:

    def __init__(self):
        # Init parameters, load model here
        self.model = models.resnet50(pretrained=True)
        for param in self.model.parameters():
                param.require_grad = False
        
        self.model=torch.load("data_model_47.pt")
    # TODO: implement find label
    def find_label(self, img_path):
        # transform = transforms.Compose([
        #     transforms.Resize(size=256),
        #     transforms.CenterCrop(size=224),
        #     transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406],
        #                      [0.229, 0.224, 0.225])
        # ])
        image_transforms = { 
            'train': transforms.Compose([
                transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
                transforms.RandomRotation(degrees=15),
                transforms.RandomHorizontalFlip(),
                transforms.CenterCrop(size=224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
        ]),
            'valid': transforms.Compose([
                transforms.Resize(size=256),
                transforms.CenterCrop(size=224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                    [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(size=256),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225])
        ])
    }

        self.labels = ['highlands', 'starbucks', 'phuclong', 'others']
        dataset = 'data'
        
        train_directory = os.path.join(dataset, 'train')
        valid_directory = os.path.join(dataset, 'test')
        test_directory = os.path.join(dataset, 'test')
     

        data = {
               'train': datasets.ImageFolder(root=train_directory, transform=image_transforms['train']),
                'valid': datasets.ImageFolder(root=valid_directory, transform=image_transforms['valid']),
                'test': datasets.ImageFolder(root=test_directory, transform=image_transforms['test'])
            }
        train_directory = os.path.join(dataset, 'train')
        bs = 32
        idx_to_class = {v: k for k, v in data['train'].class_to_idx.items()}
        train_data_size = len(data['train'])
        valid_data_size = len(data['valid'])
        test_data_size = len(data['test'])
        train_data_loader = DataLoader(data['train'], batch_size=bs, shuffle=True)
        valid_data_loader = DataLoader(data['valid'], batch_size=bs, shuffle=True)
        test_data_loader = DataLoader(data['test'], batch_size=bs, shuffle=True)
        test_image = Image.open(img_path)
        test_image_tensor = image_transforms['test'](test_image)
        test_image_tensor = test_image_tensor.view(1, 3, 224, 224)
        with torch.no_grad():
            self.model.eval()
            out = self.model(test_image_tensor)
            idx_to_class = {v: k for k, v in data['train'].class_to_idx.items()}
            ps = torch.exp(out)
            topk, topclass = ps.topk(3, dim=1)
            cls = topclass.cpu().numpy()[0][0]
            text=idx_to_class[topclass.cpu().numpy()[0][0]]
            return text; 
