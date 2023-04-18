import numpy as np
import skimage
import torch
from PIL import Image
from skimage import color
from skimage.util import img_as_float
from tqdm import tqdm

from torch import nn
from torch.optim import Adam
import matplotlib.pyplot as plt
from albumentations.pytorch import ToTensorV2
import albumentations as A

from model.FoodDataset import FoodDataset
from model.unet import Unet_model

#
# data_dir = ['E:/PythonModels/archive/data'+i+'/data'+i for i in ['X','Y']]
# def get_images(image_dir,transform = None,batch_size=1,shuffle=True,pin_memory=True):
#     data = FoodDataset(image_dir,transform = t1)
#     train_size = int(0.8 * data.__len__())
#     test_size = data.__len__() - train_size
#     train_dataset, test_dataset = torch.utils.data.random_split(data, [train_size, test_size])
#     train_batch = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory)
#     test_batch = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory)
#     return train_batch,test_batch
#
t1 = A.Compose([
    A.Resize(160,240),
    A.augmentations.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2()
])
#
# train_batch,test_batch = get_images(data_dir,transform =t1,batch_size=8)
#
#
#
# for img,mask in train_batch:
#     img1 = np.transpose(img[0,:,:,:],(1,2,0))
#     mask1 = np.array(mask[0,:,:])
#     img2 = np.transpose(img[1,:,:,:],(1,2,0))
#     mask2 = np.array(mask[1,:,:])
#     img3 = np.transpose(img[2,:,:,:],(1,2,0))
#     mask3 = np.array(mask[2,:,:])
#     fig , ax =  plt.subplots(3, 2, figsize=(18, 18))
#     ax[0][0].imshow(img1)
#     ax[0][1].imshow(mask1)
#     ax[1][0].imshow(img2)
#     ax[1][1].imshow(mask2)
#     ax[2][0].imshow(img3)
#     ax[2][1].imshow(mask3)
#     break
#
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
#
# model = Unet_model().to(DEVICE)
#
#
# from torchsummary import summary
# summary(model, (3, 256, 256))
#
# LEARNING_RATE = 1e-4
# num_epochs = 2
#
# loss_fn = nn.CrossEntropyLoss()
# optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
# scaler = torch.cuda.amp.GradScaler()
#
# for epoch in range(num_epochs):
#     loop = tqdm(enumerate(train_batch),total=len(train_batch))
#     for batch_idx, (data, targets) in loop:
#         data = data.to(DEVICE)
#         targets = targets.to(DEVICE)
#         targets = targets.type(torch.long)
#         # forward
#         with torch.cuda.amp.autocast():
#             predictions = model(data)
#             loss = loss_fn(predictions, targets)
#         # backward
#         optimizer.zero_grad()
#         scaler.scale(loss).backward()
#         scaler.step(optimizer)
#         scaler.update()
#
#         # update tqdm loop
#         loop.set_postfix(loss=loss.item())
#
#
# def check_accuracy(loader, model):
#     num_correct = 0
#     num_pixels = 0
#     dice_score = 0
#     model.eval()
#
#     with torch.no_grad():
#         for x, y in loader:
#             x = x.to(DEVICE)
#             y = y.to(DEVICE)
#             softmax = nn.Softmax(dim=1)
#             preds = torch.argmax(softmax(model(x)),axis=1)
#             num_correct += (preds == y).sum()
#             num_pixels += torch.numel(preds)
#             dice_score += (2 * (preds * y).sum()) / ((preds + y).sum() + 1e-8)
#
#     print(f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}")
#     print(f"Dice score: {dice_score/len(loader)}")
#     model.train()
#
#
# check_accuracy(train_batch, model)
#
#
# check_accuracy(test_batch, model)
#
#
# for x,y in test_batch:
#     x = x.to(DEVICE)
#     fig , ax =  plt.subplots(3, 3, figsize=(18, 18))
#     softmax = nn.Softmax(dim=1)
#     preds = torch.argmax(softmax(model(x)),axis=1).to('cpu')
#     img1 = np.transpose(np.array(x[0,:,:,:].to('cpu')),(1,2,0))
#     preds1 = np.array(preds[0,:,:])
#     mask1 = np.array(y[0,:,:])
#     img2 = np.transpose(np.array(x[1,:,:,:].to('cpu')),(1,2,0))
#     preds2 = np.array(preds[1,:,:])
#     mask2 = np.array(y[1,:,:])
#     img3 = np.transpose(np.array(x[2,:,:,:].to('cpu')),(1,2,0))
#     preds3 = np.array(preds[2,:,:])
#     mask3 = np.array(y[2,:,:])
#     ax[0,0].set_title('Image')
#     ax[0,1].set_title('Prediction')
#     ax[0,2].set_title('Mask')
#     ax[1,0].set_title('Image')
#     ax[1,1].set_title('Prediction')
#     ax[1,2].set_title('Mask')
#     ax[2,0].set_title('Image')
#     ax[2,1].set_title('Prediction')
#     ax[2,2].set_title('Mask')
#     ax[0][0].axis("off")
#     ax[1][0].axis("off")
#     ax[2][0].axis("off")
#     ax[0][1].axis("off")
#     ax[1][1].axis("off")
#     ax[2][1].axis("off")
#     ax[0][2].axis("off")
#     ax[1][2].axis("off")
#     ax[2][2].axis("off")
#     ax[0][0].imshow(img1)
#     ax[0][1].imshow(preds1)
#     ax[0][2].imshow(mask1)
#     ax[1][0].imshow(img2)
#     ax[1][1].imshow(preds2)
#     ax[1][2].imshow(mask2)
#     ax[2][0].imshow(img3)
#     ax[2][1].imshow(preds3)
#     ax[2][2].imshow(mask3)
#     break


# import torch
# from torch import nn
#
# from model.unet import Unet_model
#
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
#
# def load_checkpoint(checkpoint, model):
#     model.load_state_dict(checkpoint["state_dict"])
#
# model = Unet_model().to(DEVICE)
# load_checkpoint(torch.load('checkpoint.pth.tar'), model)
#
# from PIL import Image
#
# x = np.array(Image.open('E:\Licenta_DOC\API_Segmentation\data\\train\img\\00000000.jpg'))
#
# x = t1(image=x)['image']
# x = x.to(DEVICE)
#
# print(x.shape)
# x = x.unsqueeze(0)
# print(x.shape)
# softmax = nn.Softmax(dim = 1)
# preds = torch.argmax(softmax(model(x)), axis=1).to('cpu')
# img1 = np.transpose(np.array(x[0, :, :, :].to('cpu')), (1, 2, 0))
# preds1 = np.array(preds[0, :, :])
# # 160 x 240
# print(preds1)

photo = np.array(Image.open('E:\Licenta_DOC\App\src\main\\resources\image\\target_python.jpg').convert('RGB'))


mask  = np.random.randint(1,4, (photo.shape[0], photo.shape[1]))

mask[:300, :] = 1
mask[300:, ] = 0

overlay_photo = color.label2rgb(mask, photo, saturation=1, alpha=0.5, bg_color=None)
skimage.io.imshow(overlay_photo)
plt.show()

Image.fromarray((overlay_photo * 255).astype(np.uint8)).save('overlay-photo.png')
