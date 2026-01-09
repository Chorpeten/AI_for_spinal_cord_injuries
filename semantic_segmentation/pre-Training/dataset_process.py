import os
import glob
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

path = './image_dataset/'
dirs = glob.glob(path+'*')
print(dirs)

data_img = []
data_lable = []

for subdir in dirs:
    dirname = subdir.split('/')[-1]
    for filename in os.listdir(subdir):
        img_path = subdir + '/' + filename
        if 'png' in img_path:
            data_lable.append(img_path)
        else:
            data_img.append(img_path)
print(data_img)
print(data_lable)
print(len(data_img))

train_transformer = transforms.Compose([
    transforms.Resize((384,384)), #d 256,256
    transforms.ToTensor(),
])
test_transformer = transforms.Compose([
    transforms.Resize((384,384)), #d 256,256
    transforms.ToTensor(),
])

class SpineMRIdataset(Dataset):
    def __init__(self, img, mask, transformer):
        self.img = img
        self.mask = mask
        self.transformer = transformer
    def __getitem__(self, index):
        img = self.img[index]
        mask = self.mask[index]

        img_open = Image.open(img)
        img_tensor = self.transformer(img_open)

        mask_open = Image.open(mask)
        mask_tensor = self.transformer(mask_open)

        mask_tensor = torch.squeeze(mask_tensor).type(torch.long)

        return img_tensor, mask_tensor
    def __len__(self):
        return len(self.img)
s = 39
train_img = data_img[:s]
train_lable = data_lable[:s]
test_img = data_img[s:]
test_lable = data_lable[s:]

train_data = SpineMRIdataset(train_img, train_lable, train_transformer)
test_data = SpineMRIdataset(test_img, test_lable, test_transformer)

dl_train = DataLoader(train_data, batch_size=8, shuffle=True)
dl_test = DataLoader(test_data, batch_size=8, shuffle=True)

img,lable = next(iter(dl_train))
print(img.type())

torch.save(img, './Tensor/img_tensor.pt')
torch.save(lable, './Tensor/lable_tensor.pt')

# plt.figure(figsize=(12,8))
# for i, (img,lable) in enumerate(zip(img[:4],lable[:4])):
#     img=img.permute(1,2,0).numpy()
#     lable=lable.numpy()
#     plt.subplot(2,4,i+1)
#     plt.imshow(img)
#     plt.subplot(2,4,i+5)
#     plt.imshow(lable)
# plt.show()
