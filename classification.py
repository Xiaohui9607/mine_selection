import os
import torch
import glob
import random
import numpy as np
from torch.utils.data import DataLoader, Dataset
import PIL.Image as Image
from torch import nn
from torchvision import transforms
from torchvision.transforms import functional as F

# create dataset
class mineDataset(Dataset):
    def __init__(self, root, image_transform=None, device='cpu'):
        self.transform = image_transform
        self.device = device
        self.samples = glob.glob(os.path.join(root, '*/high/*/*.tif'))
        pass

    def __getitem__(self, index):
        ih = Image.open(self.samples[index])
        il = Image.open(self.samples[index].replace('high', 'low'))
        # ir = ih / il
        # ih = ih / (2.0**16)
        # il = il / (2.0**16)
        # compact_image = np.stack([ih,il,ir], axis=-1).astype(np.float)
        image = self.transform((ih,il))
        label = torch.ones(1, dtype=torch.long) if 'fine' in self.samples[index] else torch.zeros(1, dtype=torch.long)
        return image.to(self.device), label.to(self.device)

    def __len__(self):
        return len(self.samples)

# create dataloader
def build_dataloader(data_dir):
    class ToPILImage_pair:
        def __call__(self, imgs):
            return F.to_pil_image(imgs[0]), F.to_pil_image(imgs[1])

    class RandomHorizontalFlip_pair:
        def __init__(self, p=0.5):
            self.p = p
        def __call__(self, imgs):
            if random.random() < self.p:
                return F.hflip(imgs[0]), F.hflip(imgs[1])
            return imgs


    class RandomVerticalFlip_pair:
        def __init__(self, p=0.5):
            self.p = p
        def __call__(self, imgs):
            if random.random() < self.p:
                return F.vflip(imgs[0]), F.vflip(imgs[1])
            return imgs

    class Random90Rotation_pair:
        def __init__(self, p=0.5):
            self.p = p
        def __call__(self, imgs):
            if random.random() < self.p:
                return F.rotate(imgs[0], 90), F.rotate(imgs[1], 90)
            return imgs

    class ToTensor_pair:
        def __call__(self, imgs):
            return torch.from_numpy(np.array(imgs[0]).astype(np.float)), torch.from_numpy(np.array(imgs[1]).astype(np.float))

    class Resize_pair:
        def __init__(self, size):
            self.size = size
        def __call__(self, imgs):
            return imgs[0].resize((self.size, self.size)), \
                   imgs[1].resize((self.size, self.size))

    def compact(imgs):
        ih, il = imgs
        ir = ih / il
        ih = ih / (2.0**16)
        il = il / (2.0**16)
        compact_image = torch.stack([ih,il,ir], dim=-1).permute([2,0,1]).float()
        return compact_image

    train_transform = transforms.Compose([
        Resize_pair(32),
        RandomVerticalFlip_pair(),
        # Random90Rotation_pair(90),
        RandomVerticalFlip_pair(),
        ToTensor_pair(),
        transforms.Lambda(compact)
    ])

    test_transform = transforms.Compose([
        # ToPILImage_pair(),
        Resize_pair(32),
        ToTensor_pair(),
        transforms.Lambda(compact)
    ])

    train_ds = mineDataset(
        root=os.path.join(data_dir + '/train'),
        image_transform=train_transform,
        device='cuda'
    )

    valid_ds = mineDataset(
        root=os.path.join(data_dir + '/test'),
        image_transform=test_transform,
        device='cuda'
    )

    train_dl = DataLoader(dataset=train_ds, batch_size=32, shuffle=True, drop_last=False)
    valid_dl = DataLoader(dataset=valid_ds, batch_size=32, shuffle=False, drop_last=False)
    return train_dl, valid_dl


# train

def train():
    import torch.optim as optim
    from networks import LeNet
    net = LeNet()
    net.to('cuda')
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    trainloader, testloader = build_dataloader('./data')

    for epoch in range(20):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            labels = labels.squeeze()
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            if i % 5 == 0:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, loss.item()))

        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                labels = labels.squeeze()
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the test images: %d %%' % (
                100 * correct / total))
    print('Finished Training')

if __name__ == '__main__':
    train()