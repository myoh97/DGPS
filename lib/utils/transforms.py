import random
from torchvision.transforms import functional as F
# import matplotlib.pyplot as plt

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class RandomHorizontalFlip(object):

    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)
            bbox = target["boxes"]
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target["boxes"] = bbox
            target["flipped"] = True
        return image, target

class Resize(object):
    def __init__(self, prob):
        self.prob = prob
        
    def __call__(self, image, target):
        if random.random() < self.prob:
            width, height = image.size[-2:]
            target_size = (int(height/2), int(width/2))
            image = F.resize(image, target_size)
            bbox = target["boxes"]
            bbox[:,[0,2]] = bbox[:,[0,2]]/width*image.size[0]
            bbox[:,[1,3]] = bbox[:,[1,3]]/height*image.size[1]
            target["boxes"] = bbox

        return image, target

class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target


def get_transform(train, resize=True, p=0.5):
    transforms = []
    if train and resize:
        transforms.append(Resize(0.5))
    transforms.append(ToTensor())  # converts [0, 255] to [0, 1]
    if train:
        transforms.append(RandomHorizontalFlip(p))
    return Compose(transforms)

