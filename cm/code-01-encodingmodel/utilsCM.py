import os
from PIL import Image
import numpy as np
import scipy.io
import torch
from torchvision.transforms import functional as tr
from sklearn.linear_model import Ridge

#Modified by Cate 11 August 2020, Subject function to make it fit with local directory folder name

# ImageNet mean and standard deviation. All images
# passed to a PyTorch pre-trained model (e.g. AlexNet) must be
# normalized by these quantities, because that is how they were trained.
imagenet_mean = (0.485, 0.456, 0.406)
imagenet_std = (0.229, 0.224, 0.225)


def listdir(dir, path=True):
    files = os.listdir(dir)
    files = [f for f in files if f != '.DS_Store']
    files = sorted(files)
    if path:
        files = [os.path.join(dir, f) for f in files]
    return files


def image_to_tensor(image, resolution=None, do_imagenet_norm=True):
    if isinstance(image, str):
        image = Image.open(image).convert('RGB')
    if resolution is not None:
        image = tr.resize(image, resolution)
    if image.width != image.height:     # if not square image, crop the long side's edges
        r = min(image.width, image.height)
        image = tr.center_crop(image, (r, r))
    image = tr.to_tensor(image)
    if do_imagenet_norm:
        image = imagenet_norm(image)
    return image


def tensor_to_image(image, do_imagenet_unnorm=True):
    if do_imagenet_unnorm:
        image = imagenet_unnorm(image)
    image = tr.to_pil_image(image)
    return image


def imagenet_norm(image):
    dims = len(image.shape)
    if dims < 4:
        image = [image]
    image = [tr.normalize(img, mean=imagenet_mean, std=imagenet_std) for img in image]
    image = torch.stack(image, dim=0)
    if dims < 4:
        image = image.squeeze(0)
    return image


def imagenet_unnorm(image):
    mean = torch.tensor(imagenet_mean, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(imagenet_std, dtype=torch.float32).view(3, 1, 1)
    image = image.cpu()
    image = image * std + mean
    return image


class Subject:

    def __init__(self, subject_num, rois):
        roistack = scipy.io.loadmat('data-object2vec/fmri/subj{:03}'.format(subject_num) +
                                    '/roistack.mat')['roistack']

        self.roi_names = [d[0] for d in roistack['rois'][0, 0][:, 0]]
        self.conditions = [d[0] for d in roistack['conds'][0, 0][:, 0]]

        roi_indices = roistack['indices'][0, 0][0]
        roi_masks = {roi: roi_indices == (i + 1) for i, roi in enumerate(self.roi_names)}
        voxels = roistack['betas'][0, 0]
        self.condition_voxels = {cond: np.concatenate([voxels[i][roi_masks[r]] for r in rois])
                                 for i, cond in enumerate(self.conditions)}
        self.n_voxels = np.sum([roi_masks[r] for r in rois])

        sets = scipy.io.loadmat('data-object2vec/fmri/subj{:03}'.format(subject_num) +
                                   '/sets.mat')['sets']
        self.cv_sets = [[cond[0] for cond in s[:, 0]] for s in sets[0, :]]


def cv_regression(condition_features, subject, l2=0.0):
    # Get cross-validated mean test set correlation
    rs = []
    for test_conditions in subject.cv_sets:
        train_conditions = [c for c in subject.conditions if c not in test_conditions]
        train_features = np.stack([condition_features[c] for c in train_conditions])
        test_features = np.stack([condition_features[c] for c in test_conditions])
        train_voxels = np.stack([subject.condition_voxels[c] for c in train_conditions])
        test_voxels = np.stack([subject.condition_voxels[c] for c in test_conditions])
        _, r = regression(train_features, train_voxels, test_features, test_voxels, l2=l2)
        rs.append(r)
    mean_r = np.mean(rs)

    # Train on all of the data
    train_conditions = subject.conditions
    train_features = np.stack([condition_features[c] for c in train_conditions])
    train_voxels = np.stack([subject.condition_voxels[c] for c in train_conditions])
    weights = regression(train_features, train_voxels, None, None, l2=l2, validate=False)

    return weights, mean_r


def regression(x_train, y_train, x_test, y_test, l2=0.0, validate=True):
    regr = Ridge(alpha=l2, fit_intercept=False)
    regr.fit(x_train, y_train)
    weights = regr.coef_
    if validate:
        y_pred = regr.predict(x_test)
        r = correlation(y_test, y_pred)
        return weights, r
    else:
        return weights


def correlation(a, b):
    zs = lambda v: (v - v.mean(0)) / v.std(0)
    r = (zs(a) * zs(b)).mean()
    return r
