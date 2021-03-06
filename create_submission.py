import os
from argparse import ArgumentParser

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torchvision.transforms import Compose, Normalize, ToPILImage, ToTensor
from tqdm import tqdm
from model.model import Model


def decode(rows):
    '''
    ImageId,ClassId,EncodedPixels
    '''
    mask = np.zeros((256 * 1600), np.uint8)
    for j in range(len(rows)):
        row = rows.iloc[j]
        class_id = row['ClassId']
        encoded_pixels = np.array([int(ele)
                                   for ele in row['EncodedPixels'].split(' ')])
        starts, lengths = encoded_pixels[:: 2], encoded_pixels[1:: 2]
        starts -= 1  # 因为起始值是1，所以先要把坐标减一下
        for index, start in enumerate(starts):
            mask[int(start):int(start + lengths[index])] = class_id
    mask = mask.reshape((1600, 256)).T

    return mask

# https://www.kaggle.com/paulorzp/rle-functions-run-lenght-encode-decode


def mask2rle(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def create_submission(model, csv, root, transform, threshold):
    '''

    :param classify_splits: 分类模型的折数，类型为字典
    :param seg_splits: 分割模型的折数，类型为字典
    :param batch_size: batch的大小
    :param num_workers: 加载数据的线程
    :param mean: 均值
    :param std: 方差
    :param test_data_folder: 测试数据存放的路径
    :param sample_submission_path: 提交样例csv存放的路径
    :param model_path: 当前模型权重存放的目录
    :param tta_flag: 是否使用tta
    :param average_strategy: 是否使用平均策略
    :param kaggle: 线上或线下
    :return: None
    '''
    # 加载数据集
    df = pd.read_csv(csv)
    predictions = []
    for x in tqdm(os.listdir(root)):
        img_id = os.path.join(root, x)
        imgId = img_id.split(os.sep)[-1]
        img = cv2.imread(img_id)
        left = img[:, :800, :]
        right = img[:, 800:, :]
        left = transform(left).unsqueeze(0).to('cuda')
        right = transform(right).unsqueeze(0).to('cuda')
        left_out = torch.sigmoid(model(left))
        right_out = torch.sigmoid(model(right))
        left_out = (left_out > threshold).type(torch.float)
        right_out = (right_out > threshold).type(torch.float)
        masks = torch.cat([left_out, right_out], dim=-1)
        for i in range(1, 5):
            mask = masks[0, i]
            rle = mask2rle(mask.detach().cpu().numpy())
            predictions.append([imgId, rle, i])
    df = pd.DataFrame(predictions, columns=[
                      'ImageId', 'EncodedPixels','ClassId'])
    df.to_csv("submission.csv", index=False)


def main():
    transform = Compose([
        ToPILImage(),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225])])

    threshold = 0.5
    model = Model.load_from_checkpoint('resnet34_fpn-v0.ckpt', decoder='fpn')
    model.to('cuda')
    model.eval()
    create_submission(model, 'data/steel/sample_submission.csv',
                      'data/steel/test_images', transform, threshold)


if __name__ == '__main__':
    main()
