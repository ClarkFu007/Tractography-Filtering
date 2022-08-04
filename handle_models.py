import copy
import torch
import numpy as np


def main():
    checkpoint1 = torch.load("saved_models/best_model_1_0.001_0.00001.pth", map_location=torch.device('cpu'))
    checkpoint1_infor = checkpoint["auto_state_dict"]
    checkpoint2 = torch.load("saved_models/best_single_model_tr3.pth", map_location=torch.device('cpu'))
    checkpoint2_infor = checkpoint["auto_state_dict"]

    checkpoint1_infor['linear_nn2.weight'] = checkpoint2_infor['linear_nn2.weight']
    checkpoint1_infor['linear_nn2.bias'] = checkpoint2_infor['linear_nn2.bias']
    checkpoint1_infor['up1.weight'] = checkpoint2_infor['up1.weight']
    checkpoint1_infor['up1.bias'] = checkpoint2_infor['up1.bias']
    checkpoint1_infor['up_batch_norm1.weight'] = checkpoint2_infor['up_batch_norm1.weight']
    checkpoint1_infor['up_batch_norm1.bias'] = checkpoint2_infor['up_batch_norm1.bias']
    checkpoint1_infor['up2.weight'] = checkpoint2_infor['up2.weight']
    checkpoint1_infor['up2.bias'] = checkpoint2_infor['up2.bias']
    checkpoint1_infor['up_batch_norm2.weight'] = checkpoint2_infor['up_batch_norm2.weight']
    checkpoint1_infor['up_batch_norm2.bias'] = checkpoint2_infor['up_batch_norm2.bias']
    checkpoint1_infor['up3.weight'] = checkpoint2_infor['up3.weight']
    checkpoint1_infor['up3.bias'] = checkpoint2_infor['up3.bias']


if __name__ == '__main__':
    main()