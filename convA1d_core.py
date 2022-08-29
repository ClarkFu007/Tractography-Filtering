import copy
import random
import torch
import torch.nn.functional as F
import time

import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn

torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

random.seed(42)
np.random.seed(42)
torch.cuda.manual_seed_all(42)


class MSELoss(nn.Module):
    """
       Mean squared error loss.
    """

    def __init__(self):
        super().__init__()

    def forward(self, _input, _target):
        """
           Implements mean squared error loss.
        """
        # For the reconstruction loss:
        mse_values = (_input - _target) ** 2
        #mse_values = mse_values.view(mse_values.size(0), -1)
        #mse_loss = torch.mean(mse_values, 1)
        mse_loss = torch.mean(mse_values)
        return mse_loss


class OrientationLoss(nn.Module):
    """
       Orientation Loss.
    """
    def __init__(self):
        super().__init__()

    def forward(self, _input, indices, feat_num):
        """
           Implements orientation loss.
        """
        # For the orientation loss:
        orien_loss_batch = torch.zeros((len(indices), feat_num - 2),
                                       dtype=torch.float32)
        for index_j in range(feat_num - 2):
            vector_x = _input[:, :, index_j + 1] - _input[:, :, index_j]
            vector_y = _input[:, :, index_j + 2] - _input[:, :, index_j + 1]
            dot_prdct_value = vector_x.mul(vector_y)
            dot_prdct_value = dot_prdct_value.sum(dim=1)

            norm_prdct_value = vector_x.pow(2).sum(dim=1).sqrt().mul(vector_y.pow(2).sum(dim=1).sqrt())

            orientation_value = dot_prdct_value / norm_prdct_value
            orien_loss_batch[:, index_j] = -orientation_value
        orien_loss = torch.mean(orien_loss_batch)

        return orien_loss


class LatentSpaceLoss(nn.Module):
    """
       Latent Space Loss.
    """
    def __init__(self):
        super().__init__()

    def forward(self, latent_data_list, dim_num, portion_value):
        """
           Implements latent space loss.
        """
        # For the orientation loss:
        list_num = len(latent_data_list)
        portion_num = int(dim_num * portion_value)
        latent_loss_array = torch.zeros(int((list_num + 1) * list_num / 2), dtype=torch.float32,
                                        requires_grad=True)
        s_value_list = []
        for list_i in range(list_num):
            u, s, vh = np.linalg.svd(latent_data_list[list_i].cpu().detach().numpy())
            # u, s, vh = torch.linalg.svd(latent_data_list[list_i])
            s = torch.FloatTensor(s)
            s_value_list.append(s[0: portion_num])
        loss_i = 0
        for s_list_i in range(list_num - 1):
            for s_list_j in range(s_list_i + 1, list_num):
                s_loss_value = \
                    torch.mean((s_value_list[s_list_i] - s_value_list[s_list_j]) ** 2)
                with torch.no_grad():
                    latent_loss_array[loss_i] = s_loss_value
                loss_i += 1

        latent_loss = torch.mean(latent_loss_array)

        return latent_loss


class RelationLoss(nn.Module):
    """
       Relation Loss.
    """
    def __init__(self):
        super().__init__()

    def forward(self, _input, _target, indices, percentage):
        """
           Implements relation loss.
        """
        # For the orientation loss:
        relation_loss_batch = torch.zeros((len(indices)), dtype=torch.float32)
        for index_i in range(len(indices)):
            in_similarity = torch.matmul(_input[index_i], torch.transpose(_input[index_i], 0, 1))
            zero = torch.zeros_like(in_similarity).cuda()
            value, _ = torch.topk(in_similarity.view(-1), int(list(in_similarity.view(-1).shape)[0] * percentage))
            in_similarity = torch.where(in_similarity < value[-1], zero, in_similarity)

            target_similarity = torch.matmul(_target[index_i], torch.transpose(_target[index_i], 0, 1))
            zero = torch.zeros_like(target_similarity).cuda()
            value, _ = torch.topk(target_similarity.view(-1), int(list(target_similarity.view(-1).shape)[0] * percentage))
            target_similarity = torch.where(target_similarity < value[-1], zero, target_similarity)

            relation_loss_batch[index_i] = torch.mean(torch.square(in_similarity - target_similarity))

        relation_loss = torch.mean(relation_loss_batch)

        return relation_loss


class Conv1dAutoencoder(object):
    """
       Implements tractography filtering with the 1d-conv autoencoder.
    Methods:
        __init__
        fit()
        encode_data()
    """

    def __init__(self, latent_dim_num, model_case, cuda_id, str_data_tr=None,
                 str_data_val=None, interest_data=None, auto_lr=0.0008, weight_decay=0):
        """
            Initializes the instance variables.
        traditional autoencoder: auto_lr = 0.0008, step_size=10, gamma=0.9
        """
        self.latent_dim_num = latent_dim_num  # number of dimensions of latent space
        self.model_case = model_case  # choose which model to use

        self.str_data_tr = str_data_tr  # training data
        if self.str_data_tr is not None:
            self.sample_num = str_data_tr.shape[0]  # number of samples
            self.dim_num = str_data_tr.shape[1]  # number of dimensions for each point
            self.feat_num = str_data_tr.shape[2]  # number of features (points of each streamline)

        self.str_data_val = str_data_val  # validation data

        self.itst_sample_num = None if interest_data is None else interest_data.shape[0]
        self.itst_dim_num = None if interest_data is None else interest_data.shape[1]
        self.itst_feat_num = None if interest_data is None else interest_data.shape[2]

        self.auto_lr = auto_lr  # learning rate
        self.weight_decay = weight_decay  # weight decay

        if cuda_id >= 4:
            cuda_name = 'cpu'
        else:
            cuda_name = "cuda:" + str(cuda_id)
        print(cuda_name)
        self.device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
        print("ConvA is running on", self.device)
        print("The version of PyTorch is", torch.__version__)

        """
           A toy autoencoder with 1d-convolutional layers.
        """
        # construct an autoencoder
        from models.convA1d_model.final_VAE_model import AutoencoderModel
        if self.model_case == 'final_VAE':
            self.autoencoder_model = \
                AutoencoderModel(dim_num=self.dim_num,
                                 latent_dim_num=self.latent_dim_num).to(self.device)

        if self.str_data_tr is not None:
            self.autoencoder_model = \
                AutoencoderModel(dim_num=self.dim_num,
                                 latent_dim_num=self.latent_dim_num).to(self.device)
        else:
            self.autoencoder_model = \
                AutoencoderModel(dim_num=self.itst_dim_num,
                                 latent_dim_num=self.latent_dim_num).to(self.device)

        # construct an optimizer
        self.optim_auto = torch.optim.AdamW(self.autoencoder_model.parameters(),
                                            lr=self.auto_lr,
                                            betas=(0.0, 0.999),
                                            weight_decay=self.weight_decay)
        """
        self.optim_auto = torch.optim.SGD(self.conv1d_autoencoder.parameters(),
                                           lr=self.auto_lr, momentum=0.9,
                                           weight_decay=self.weight_decay, nesterov=True)
        """

        # and a learning rate scheduler
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optim_auto,
                                                            step_size=10,
                                                            gamma=0.9)
        return

    def fit(self, epochs, rec_weight_coeff=1.0, ori_weight_coeff=0.0001, lat_weight_coeff=0.01,
            rel_weight_coeff=0.0, fine_tune=False, verbose=False):
        """
            Trains the autoencoder, if fine_tune=True uses the previous state
        of the optimizer instantiated before.
        """
        if fine_tune:
            checkpoint = torch.load("best_model.pth")
            self.autoencoder_model.load_state_dict(checkpoint["auto_state_dict"])
            self.optim_auto.load_state_dict(checkpoint["optim_auto_state_dict"])

        """
           Create criteria for the loss function.
        """
        # huber_criterion = nn.HuberLoss()
        # loss_criterion = nn.MSELoss().to(self.device)
        # sl1_criterion = nn.SmoothL1Loss(size_average=True)
        # mccr_loss = MCCRLoss(sigma=0.3).to(self.device)
        mse_loss = MSELoss().to(self.device)
        orientation_loss = OrientationLoss().to(self.device)
        loss_values = np.zeros(epochs, dtype='float')

        # start0 = time.time()
        batch_size = 3 * 433
        minimum_loss = np.inf  # Make it big enough
        self.autoencoder_model.train()

        if self.model_case == 'final_VAE':
            weights = self.autoencoder_model.state_dict()
            for name, weight in weights.items():
                if 'weight' in name and ('encoder' in name or 'decoder' in name):
                    if len(weight.shape) == 3:
                        temp_weight = weights[name][:, :, 0] 
                        weights[name][:, :, 0] = temp_weight
                        weights[name][:, :, 2] = temp_weight
            self.autoencoder_model.load_state_dict(weights)

        for epoch in range(epochs):
            t0, total_loss = time.time(), 0
            permutation = torch.randperm(self.sample_num)
            
            iter_i = 0
            for batch_i in range(0, self.sample_num, batch_size):
                indices = permutation[batch_i:batch_i + batch_size]  # <class 'torch.Tensor'>
                featT = torch.FloatTensor(self.str_data_tr)
                batch_X, batch_y = featT[indices], featT[indices]
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

                if self.model_case == 'final_VAE':
                    reconstructed_X, z_mean, z_log_var = \
                        self.autoencoder_model(X=batch_X, encode=False, reconstruct=False)

                rec_loss = mse_loss(reconstructed_X, batch_y)  # tensor(0.0844, grad_fn=<MeanBackward0>)
                orien_loss = orientation_loss(reconstructed_X, indices=indices, feat_num=self.feat_num) + 1

                if self.model_case == 'final_VAE':
                    kl_div = torch.mean((0.25 - z_mean) ** 2)
                    num_loss = rec_weight_coeff * rec_loss + ori_weight_coeff * orien_loss + \
                               lat_weight_coeff * kl_div

                self.optim_auto.zero_grad()  # Zero the gradients
                num_loss.backward()  # Calculate the gradients
                # nn.utils.clip_grad_norm_(self.autoencoder_model.parameters(), max_norm=2.0 )
                self.optim_auto.step()  # Update the weights

                # print("num_loss.item() is: %f," % (num_loss.item()))

                total_loss = total_loss + num_loss.item()
                iter_i += 1

                if self.model_case == 'final_VAE':
                    weights = self.autoencoder_model.state_dict()
                    for name, weight in weights.items():
                        if 'weight' in name and ('encoder' in name or 'decoder' in name):
                            if len(weight.shape) == 3:
                                if epoch % 2 == 1:
                                    temp_weight = weights[name][:, :, 0]
                                else:
                                    temp_weight = weights[name][:, :, 2] 
                                weights[name][:, :, 0] = temp_weight
                                weights[name][:, :, 2] = temp_weight
                    self.autoencoder_model.load_state_dict(weights)

                """
                if epoch >= 300 and self.model_case == 'case1_VAE':
                    weights = self.autoencoder_model.state_dict()
                    for name, weight in weights.items():
                        if 'weight' in name and ('encoder' in name or 'decoder' in name):
                            if len(weight.shape) == 3:
                                temp_weight = (weights[name][:, :, 0] + weights[name][:, :, 2]) / 2
                                weights[name][:, :, 0] = temp_weight
                                weights[name][:, :, 2] = temp_weight
                    self.autoencoder_model.load_state_dict(weights)
                """

            self.lr_scheduler.step()  # update the learning rate
            training_loss = total_loss
            # validation_loss = self.validate_data(interest_data=self.str_data_val)

            if verbose:
                print("Epoch is %04d/%04d | training loss is: %f "
                      % (epoch + 1, epochs, training_loss))
                """
                                print("Epoch is %04d/%04d, training loss is: %f, "
                      "validation loss is: %f" % (epoch + 1, epochs,
                                                  training_loss, validation_loss))
                """
            loss_values[epoch] = training_loss

            # if validation_loss < minimum_loss:
            if epoch >= 300:
                if training_loss < minimum_loss:
                    minimum_loss = training_loss
                    torch.save({"auto_state_dict": self.autoencoder_model.state_dict(),
                                "optim_auto_state_dict": self.optim_auto.state_dict()},
                               "best_model" + str(self.weight_decay) + ".pth")

        # end0 = time.time()
        # total_time = end0 - start0
        # print("The total training time is {:4.4f} seconds for {} epochs".format(total_time, epochs))
        # print("The mean training time is %f seconds" % (dur_train.mean()))

        return

    def encode_data(self, interest_data, model_path):
        """
           Encodes the data of interest.
        """
        # model_path = "training_logs/pretrained models/case1_dim32_regularization/best_model1.0.pth"
        if torch.cuda.is_available():
            checkpoint = torch.load(model_path)
        else:
            checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        self.autoencoder_model.load_state_dict(checkpoint["auto_state_dict"])
        self.itst_sample_num = interest_data.shape[0]
        self.itst_dim_num = interest_data.shape[1]
        self.itst_feat_num = interest_data.shape[2]
        latent_data = np.zeros((self.itst_sample_num, self.latent_dim_num), dtype='float')
        for sample_i in range(self.itst_sample_num):
            interest_data_i = interest_data[sample_i]
            interest_data_i = np.expand_dims(interest_data_i, axis=0)
            # interest_data_i = np.expand_dims(interest_data_i, axis=0)
            interest_data_i = torch.FloatTensor(interest_data_i).to(self.device)

            # Reconstruction
            with torch.no_grad():
                self.autoencoder_model.eval()
                if self.model_case == 'final_VAE':
                    encode_data_i, _, _ = self.autoencoder_model(X=interest_data_i,
                                                                 encode=True,
                                                                 reconstruct=False)

            encode_data_i = encode_data_i.cpu().detach().numpy()
            encode_data_i = np.squeeze(encode_data_i, axis=0)

            latent_data[sample_i] = copy.deepcopy(encode_data_i)

        return latent_data

    def decode_data(self, latent_data, model_path):
        """
           Decodes the latent data of interest.
        """
        if torch.cuda.is_available():
            checkpoint = torch.load(model_path)
        else:
            checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

        self.autoencoder_model.load_state_dict(checkpoint["auto_state_dict"])
        reconstructed_data = np.zeros((latent_data.shape[0], self.itst_dim_num, self.itst_feat_num), dtype='float')
        for latent_i in range(latent_data.shape[0]):
            latent_data_i = latent_data[latent_i]
            latent_data_i = np.expand_dims(latent_data_i, axis=0)
            latent_data_i = torch.FloatTensor(latent_data_i).to(self.device)

            # Reconstruction
            with torch.no_grad():
                self.autoencoder_model.eval()
                decode_data_i = self.autoencoder_model(X=latent_data_i, encode=False,
                                                       reconstruct=True)
            decode_data_i = decode_data_i.cpu().detach().numpy()
            decode_data_i = np.squeeze(decode_data_i, axis=0)

            reconstructed_data[latent_i] = copy.deepcopy(decode_data_i)

        return reconstructed_data

    def validate_data(self, interest_data):
        """
           Validates data while training.
        :param interest_data: validation data.
        :return: mse loss value.
        """
        self.itst_sample_num = interest_data.shape[0]
        self.itst_dim_num = interest_data.shape[1]
        self.itst_feat_num = interest_data.shape[2]
        total_loss = 0
        mse_loss = MSELoss().to(self.device)
        for sample_i in range(0, self.itst_sample_num, 10):
            featT_i = interest_data[sample_i]
            featT_i = np.expand_dims(featT_i, axis=0)
            featT_i = torch.FloatTensor(featT_i)
            X_i, y_i = featT_i, featT_i
            X_i, y_i = X_i.to(self.device), y_i.to(self.device)
            with torch.no_grad():
                self.conv1d_autoencoder.eval()
                iX_i = self.conv1d_autoencoder(X=X_i, encode=False, reconstruct=False)
            rec_loss = mse_loss(iX_i, y_i)
            total_loss = total_loss + rec_loss.item()

        return total_loss
