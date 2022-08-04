import copy
import numpy as np


def main():
    # reconstruct_data
    if False: 
        from utils import reconstruct_data
        latent_dim_num = 32
        model_path = 'best_model0.pth'
        model_case = 'case1_VAE'
        reconstruct_data(str_data=prepro_str_data_tr[0:2568], scaler=minmax_scaler,
                         resample_size=resample_num, latent_dim=latent_dim_num,
                         mdl_case=model_case, filename=filename_tr1, case='tr1',
                         model_path=model_path, to_save=True, to_evaluate=False)

    # get_str_cluster
    if False:
        from utils import get_str_cluster
        smallest_num = 5
        biggest_num = 5
        step_size = 20
        method = 'k-means clustering'
        #method = 'k-medoids clustering'
        get_str_cluster(smallest_num=smallest_num, biggest_num=biggest_num, step_size=step_size,
                        prepro_data=prepro_str_data_tr[0:2568], latent_dim=latent_dim_num,
                        scaler=minmax_scaler, resample_size=resample_num, model_path=model_path,
                        mdl_case=model_case, method=method, filename=filename_tr1)

    # do_cluster_filtering
    if False:
        from utils import do_cluster_filtering
        percentage_value = 0.95
        smallest_num = 5
        biggest_num = 5
        step_size = 20
        do_cluster_filtering(smallest_num=smallest_num, biggest_num=biggest_num, step_size=step_size,
                             prepro_data=prepro_str_data_tr[0:2568], latent_dim=latent_dim_num,
                             percentage_value=percentage_value, resample_size=resample_num,
                             model_path=model_path, mdl_case=model_case, filename=filename_tr1)

    # do_cluster_segmentation
    if False:
        from utils import do_cluster_segmentation
        smallest_num = 5
        biggest_num = 5
        step_size = 5
        do_cluster_segmentation(smallest_num=smallest_num, biggest_num=biggest_num, step_size=step_size,
                                prepro_data=prepro_str_data_tr[0:2568], latent_dim=latent_dim_num,
                                resample_size=resample_num, mdl_case=model_case, filename=filename_tr1,
                                model_path=model_path)

if __name__ == '__main__':
    main()