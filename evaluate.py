import os
import copy
import pickle
import argparse

import numpy as np

from utils import get_streamlines


def evaluate_main():
    subject_list = []
    with open(args.subject_ids_dir) as f:
        for line in f:
            subject_list.append(int(line))
    tr_list, te_list = [], []
    subj_name_list_tr = []
    orig_subj_name_list_tr = [] if args.sphere_version else None
    index_value = 0
    for subject_i in subject_list:
        subject_i_dir = os.path.join(args.subject_dir, str(subject_i))
        sub_i_file = os.path.join(subject_i_dir, args.rest_dir)
        
        temp_streamlines = get_streamlines(filename=sub_i_file, resample_way="set",
                                           resample_num=args.resample_num)
        if index_value < args.subj_tr_num:
            tr_list.append(temp_streamlines)
            subj_name_list_tr.append(sub_i_file)
            if args.sphere_version:
                orig_sub_i_file = os.path.join(subject_i_dir, args.orig_rest_dir)
                orig_subj_name_list_tr.append(orig_sub_i_file)
        else:
            te_list.append(temp_streamlines)
        index_value += 1

    print("The number of training subjects is {}.".format(len(tr_list)))
    print("The number of test subjects is {}.".format(len(te_list)))

    streamlines_data_tr = tr_list[0]
    subj_num_list_tr = [int(len(tr_list[0]) / args.resample_num)]
    print("Initially, streamlines_data_tr.shape", streamlines_data_tr.shape)
    for index_i in range(1, args.subj_tr_num):
        streamlines_data_tr = np.vstack((streamlines_data_tr, tr_list[index_i]))
        subj_num_list_tr.append(int(len(tr_list[index_i]) / args.resample_num))
    print("Finally, streamlines_data_tr.shape", streamlines_data_tr.shape)
    print("The number of training streamlines is {}".format(streamlines_data_tr.shape[0] // args.resample_num))
    if np.isnan(np.sum(streamlines_data_tr)):
        print("Here!")

    from utils import preprocess_data
    if args.sphere_version:
        from utils import get_sphere_data
        tan_str_data_tr = get_sphere_data(str_data=streamlines_data_tr)
        prepro_str_data_tr, minmax_scaler = preprocess_data(tr_set=streamlines_data_tr)
    else:
        prepro_str_data_tr, minmax_scaler = preprocess_data(tr_set=streamlines_data_tr)
    # prepro_str_data_te = minmax_scaler.transform(streamlines_data_te)

    from utils import handle_data
    # Training data
    prepro_str_data_tr = handle_data(prepro_str_data=prepro_str_data_tr,
                                     resample_num=args.resample_num, mode='training')
    # Test data
    """
    prepro_str_data_te1 = handle_data(prepro_str_data=prepro_str_data_te1,
                                      resample_num=resample_num, mode='test')
    """
    from utils import cluster_filtering
    last_subj_i_num = 0
    filtered_tr_subj_list = []
    assert len(subj_num_list_tr) == len(subj_name_list_tr)
    for index_i in range(args.subj_tr_num):
        subj_i_num = subj_num_list_tr[index_i]
        if args.sphere_version:
            subj_i_file = orig_subj_name_list_tr[index_i]
        else:
            subj_i_file = subj_name_list_tr[index_i]
        filtered_tr_subj_list.append(cluster_filtering(cluster_num=args.cluster_num,
                                                       prepro_data=prepro_str_data_tr[last_subj_i_num:last_subj_i_num + subj_i_num],
                                                       cuda_id=args.cuda_id,
                                                       latent_dim=args.latent_dim_num,
                                                       percentage_value=args.percentage_value,
                                                       resample_size=args.resample_num,
                                                       model_path=args.model_path,
                                                       mdl_case=args.model_case, filename=subj_i_file))
        last_subj_i_num += subj_i_num
    
    print("The process of filtering is done!")
    filename = args.subject_name + '_' + args.voxel_version + '_' + str(args.percentage_value)
    filename = os.path.join('./saved_data', filename)
    with open(filename + '.pkl', 'wb') as (pickle_file):
        pickle.dump(filtered_tr_subj_list, pickle_file)
    pickle_file.close()
    print("The data has been saved as {}".format(filename))


if __name__ == '__main__':
    # Variable Space
    parser = argparse.ArgumentParser(description="Evaluate consistency measures of 200 subjects")
    # Paths
    parser.add_argument('--subject_ids_dir', type=str, default='subject_ids_final.txt',
                        help='interesting subjects to evaluate')
    parser.add_argument('--subject_dir', type=str,
                        default='/ifs/loni/faculty/shi/spectrum/Student_2020/yuanli/SWM_UFiber/HCP/for_yao',
                        help='initial path of experiment data')
    parser.add_argument('--orig_rest_dir', type=str,
                        default='DTISpace/UFiber/lh_MotorSensory_SulcalPatch_UFiber_Angle7_FODTHD0.5_ShrinkDis0.75.trk',
                        help='original rest path of experiment data')
    parser.add_argument('--rest_dir', type=str,
                        default='DTISpace/UFiber/Sphere_coord_reg_lh_MotorSensory_SulcalPatch_UFiber_Angle7_FODTHD0.5_ShrinkDis0.75.trk',
                        help='rest path of experiment data')
    parser.add_argument('--model_path', type=str,
                        default='/ifs/loni/faculty/shi/spectrum/Student_2020/yfu/nicr/autoencoder/saved_models/MS_normal_best_model.pth',
                        help='path of trained models')
    parser.add_argument('--output_dir', type=str, default='./saved_results',
                        help='saves the results')
    parser.add_argument('--cuda_id', type=int, default=1,
                        help='id for GPUs')

    # evaluation params
    parser.add_argument('--start_index', type=int, default=0, help='which subject to be measured initially')
    parser.add_argument('--subj_num', type=int, default=5, help='number of subjects to be measured')
    parser.add_argument('--cluster_num', type=int, default=100, help='number of clusters for K-means')
    parser.add_argument('--percentage_value', type=float, default=0.9, help='how many streamlines to keep')
    parser.add_argument('--refer_subj_num', type=int, default=10, help='number of subjects for reference')
    parser.add_argument('--refer_str_num', type=int, default=10, help='number of streamlines for reference')
    parser.add_argument('--sigma_value', type=float, default=0.9, help='sigma_value to control consistency')

    # parameters
    parser.add_argument('--subject_name', type=str, default='motor_sensory', help='name of subjects')
    parser.add_argument('--voxel_version', type=str, default='normal', help='normal or sphere')
    parser.add_argument('--sphere_version', default=False, action='store_true',
                        help='indicates whether to handle the sphere version')
    parser.add_argument('--resample_num', type=int, default=128, help='number of points for each streamline')
    parser.add_argument('--subj_tr_num', type=int, default=200, help='number of training subjects')
    parser.add_argument('--latent_dim_num', type=int, default=32, help='the number of latent dimensions')
    parser.add_argument('--model_case', type=str, default='final_VAE', help='which model architecture to use')

    args = parser.parse_args()
    #print(args)
    evaluate_main()