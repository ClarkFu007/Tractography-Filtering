import os
import copy
import pickle
import argparse

import numpy as np


def evaluate_final():
    print("Loading from ", args.filename)
    with open(args.filename, 'rb') as (pickle_file):
        filtered_tr_subj_list = pickle.load(pickle_file)
    pickle_file.close()
    print("len(filtered_tr_subj_list)", len(filtered_tr_subj_list))
    print("Begin evaluating...")
    
    consistent_measure = np.zeros(args.subj_num, dtype='float')
    for subj_i in range(args.start_index, args.start_index + args.subj_num):
        filtered_tr_subj_i = filtered_tr_subj_list[subj_i]
        subj_i_num = filtered_tr_subj_i.shape[0] * filtered_tr_subj_i.shape[1]
        subj_i_consistent_measure = get_consistent_measure(filtered_tr_subj_i=filtered_tr_subj_i,
                                                           filtered_tr_subj_list=filtered_tr_subj_list,
                                                           current_subj=subj_i,
                                                           refer_subj_num=args.refer_subj_num,
                                                           refer_str_num=args.refer_str_num)
        consistent_measure[subj_i - args.start_index] = subj_i_consistent_measure / subj_i_num
    print("For ", os.path.basename(args.rest_dir))
    print("start_index is {}, subj_num is {}, refer_subj_num is {}, refer_str_num is {}"
          .format(args.start_index, args.subj_num, args.refer_subj_num, args.refer_str_num))
    print("np.sum(consistent_measure): ", np.sum(consistent_measure))
    print("np.mean(consistent_measure): ", np.mean(consistent_measure))
    print("np.std(consistent_measure): ", np.std(consistent_measure))
    saved_name = args.subject_name + '_' + args.voxel_version + '_' + str(args.percentage_value) + '_'\
                 + str(args.start_index) + '_' + str(args.subj_num) + '_' + str(args.refer_subj_num)\
                 + '_' + str(args.refer_str_num) + '_consistent_measure.npy'
    
    saved_dir = args.subject_name + '_' + args.voxel_version + '_' + str(args.percentage_value) + '_'\
                 + str(args.refer_subj_num) + '_' + str(args.refer_str_num)
    saved_path = os.path.join(args.output_dir, saved_dir)
    if not os.path.isdir(saved_path):
         os.mkdir(saved_path)
    saved_path = os.path.join(saved_path, saved_name)
    np.save(saved_path, consistent_measure)


def get_consistent_measure(filtered_tr_subj_i, filtered_tr_subj_list,
                           current_subj, refer_subj_num, refer_str_num):
    """
       Calculates consistent measures between subjects.
    """
    """Gets the reference set"""
    str_reference_list = []  # [streamline_num, refer_subj_num, refer_str_num, 128, 3]
    total_subj_num = len(filtered_tr_subj_list)
    for index_i in range(filtered_tr_subj_i.shape[0]):
        subj_i_streamline_i = filtered_tr_subj_i[index_i]  # [128, 3]
        rest_subj_i_ref_total = np.zeros(total_subj_num, dtype='float')
        rest_subj_i_ref = np.zeros((total_subj_num, refer_str_num, 128, 3), dtype='float')
        for subj_i_index, rest_subj_i in enumerate(filtered_tr_subj_list):
            if subj_i_index != current_subj:
                dist_infor = np.zeros(rest_subj_i.shape[0], dtype='float')  # Records distance
                for index_j in range(rest_subj_i.shape[0]):
                    rest_subj_i_streamline_j = rest_subj_i[index_j]
                    dist_infor[index_j] = \
                        np.sqrt(np.sum(np.square(subj_i_streamline_i - rest_subj_i_streamline_j)))
                saved_str_indices = np.argsort(dist_infor)
                rest_subj_i_ref_total[subj_i_index] = np.sum(rest_subj_i[saved_str_indices[0: refer_str_num]])
                rest_subj_i_ref[subj_i_index] = rest_subj_i[saved_str_indices[0: refer_str_num]]

        saved_subj_indices = np.argsort(rest_subj_i_ref_total)
        str_reference_list.append(rest_subj_i_ref[saved_subj_indices[1: refer_subj_num + 1]])

    """
       implements the message passing.
    str_reference_list: [streamline_num, refer_subj_num, refer_str_num, 128, 3]
    """
    str_num, point_num = filtered_tr_subj_i.shape[0: 2]
    total_nearest_dist = 0
    for str_index, str_i_ref_set in enumerate(str_reference_list):
        for point_index in range(point_num):
            point_i = filtered_tr_subj_i[str_index, point_index]
            ref_subj_i = \
                str_i_ref_set.reshape((str_i_ref_set.shape[0]*str_i_ref_set.shape[1]*str_i_ref_set.shape[2], 3))
            temp_point_i = copy.deepcopy(np.expand_dims(point_i, axis=0))
            temp_point_i = np.repeat(temp_point_i, ref_subj_i.shape[0], axis=0)
            nearest_dist = np.min(np.sum(np.square(temp_point_i - ref_subj_i), axis=1))
                    
        total_nearest_dist += np.exp(- nearest_dist / args.sigma_value ** 2)

    return total_nearest_dist


if __name__ == '__main__':
    # Variable Space
    parser = argparse.ArgumentParser(description="Evaluate consistency measures of 200 subjects",
                                     epilog="Referenced from https://github.com/fxia22/pointnet.pytorch")
    # Paths
    parser.add_argument('--filename', type=str, default='./saved_data/motor_sensory_normal_0.7.pkl',
                        help='the file for filtered_tr_subj_list')
    parser.add_argument('--rest_dir', type=str,
                        default='DTISpace/UFiber/Sphere_coord_reg_lh_MotorSensory_SulcalPatch_UFiber_Angle7_FODTHD0.5_ShrinkDis0.75.trk',
                        help='rest path of experiment data')
    parser.add_argument('--output_dir', type=str, default='./saved_results',
                        help='saves the results')
    # evaluation params
    parser.add_argument('--start_index', type=int, default=0, help='which subject to be measured initially')
    parser.add_argument('--subj_num', type=int, default=0, help='number of subjects to be measured')
    parser.add_argument('--percentage_value', type=float, default=0.7, help='how many streamlines to keep')
    parser.add_argument('--refer_subj_num', type=int, default=10, help='number of subjects for reference')
    parser.add_argument('--refer_str_num', type=int, default=10, help='number of streamlines for reference')
    parser.add_argument('--sigma_value', type=float, default=0.9, help='sigma_value to control consistency')

    # parameters
    parser.add_argument('--subject_name', type=str, default='motor_sensory', help='name of subjects')
    parser.add_argument('--voxel_version', type=str, default='normal', help='normal or sphere')

    args = parser.parse_args()
    evaluate_final()