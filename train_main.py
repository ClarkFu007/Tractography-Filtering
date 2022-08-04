import os
import argparse

import numpy as np


def train_model():
    subject_list = []
    with open(args.subject_ids_dir) as f:
        for line in f:
            subject_list.append(int(line))
    tr_list, te_list = [], []
    index_value = 0
    for subject_i in subject_list:
        subject_i_dir = os.path.join(args.subject_dir, str(subject_i))
        sub_i_file = os.path.join(subject_i_dir, args.rest_dir)
        from utils import get_streamlines
        if index_value < args.subj_tr_num:
            # Training subjects
            tr_list.append(get_streamlines(filename=sub_i_file, resample_way="set",
                                           resample_num=args.resample_num))
        else:
            # Test subjects
            te_list.append(get_streamlines(filename=sub_i_file, resample_way="set",
                                           resample_num=args.resample_num))
        index_value += 1

    print("The number of training subjects is {}.".format(len(tr_list)))
    print("The number of test subjects is {}.".format(len(te_list)))

    streamlines_data_tr = tr_list[0]
    print("Initially, streamlines_data_tr.shape", streamlines_data_tr.shape)
    for index_i in range(1, args.subj_tr_num):
        streamlines_data_tr = np.vstack((streamlines_data_tr, tr_list[index_i]))
    print("Finally, streamlines_data_tr.shape", streamlines_data_tr.shape)
    print("The number of training streamlines is {}".format(streamlines_data_tr.shape[0] // args.resample_num))

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
    prepro_str_data_te1 = handle_data(prepro_str_data=prepro_str_data_te,
                                      resample_num=resample_num, mode='test')
    """
    from utils import train
    train(str_data_tr=prepro_str_data_tr, epochs=args.epoch_num, mdl_case=args.model_case,
          latent_dim=args.latent_dim_num, cuda_id=args.cuda_id)


if __name__ == '__main__':
    # Variable Space
    parser = argparse.ArgumentParser(description="Evaluate consistency measures of 200 subjects")
    # Paths
    parser.add_argument('--subject_ids_dir', type=str, default='subject_ids_final.txt',
                        help='interesting subjects to evaluate')
    parser.add_argument('--subject_dir', type=str,
                        default='/ifs/loni/faculty/shi/spectrum/Student_2020/yuanli/SWM_UFiber/HCP/for_yao',
                        help='initial path of experiment data')
    parser.add_argument('--rest_dir', type=str,
                        default='DTISpace/UFiber'
                                '/Sphere_coord_reg_lh_MotorSensory_SulcalPatch_UFiber_Angle7_FODTHD0.5_ShrinkDis0.75.trk',
                        help='/lh_MotorSensory_SulcalPatch_UFiber_Angle7_FODTHD0.5_ShrinkDis0.75.trk or '
                             '/Sphere_coord_reg_lh_MotorSensory_SulcalPatch_UFiber_Angle7_FODTHD0.5_ShrinkDis0.75.trk')
    parser.add_argument('--output_dir', type=str, default='./saved_results',
                        help='saves the results')
    parser.add_argument('--cuda_id', type=int, default=1,
                        help='id for GPUs')
    # Parameters
    parser.add_argument('--sphere_version', default=False, action='store_true',
                        help='indicates whether to handle the sphere version')
    parser.add_argument('--resample_num', type=int, default=128, help='number of points for each streamline')
    parser.add_argument('--subj_tr_num', type=int, default=200, help='number of training subjects')
    parser.add_argument('--epoch_num', type=int, default=1000, help='the number of epochs for training')
    parser.add_argument('--latent_dim_num', type=int, default=32, help='the number of latent dimensions')
    parser.add_argument('--model_case', type=str, default='final_VAE', help='which model architecture to use')

    args = parser.parse_args()
    # print(args)

    train_model()
