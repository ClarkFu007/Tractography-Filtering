import copy

import numpy as np

from dipy.io.streamline import load_tractogram, save_trk
from dipy.tracking.streamline import set_number_of_points
from dipy.io.stateful_tractogram import StatefulTractogram


def main():
    resample_num = 128
    from utils import get_streamlines
    # For training:
    filename_tr1 = "./u_fiber2/100206/lh_LateralFrontal_SulcalPatch_UFiber_Angle7_FODTHD0.5_ShrinkDis1.5.trk"
    filename_tr2 = "./u_fiber2/100408/lh_LateralFrontal_SulcalPatch_UFiber_Angle7_FODTHD0.5_ShrinkDis1.5.trk"
    filename_tr3 = "./u_fiber2/101006/lh_LateralFrontal_SulcalPatch_UFiber_Angle7_FODTHD0.5_ShrinkDis1.5.trk"
    filename_tr4 = "./u_fiber2/101309/lh_LateralFrontal_SulcalPatch_UFiber_Angle7_FODTHD0.5_ShrinkDis1.5.trk"
    streamlines_data_tr1 = get_streamlines(filename=filename_tr1, resample_way="set", resample_num=resample_num)
    streamlines_data_tr2 = get_streamlines(filename=filename_tr2, resample_way="set", resample_num=resample_num)
    streamlines_data_tr3 = get_streamlines(filename=filename_tr3, resample_way="set", resample_num=resample_num)
    streamlines_data_tr4 = get_streamlines(filename=filename_tr4, resample_way="set", resample_num=resample_num)
    streamlines_data_tr = np.vstack((streamlines_data_tr1, streamlines_data_tr2,
                                     streamlines_data_tr3, streamlines_data_tr4))
    print("After resampling, training data shape", streamlines_data_tr.shape)
    # For validation:
    filename_val1 = "./u_fiber2/102008/lh_LateralFrontal_SulcalPatch_UFiber_Angle7_FODTHD0.5_ShrinkDis1.5.trk"
    streamlines_data_val = get_streamlines(filename=filename_val1, resample_way="set",
                                           resample_num=resample_num)
    print("After resampling, validation data shape", streamlines_data_val.shape)
    # For test:
    filename_te1 = "./u_fiber2/103515/lh_LateralFrontal_SulcalPatch_UFiber_Angle7_FODTHD0.5_ShrinkDis1.5.trk"
    filename_te2 = "./u_fiber2/104820/lh_LateralFrontal_SulcalPatch_UFiber_Angle7_FODTHD0.5_ShrinkDis1.5.trk"
    streamlines_data_te1 = get_streamlines(filename=filename_te1, resample_way="set",
                                           resample_num=resample_num)
    streamlines_data_te2 = get_streamlines(filename=filename_te2, resample_way="set",
                                           resample_num=resample_num)

    from utils import preprocess_data
    prepro_str_data_tr, prepro_str_data_val, \
    prepro_str_data_te1, minmax_scaler = preprocess_data(tr_set=streamlines_data_tr,
                                                         val_set=streamlines_data_val,
                                                         te_set=streamlines_data_te1)
    prepro_str_data_te2 = minmax_scaler.transform(streamlines_data_te2)

    from utils import handle_data
    # Training data
    prepro_str_data_tr = handle_data(prepro_str_data=prepro_str_data_tr,
                                     resample_num=resample_num, mode='training')
    # Validation data
    prepro_str_data_val = handle_data(prepro_str_data=prepro_str_data_val,
                                      resample_num=resample_num, mode='validation')
    # Test data
    prepro_str_data_te1 = handle_data(prepro_str_data=prepro_str_data_te1,
                                      resample_num=resample_num, mode='test')
    prepro_str_data_te2 = handle_data(prepro_str_data=prepro_str_data_te2,
                                      resample_num=resample_num, mode='test')

    rcst_str_data = np.swapaxes(prepro_str_data_tr[0:4448], axis1=1, axis2=2)
    rcst_str_data = np.reshape(rcst_str_data, (rcst_str_data.shape[0] * rcst_str_data.shape[1], -1))
    print(rcst_str_data.shape)

    outlier_num = 10
    outlier_srt = np.zeros((outlier_num * resample_num, 3), dtype='float')
    """
        for coord_i in range(3):
        np.random.seed(coord_i)
        # outlier_srt[:, coord_i] = np.random.uniform(low=0.4, high=0.6, size=outlier_num * resample_num)
        # outlier_srt[:, coord_i] = np.random.normal(loc=0.5, scale=0.15, size=outlier_num * resample_num)
    """
    coord_value = 0
    step_value = 1.0 / outlier_srt.shape[0]
    for point_i in range(outlier_srt.shape[0]):
        np.random.seed(point_i)
        # outlier_srt[point_i] = np.random.uniform(low=0.2, high=0.3, size=3)
        #outlier_srt[point_i] = np.random.normal(loc=0.5, scale=0.1, size=3)
        outlier_srt[point_i, 0] = np.random.uniform(low=0.3, high=0.4, size=1)
        outlier_srt[point_i, 1] = np.random.normal(loc=0.5, scale=0.1, size=1)
        outlier_srt[point_i, 2] = coord_value
        coord_value += step_value

    rcst_str_data = np.vstack((rcst_str_data, outlier_srt))
    print(rcst_str_data.shape)

    rcst_str_data = minmax_scaler.inverse_transform(X=rcst_str_data)
    rcst_str_data = np.vstack((rcst_str_data, outlier_srt))
    rcst_str_data = np.reshape(rcst_str_data, (int(rcst_str_data.shape[0] / resample_num),
                                               resample_num, -1))
    print(rcst_str_data.shape)
    print(np.min(rcst_str_data))
    print(np.max(rcst_str_data))

    reference_file = load_tractogram(filename=filename_tr1, reference='same', bbox_valid_check=False)
    final_tractogram = StatefulTractogram(streamlines=rcst_str_data,
                                          reference=reference_file,
                                          space=reference_file.space)
    save_trk(sft=final_tractogram, filename='rcst_str_data.trk', bbox_valid_check=True)


if __name__ == '__main__':
    main()