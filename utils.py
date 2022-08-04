import copy
import os

import numpy as np

from convA1d_core import Conv1dAutoencoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
# from sklearn_extra.cluster import KMedoids
from dipy.io.streamline import load_tractogram, save_trk
from dipy.tracking.streamline import set_number_of_points
from dipy.io.stateful_tractogram import StatefulTractogram


def get_streamlines(filename, resample_way, resample_num, verbose=False):
    """
       Get data of streamlines of the resampling version.
    :param filename: name of the file
    :param resample_way: max, mean, median, min, or set
    :param resample_num: number of points after resampling
    :param verbose: whether to print statements
    :return: upsampled streamline data
    """

    tractogram_file = load_tractogram(filename=filename, reference='same', bbox_valid_check=False)
    streamlines = tractogram_file.streamlines
    if verbose:
        print("Before resampling, total_nb_rows", streamlines.total_nb_rows)
        print("Before resampling, streamlines._lengths", streamlines._lengths)
        print("Before resampling, streamlines._offsets", streamlines._offsets)

    # Resample to make sure our streamlines have the same number of points.
    if resample_way == 'max':
        new_streamlines = set_number_of_points(streamlines, nb_points=np.max(streamlines._lengths))
    elif resample_way == 'mean':
        new_streamlines = set_number_of_points(streamlines, nb_points=int(np.mean(streamlines._lengths)))
    elif resample_way == 'median':
        new_streamlines = set_number_of_points(streamlines, nb_points=np.median(streamlines._lengths))
    elif resample_way == 'min':
        new_streamlines = set_number_of_points(streamlines, nb_points=np.min(streamlines._lengths))
    elif resample_way == 'set':
        new_streamlines = set_number_of_points(streamlines, nb_points=resample_num)
    else:
        raise SystemExit("Wrong! The resample_way argument has to be max, mean, median, min or set!")
    if verbose:
        print("After resampling, total_nb_rows", new_streamlines.total_nb_rows)
        print("After resampling, streamlines._lengths", new_streamlines._lengths)
        print("After resampling, streamlines._offsets", new_streamlines._offsets)

    """
    tractogram_file.streamlines = copy.deepcopy(new_streamlines)
    save_trk(sft=tractogram_file, filename='new_tractogram_file.trk', bbox_valid_check=True)
    """

    new_data = new_streamlines.get_data()
    print("new_data.shape", new_data.shape) if verbose else None
    if False:
        new_data = np.reshape(new_data, (int(new_data.shape[0] / resample_num),
                                         resample_num, -1))
        reference_file = load_tractogram(filename="Motor_CC_LH.trk", reference='same', bbox_valid_check=False)
        final_tractogram = StatefulTractogram(streamlines=new_data,
                                              reference=reference_file,
                                              space=reference_file.space)
        save_trk(sft=final_tractogram, filename='my_new_tractogram_file.trk', bbox_valid_check=True)

    return new_data


def preprocess_data(tr_set):  # Pass by reference!
    """
       Preprocesses the streamline data for training.
    :param tr_set: streamline data for training
    :return: tr_set0, minmax_scaler
    """
    tr_set0 = copy.deepcopy(tr_set)
    minmax_scaler = MinMaxScaler(feature_range=(0, 1))
    minmax_scaler.fit(tr_set0)
    tr_set0 = minmax_scaler.transform(tr_set0)

    return tr_set0, minmax_scaler


def get_sphere_data(str_data):
    """
       Gets sphere streamline data with 2 dimensions.
    :param str_data: streamline data to be converted
    :return: tan_str_data
    """
    tan_str_data = np.zeros((str_data.shape[0], 2), dtype='float')
    tan_str_data[:, 0] = np.arctan(str_data[:, 1] / str_data[:, 0])
    tan_str_data[:, 1] = np.arctan(str_data[:, 2] / np.sqrt(np.square(str_data[:, 0]) +
                                                            np.square(str_data[:, 1])))
    return tan_str_data


def preprocess_sphere_data(tr_set, val_set, te_set):  # Pass by reference!
    """
       Preprocesses the sphere streamline data for training,
    validation, and test.
    :param tr_set: streamline data for training
    :param val_set: streamline data for validation
    :param te_set: streamline data for test
    :return: tr_set0, val_set0, te_set0, minmax_scaler
    """
    tr_set0 = copy.deepcopy(tr_set)
    val_set0 = copy.deepcopy(val_set)
    te_set0 = copy.deepcopy(te_set)
    max_abs_xyz = np.max(np.absolute(tr_set), axis=0)
    tr_set0 = tr_set0 / max_abs_xyz
    val_set0 = val_set0 / max_abs_xyz
    te_set0 = te_set0 / max_abs_xyz

    if False:
        print("np.max(tr_set)", np.max(tr_set, axis=0))
        print("np.max(tr_set0)", np.max(tr_set0, axis=0))
        print("np.min(tr_set)", np.min(tr_set, axis=0))
        print("np.min(tr_set0)", np.min(tr_set0, axis=0))

        print("np.max(val_set)", np.max(val_set, axis=0))
        print("np.max(val_set0)", np.max(val_set0, axis=0))
        print("np.min(val_set)", np.min(val_set, axis=0))
        print("np.min(val_set0)", np.min(val_set0, axis=0))

        print("np.max(te_set)", np.max(te_set, axis=0))
        print("np.max(te_set0)", np.max(te_set0, axis=0))
        print("np.min(te_set0)", np.min(te_set, axis=0))
        print("np.min(te_set0)", np.min(te_set0, axis=0))

    return tr_set0, val_set0, te_set0, max_abs_xyz


def handle_data(prepro_str_data, resample_num, mode, verbose=False):
    """
       Reshapes the dataset and swaps their axes.
    :param prepro_str_data: data to be handled
    :param resample_num: number of points after resampling
    :param mode: training, validation, and test
    :param verbose: whether to print statements
    :return: data after being handled
    """
    print("For {}, prepro_str_data.shape is {}."
          .format(mode, prepro_str_data.shape)) if verbose else None
    prepro_str_data = np.reshape(prepro_str_data, (int(prepro_str_data.shape[0] / resample_num),
                                                   resample_num, prepro_str_data.shape[1]))
    print("For {}, after reshaping, prepro_str_data.shape is {}."
          .format(mode, prepro_str_data.shape)) if verbose else None
    prepro_str_data = np.swapaxes(prepro_str_data, axis1=1, axis2=2)
    print("For {}, after swapping, prepro_str_data.shape is {}"
          .format(mode, prepro_str_data.shape)) if verbose else None
    return prepro_str_data


def train(str_data_tr, epochs, mdl_case, latent_dim, cuda_id, 
          decay_lambda=0):
    """
       Trains the autoencoder.
    """
    my_model = Conv1dAutoencoder(str_data_tr=str_data_tr, str_data_val=None,
                                 latent_dim_num=latent_dim, model_case=mdl_case,
                                 cuda_id=cuda_id, weight_decay=decay_lambda)
    my_model.fit(epochs=epochs, verbose=True)

    return


def get_latent_clusters(str_data, latent_dim, model_path, cluster_num, mdl_case, method):
    """
       Gets the latent clusters.
    """
    my_model = Conv1dAutoencoder(latent_dim_num=latent_dim, model_case=mdl_case, interest_data=str_data)
    latent_str_data = my_model.encode_data(interest_data=str_data, model_path=model_path)
    print("latent_str_data.shape", latent_str_data.shape)
    if method == 'k-means clustering':
        result = KMeans(n_clusters=cluster_num, max_iter=300, random_state=66).fit(X=latent_str_data)
    elif method == 'k-medoids clustering':
        result = KMedoids(n_clusters=cluster_num, max_iter=300, random_state=66).fit(X=latent_str_data)

    return result.cluster_centers_


def get_final_cluster(str_data, latent_dim, model_path, raw_clusters, mdl_case):
    """
       Gets final reconstructed clusters.
    """
    my_model = Conv1dAutoencoder(latent_dim_num=latent_dim, model_case=mdl_case, interest_data=str_data)
    str_cluster_data = my_model.decode_data(latent_data=raw_clusters, model_path=model_path,)

    return str_cluster_data


def get_str_cluster(smallest_num, biggest_num, step_size, prepro_data, latent_dim,
                    scaler, resample_size, model_path, mdl_case, method, filename):
    """
       Saves final reconstructed clusters as .trk files.
    """
    for cluster_i in range(smallest_num, biggest_num + step_size, step_size):
        latent_clusters = get_latent_clusters(str_data=prepro_data, latent_dim=latent_dim, model_path=model_path,
                                              cluster_num=cluster_i, mdl_case=mdl_case, method=method)
        print("latent_clusters.shape", latent_clusters.shape)

        final_clusters = get_final_cluster(str_data=prepro_data, latent_dim=latent_dim, model_path=model_path,
                                           raw_clusters=latent_clusters, mdl_case=mdl_case)
        print("final_clusters.shape", final_clusters.shape)
        final_clusters = np.swapaxes(final_clusters, axis1=1, axis2=2)
        final_clusters = np.reshape(final_clusters, (final_clusters.shape[0] * final_clusters.shape[1], -1))
        final_clusters = scaler.inverse_transform(X=final_clusters)
        final_clusters = np.reshape(final_clusters, (int(final_clusters.shape[0] / resample_size),
                                                     resample_size, -1))
        print("final_clusters.shape", final_clusters.shape)
        reference_file = load_tractogram(filename=filename, reference='same', bbox_valid_check=False)
        final_tractogram = StatefulTractogram(streamlines=final_clusters,
                                              reference=reference_file,
                                              space=reference_file.space)
        save_trk(sft=final_tractogram, filename=method + os.path.basename(filename) + 'clusters_' +
                                                str(cluster_i) + '.trk',
                 bbox_valid_check=True)


def do_cluster_filtering(smallest_num, biggest_num, step_size, prepro_data,
                         latent_dim, percentage_value, resample_size, model_path,
                         mdl_case, filename):
    """
       Utilizes the latent space to do tractography filtering by some clustering techniques.
    """
    for cluster_i in range(smallest_num, biggest_num + step_size, step_size):
        my_model = Conv1dAutoencoder(latent_dim_num=latent_dim, model_case=mdl_case, interest_data=prepro_data)
        latent_str_data = my_model.encode_data(interest_data=prepro_data, model_path=model_path,)  # (10950, 32) (total_num, latent_dim)
        kmeans_result = KMeans(n_clusters=cluster_i, max_iter=300, random_state=66).fit(X=latent_str_data)
        final_clusters = kmeans_result.cluster_centers_  # (30, 32) (cluster_num, latent_dim)
        str_data_labels = kmeans_result.labels_  # (10950,) (cluster_num, )

        each_label_num = {}
        for i in range(cluster_i):
            each_label_num[i] = 0
        for i in range(str_data_labels.shape[0]):
            each_label_num[str_data_labels[i]] += 1
        print("")
        print("each_label_num", each_label_num)

        distance_infor = []
        for i in range(cluster_i):
            distance_data_i = np.zeros(each_label_num[i], dtype={'names': ('index', 'distance'),
                                                                 'formats': ('i4', 'f8')})
            distance_infor.append(distance_data_i)

        for index_i in range(latent_str_data.shape[0]):
            label_i = str_data_labels[index_i]
            cluster_ii = final_clusters[label_i]
            latent_data_i = latent_str_data[index_i]
            distance_i = np.sqrt(np.sum(np.square(latent_data_i - cluster_ii)))
            index_j = 0
            while True:
                if distance_infor[label_i][index_j]['distance'] == 0:
                    distance_infor[label_i][index_j]['index'] = index_i
                    distance_infor[label_i][index_j]['distance'] = distance_i
                    break
                else:
                    index_j += 1

        remaining_str = []
        for label_i in range(len(distance_infor)):
            label_i_num = each_label_num[label_i]
            saved_i_num = int(label_i_num * percentage_value)
            distance_infor_i = copy.deepcopy(distance_infor[label_i])
            sort_i = np.sort(distance_infor_i['distance'])
            max_dist_i = sort_i[saved_i_num]
            saved_i_indices = distance_infor_i[distance_infor_i['distance'] <= max_dist_i]['index']
            remaining_str.append(saved_i_indices)

        remaining_str_index = []
        for label_i in range(len(remaining_str)):
            for index_i in range(remaining_str[label_i].shape[0]):
                remaining_str_index.append(remaining_str[label_i][index_i])
        print(len(remaining_str_index))

        tractogram_file = load_tractogram(filename=filename, reference='same', bbox_valid_check=False)
        streamlines = tractogram_file.streamlines
        new_streamlines = set_number_of_points(streamlines, nb_points=resample_size)
        new_data = new_streamlines.get_data()
        new_data = np.reshape(new_data, (int(new_data.shape[0] / resample_size),
                                         resample_size, -1))
        print("new_data.shape", new_data.shape)
        final_data = copy.deepcopy(new_data[remaining_str_index, :, :])
        print("final_data.shape", final_data.shape)
        reference_file = load_tractogram(filename=filename, reference='same', bbox_valid_check=False)
        final_tractogram = StatefulTractogram(streamlines=final_data,
                                              reference=reference_file,
                                              space=reference_file.space)
        """
        save_trk(sft=final_tractogram,
                 filename='filtered_' + str(cluster_i) + '_'
                          + str(percentage_value) + '_tractogram_file.trk',
                 bbox_valid_check=True)

        """
        save_trk(sft=final_tractogram,
                 filename='filtered_' + str(cluster_i) + '_'
                          + str(percentage_value) + '_tractogram_file.trk',
                 bbox_valid_check=False)


def cluster_filtering(cluster_num, prepro_data, cuda_id, latent_dim, percentage_value, resample_size,
                      model_path, mdl_case, filename):
    """
       Utilizes the latent space to do tractography filtering by some clustering techniques.
    """
    my_model = Conv1dAutoencoder(latent_dim_num=latent_dim, model_case=mdl_case, 
                                 interest_data=prepro_data, cuda_id=cuda_id)
    latent_str_data = my_model.encode_data(interest_data=prepro_data,
                                           model_path=model_path)  # (10950, 32) (total_num, latent_dim)
    kmeans_result = KMeans(n_clusters=cluster_num, max_iter=300, random_state=66).fit(X=latent_str_data)
    final_clusters = kmeans_result.cluster_centers_  # (30, 32) (cluster_num, latent_dim)
    str_data_labels = kmeans_result.labels_  # (10950,) (cluster_num, )

    each_label_num = {}
    for i in range(cluster_num):
        each_label_num[i] = 0
    for i in range(str_data_labels.shape[0]):
        each_label_num[str_data_labels[i]] += 1
    print("")
    #print("each_label_num", each_label_num)

    distance_infor = []
    for i in range(cluster_num):
        distance_data_i = np.zeros(each_label_num[i], dtype={'names': ('index', 'distance'),
                                                             'formats': ('i4', 'f8')})
        distance_infor.append(distance_data_i)

    for index_i in range(latent_str_data.shape[0]):
        label_i = str_data_labels[index_i]
        cluster_ii = final_clusters[label_i]
        latent_data_i = latent_str_data[index_i]
        distance_i = np.sqrt(np.sum(np.square(latent_data_i - cluster_ii)))
        index_j = 0
        while True:
            if distance_infor[label_i][index_j]['distance'] == 0:
                distance_infor[label_i][index_j]['index'] = index_i
                distance_infor[label_i][index_j]['distance'] = distance_i
                break
            else:
                index_j += 1

    remaining_str = []
    for label_i in range(len(distance_infor)):
        label_i_num = each_label_num[label_i]
        saved_i_num = int(label_i_num * percentage_value)
        distance_infor_i = copy.deepcopy(distance_infor[label_i])
        sort_i = np.sort(distance_infor_i['distance'])
        max_dist_i = sort_i[saved_i_num]
        saved_i_indices = distance_infor_i[distance_infor_i['distance'] <= max_dist_i]['index']
        remaining_str.append(saved_i_indices)

    remaining_str_index = []
    for label_i in range(len(remaining_str)):
        for index_i in range(remaining_str[label_i].shape[0]):
            remaining_str_index.append(remaining_str[label_i][index_i])
    print(len(remaining_str_index))

    tractogram_file = load_tractogram(filename=filename, reference='same', bbox_valid_check=False)
    streamlines = tractogram_file.streamlines
    new_streamlines = set_number_of_points(streamlines, nb_points=resample_size)
    new_data = new_streamlines.get_data()
    new_data = np.reshape(new_data, (int(new_data.shape[0] / resample_size),
                                     resample_size, -1))
    print("new_data.shape", new_data.shape)
    filtered_data = copy.deepcopy(new_data[remaining_str_index, :, :])
    print("filtered_data.shape", filtered_data.shape)
    return filtered_data


def do_cluster_segmentation(smallest_num, biggest_num, step_size, prepro_data,
                            latent_dim, resample_size, mdl_case, filename, model_path):
    """
       Utilizes the latent space to do tractography segmentation by some clustering techniques.
    """
    for cluster_i in range(smallest_num, biggest_num + step_size, step_size):
        my_model = Conv1dAutoencoder(latent_dim_num=latent_dim, model_case=mdl_case, interest_data=prepro_data)
        latent_str_data = my_model.encode_data(interest_data=prepro_data, model_path=model_path)  # (10950, 32) (total_num, latent_dim)
        kmeans_result = KMeans(n_clusters=cluster_i, max_iter=300, random_state=66).fit(X=latent_str_data)
        str_data_labels = kmeans_result.labels_  # (10950,) (cluster_num, )
        np.save(os.path.basename(filename)[0:6] + '_' + str(cluster_i) + '_' +
                '_latent_data.npy', latent_str_data)
        np.save(os.path.basename(filename)[0:6] + '_' + str(cluster_i) + '_' +
                '_latent_labels.npy', str_data_labels)

        segmentation_infor = {}
        for i in range(cluster_i):
            segmentation_infor[i] = []
        for i in range(str_data_labels.shape[0]):
            segmentation_infor[str_data_labels[i]].append(i)

        tractogram_file = load_tractogram(filename=filename, reference='same', bbox_valid_check=False)
        streamlines = tractogram_file.streamlines
        new_streamlines = set_number_of_points(streamlines, nb_points=resample_size)
        new_data = new_streamlines.get_data()
        new_data = np.reshape(new_data, (int(new_data.shape[0] / resample_size),
                                         resample_size, -1))
        reference_file = load_tractogram(filename=filename, reference='same', bbox_valid_check=False)
        for i in range(cluster_i):
            segmentation_data_i = copy.deepcopy(new_data[segmentation_infor[i], :, :])
            final_tractogram = StatefulTractogram(streamlines=segmentation_data_i,
                                                  reference=reference_file,
                                                  space=reference_file.space)
            base_filename = os.path.basename(filename)

            save_trk(sft=final_tractogram,
                     filename=base_filename[:-4]+'_'+str(i)+'.trk',
                     bbox_valid_check=False)
            """
            save_trk(sft=final_tractogram,
                     filename=base_filename[:-4]+'_'+str(i)+'.trk',
                     bbox_valid_check=True)
            """


def reconstruct_data(str_data, scaler, resample_size, latent_dim, mdl_case, filename, case, model_path,
                     to_save, to_evaluate):
    """
       Reconstructs the data from latent space.
    """
    print("str_data.shape", str_data.shape)
    my_model = Conv1dAutoencoder(latent_dim_num=latent_dim, model_case=mdl_case, interest_data=str_data)
    latent_str_data = my_model.encode_data(interest_data=str_data, model_path=model_path)
    print("latent_str_data.shape", latent_str_data.shape)
    np.save(os.path.basename(filename)[0:6] + '_' + '_latent_data.npy', latent_str_data)
    rcst_str_data = my_model.decode_data(latent_data=latent_str_data, model_path=model_path)
    rcst_str_data = np.swapaxes(rcst_str_data, axis1=1, axis2=2)
    rcst_str_data = np.reshape(rcst_str_data, (rcst_str_data.shape[0] * rcst_str_data.shape[1], -1))
    rcst_str_data = scaler.inverse_transform(X=rcst_str_data)
    rcst_str_data = np.reshape(rcst_str_data, (int(rcst_str_data.shape[0] / resample_size),
                                                 resample_size, -1))
    print("rcst_str_data.shape", rcst_str_data.shape)
    if to_save:
        reference_file = load_tractogram(filename=filename, reference='same', bbox_valid_check=False)
        final_tractogram = StatefulTractogram(streamlines=rcst_str_data,
                                              reference=reference_file,
                                              space=reference_file.space)
        save_trk(sft=final_tractogram, filename='rcst_str_data_' + case + '.trk', bbox_valid_check=True)
    if to_evaluate:
        str_data = np.swapaxes(str_data, axis1=1, axis2=2)
        str_data = np.reshape(str_data, (str_data.shape[0] * str_data.shape[1], -1))
        str_data = scaler.inverse_transform(X=str_data)
        str_data = np.reshape(str_data, (int(str_data.shape[0] / resample_size),
                                         resample_size, -1))
        mean_abs_error = np.mean(np.abs(str_data - rcst_str_data))
        mean_square_error = np.sqrt(np.mean(np.square(str_data - rcst_str_data)))

        print("For {} and {}, the mean absolute error is {:.4f}".format(model_path, filename, mean_abs_error))
        print("For {} and {}, the root mean square error is {:.4f}".format(model_path, filename, mean_square_error))

        get_dist_infor(str_data, rcst_str_data, model_path, filename)


def get_dist_infor(str_data, rcst_str_data, model_path, filename):
    """
       Gets the information between from original streamlines and reconstructed ones.
    """
    assert str_data.shape[0] == rcst_str_data.shape[0]
    data_num = str_data.shape[0]
    total_num = int(data_num - 1)
    ori_diff_array, rec_diff_array = np.zeros(total_num, dtype='float'), np.zeros(total_num, dtype='float')
    ori_l2_array, rec_l2_array = np.zeros(total_num, dtype='float'), np.zeros(total_num, dtype='float')
    for data_i in range(0, data_num - 1, 1):
        data_j = data_i + 1
        temp_ori_diff = str_data[data_i] - str_data[data_j]
        ori_diff_array[data_i] = np.sum(temp_ori_diff)
        ori_l2_array[data_i] = np.sqrt(np.sum(np.square(temp_ori_diff)))

        temp_rec_diff = rcst_str_data[data_i] - rcst_str_data[data_j]
        rec_diff_array[data_i] = np.sum(temp_rec_diff)
        rec_l2_array[data_i] = np.sqrt(np.sum(np.square(temp_rec_diff)))

    print("For {} and {}, the single difference value is {:.4f}".format(model_path, filename,
                                                                        np.mean(ori_diff_array - rec_diff_array)))
    print("For {} and {}, the l2 difference value is {:.4f}".format(model_path, filename,
                                                                    np.mean(ori_l2_array - rec_l2_array)))