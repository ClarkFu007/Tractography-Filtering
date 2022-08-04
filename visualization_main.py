import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def do_single_visualization(latent_data, latent_labels, data_name, tech_name):
    """
        Draws a single picture to visualize the latent space for a given technique.
    """
    plt.figure(figsize=(13.5, 4))
    sns.scatterplot(latent_data[:, 0], latent_data[:, 1], hue=latent_labels,
                    palette=sns.color_palette("hls", len(np.unique(latent_labels))),
                    legend="full", alpha=1)
    plt.title(tech_name + " of " + data_name + " data", fontsize=15, pad=15)
    plt.xlabel(tech_name + "1", fontsize=12)
    plt.ylabel(tech_name + "2", fontsize=12)
    plt.tight_layout()
    plt.savefig(tech_name + data_name + '.png', dpi=80)
    plt.show()


def visualize_pdf_experiments(latent_data_file):
    """
       Utilizes techniques to visualize PDFs of the latent space.
    """
    latent_data = np.load(latent_data_file)
    print("The mean vector is {}.".format(np.mean(latent_data, axis=0)))
    print("The std vector is {}.".format(np.std(latent_data, axis=0)))

    if False:
        feat_list = [1, 10, 19, 28, 37]
        for feat_i in feat_list:
            plt.hist(latent_data[:, feat_i], density=True, bins=1000)
            plt.title(latent_data_file[0:6] + '_' + str(feat_i), fontsize=15, pad=15)
            plt.ylabel('Probability', fontsize=12)
            plt.xlabel('Streamline i', fontsize=12)
            plt.tight_layout()
            plt.savefig('saved_plots/' + latent_data_file[0:6] + '_' + str(feat_i) + '.png', dpi=80)
            plt.show()

    from sklearn.decomposition import PCA
    pca_tool = PCA(n_components=5)
    pca_latent_data = pca_tool.fit_transform(latent_data)
    feat_list = [0, 1, 2, 3, 4]
    for feat_i in feat_list:
        plt.hist(pca_latent_data[:, feat_i], density=True, bins=1000)
        plt.title(latent_data_file[0:6] + '_' + str(feat_i), fontsize=15, pad=15)
        plt.ylabel('Probability', fontsize=12)
        plt.xlabel('Streamline i', fontsize=12)
        plt.tight_layout()
        plt.savefig('saved_plots/' + latent_data_file[0:6] + '_' + str(feat_i) + '.png', dpi=80)
        plt.show()
    if False:
        from sklearn.manifold import MDS
        m_mds_tool = MDS(n_components=5, metric=False, random_state=66)
        m_mds_latent_data = m_mds_tool.fit_transform(X=latent_data)
        feat_list = [0, 1, 2, 3, 4]
        for feat_i in feat_list:
            plt.hist(m_mds_latent_data[:, feat_i], density=True, bins=1000)
            plt.title(latent_data_file[0:6] + '_' + str(feat_i), fontsize=15, pad=15)
            plt.ylabel('Probability', fontsize=12)
            plt.xlabel('Streamline i', fontsize=12)
            plt.tight_layout()
            plt.savefig('saved_plots/' + latent_data_file[0:6] + '_' + str(feat_i) + '.png', dpi=80)
            plt.show()


def combine_visualization(pca_latent_data, lda_latent_data,
                          tsne_latent_data, umap_latent_data,
                          latent_labels, data_name, tech_name):
    """
       Combines all four visualizations into one figure.
    """
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(13.5, 4))
    sns.scatterplot(pca_latent_data[:, 0], pca_latent_data[:, 1], hue=latent_labels, palette='Set1', ax=ax[0][0])
    sns.scatterplot(lda_latent_data[:, 0], lda_latent_data[:, 1], hue=latent_labels, palette='Set1', ax=ax[0][1])
    sns.scatterplot(tsne_latent_data[:, 0], tsne_latent_data[:, 1], hue=latent_labels, palette='Set1', ax=ax[1][0])
    sns.scatterplot(umap_latent_data[:, 0], umap_latent_data[:, 1], hue=latent_labels, palette='Set1', ax=ax[1][1])
    ax[0][0].set_title("PCA of dataset", fontsize=15, pad=15)
    ax[0][1].set_title("LDA of dataset", fontsize=15, pad=15)
    ax[1][0].set_title("TSNE of dataset", fontsize=15, pad=15)
    ax[1][1].set_title("UMAP of dataset", fontsize=15, pad=15)
    ax[0][0].set_xlabel("PCA1", fontsize=12)
    ax[0][0].set_ylabel("PCA2", fontsize=12)
    ax[0][1].set_xlabel("LDA1", fontsize=12)
    ax[0][1].set_ylabel("LDA2", fontsize=12)
    ax[1][0].set_xlabel("TSNE1", fontsize=12)
    ax[1][0].set_ylabel("TSNE2", fontsize=12)
    ax[1][1].set_xlabel("UMAP1", fontsize=12)
    ax[1][1].set_ylabel("UMAP2", fontsize=12)
    plt.tight_layout()
    plt.savefig(tech_name + data_name + '.png', dpi=80)
    plt.show()


def do_visualization_experiments(latent_data_file, latent_labels_file):
    """
       Utilizes techniques to visualize the latent space.
    """
    latent_data = np.load(latent_data_file)
    latent_labels = np.load(latent_labels_file)
    from sklearn.decomposition import PCA
    pca_tool = PCA(n_components=2)
    pca_latent_data = pca_tool.fit_transform(latent_data)
    do_single_visualization(latent_data=pca_latent_data, latent_labels=latent_labels,
                            data_name=latent_data_file[0:6], tech_name='PCA')

    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    lda_tool = LinearDiscriminantAnalysis(n_components=2, solver='svd')
    lda_latent_data = lda_tool.fit_transform(X=latent_data, y=latent_labels)
    do_single_visualization(latent_data=lda_latent_data, latent_labels=latent_labels,
                            data_name=latent_data_file[0:6], tech_name='LDA')

    from sklearn.manifold import MDS
    m_mds_tool = MDS(n_components=2, metric=True, random_state=66)
    m_mds_latent_data = m_mds_tool.fit_transform(X=latent_data)
    do_single_visualization(latent_data=m_mds_latent_data, latent_labels=latent_labels,
                            data_name=latent_data_file[0:6], tech_name='m-MDS')
    nm_mds_tool = MDS(n_components=2, metric=False, random_state=66)
    nm_mds_latent_data = nm_mds_tool.fit_transform(X=latent_data)
    do_single_visualization(latent_data=nm_mds_latent_data, latent_labels=latent_labels,
                            data_name=latent_data_file[0:6], tech_name='nm-MDS')

    if False:
        from sklearn.manifold import TSNE
        tsne_tool = TSNE(n_components=2, random_state=66)
        tsne_latent_data = tsne_tool.fit_transform(X=latent_data)
        do_single_visualization(latent_data=tsne_latent_data, latent_labels=latent_labels,
                                data_name=latent_data_file[0:6], tech_name='TSNE')

    if False:
        import umap
        umap_tool = umap.UMAP(n_neighbors=15, min_dist=0.3, n_components=2)
        umap_latent_data = umap_tool.fit_transform(X=latent_data)
        do_single_visualization(latent_data=umap_latent_data, latent_labels=latent_labels,
                                data_name=latent_data_file[0:6], tech_name='UMAP')
        combine_visualization(pca_latent_data=pca_latent_data, lda_latent_data=lda_latent_data,
                              tsne_latent_data=tsne_latent_data, umap_latent_data=umap_latent_data,
                              latent_labels=latent_labels,
                              data_name=latent_data_file[0:6], tech_name='PCA_LDA_TSNE_UMAP')


def get_least_dist(streamline_data, tractography_data):
    """
       Gets the least distance between one streamline with a certain streamline
    from a tractography.
    """
    dist_array = np.sqrt(np.sum(np.square(streamline_data -
                                          tractography_data), axis=1))
    least_dist = np.min(dist_array)

    return least_dist


def measure_alignment():
    """
       Utilizes techniques to measure the alignment of the latent space.
    """
    latent_data = []
    latent_labels = []
    latent_data.append(np.load('Motor__5__latent_data.npy'))
    latent_labels.append(np.load('Motor__5__latent_labels.npy'))
    latent_data.append(np.load('307127_5__latent_data.npy'))
    latent_labels.append(np.load('307127_5__latent_labels.npy'))
    latent_data.append(np.load('308331_5__latent_data.npy'))
    latent_labels.append(np.load('308331_5__latent_labels.npy'))
    latent_data.append(np.load('316633_5__latent_data.npy'))
    latent_labels.append(np.load('316633_5__latent_labels.npy'))
    latent_data.append(np.load('329440_5__latent_data.npy'))
    latent_labels.append(np.load('329440_5__latent_labels.npy'))
    latent_data.append(np.load('352132_5__latent_data.npy'))
    latent_labels.append(np.load('352132_5__latent_labels.npy'))
    latent_data.append(np.load('352738_5__latent_data.npy'))
    latent_labels.append(np.load('352738_5__latent_labels.npy'))
    latent_data.append(np.load('366042_5__latent_data.npy'))
    latent_labels.append(np.load('366042_5__latent_labels.npy'))
    latent_data.append(np.load('377451_5__latent_data.npy'))
    latent_labels.append(np.load('377451_5__latent_labels.npy'))
    latent_data.append(np.load('380036_5__latent_data.npy'))
    latent_labels.append(np.load('380036_5__latent_labels.npy'))
    latent_data.append(np.load('385450_5__latent_data.npy'))
    latent_labels.append(np.load('385450_5__latent_labels.npy'))
    latent_data.append(np.load('397760_5__latent_data.npy'))
    latent_labels.append(np.load('397760_5__latent_labels.npy'))
    latent_data.append(np.load('412528_5__latent_data.npy'))
    latent_labels.append(np.load('412528_5__latent_labels.npy'))
    latent_data.append(np.load('415837_5__latent_data.npy'))
    latent_labels.append(np.load('415837_5__latent_labels.npy'))
    latent_data.append(np.load('422632_5__latent_data.npy'))
    latent_labels.append(np.load('422632_5__latent_labels.npy'))
    latent_data.append(np.load('433839_5__latent_data.npy'))
    latent_labels.append(np.load('433839_5__latent_labels.npy'))
    latent_data.append(np.load('441939_5__latent_data.npy'))
    latent_labels.append(np.load('441939_5__latent_labels.npy'))
    latent_data.append(np.load('473952_5__latent_data.npy'))
    latent_labels.append(np.load('473952_5__latent_labels.npy'))
    latent_data.append(np.load('480141_5__latent_data.npy'))
    latent_labels.append(np.load('480141_5__latent_labels.npy'))
    latent_data.append(np.load('499566_5__latent_data.npy'))
    latent_labels.append(np.load('499566_5__latent_labels.npy'))
    latent_data.append(np.load('510326_5__latent_data.npy'))
    latent_labels.append(np.load('510326_5__latent_labels.npy'))
    latent_data.append(np.load('519950_5__latent_data.npy'))
    latent_labels.append(np.load('519950_5__latent_labels.npy'))
    latent_data.append(np.load('540436_5__latent_data.npy'))
    latent_labels.append(np.load('540436_5__latent_labels.npy'))
    latent_data.append(np.load('547046_5__latent_data.npy'))
    latent_labels.append(np.load('547046_5__latent_labels.npy'))
    latent_data.append(np.load('565452_5__latent_data.npy'))
    latent_labels.append(np.load('565452_5__latent_labels.npy'))
    latent_data.append(np.load('567052_5__latent_data.npy'))
    latent_labels.append(np.load('567052_5__latent_labels.npy'))
    latent_data.append(np.load('568963_5__latent_data.npy'))
    latent_labels.append(np.load('568963_5__latent_labels.npy'))
    latent_data.append(np.load('579665_5__latent_data.npy'))
    latent_labels.append(np.load('579665_5__latent_labels.npy'))
    latent_data.append(np.load('592455_5__latent_data.npy'))
    latent_labels.append(np.load('592455_5__latent_labels.npy'))
    latent_data.append(np.load('594156_5__latent_data.npy'))
    latent_labels.append(np.load('594156_5__latent_labels.npy'))
    latent_data.append(np.load('601127_5__latent_data.npy'))
    latent_labels.append(np.load('601127_5__latent_labels.npy'))
    latent_data.append(np.load('613538_5__latent_data.npy'))
    latent_labels.append(np.load('613538_5__latent_labels.npy'))
    latent_data.append(np.load('623844_5__latent_data.npy'))
    latent_labels.append(np.load('623844_5__latent_labels.npy'))
    latent_data.append(np.load('644044_5__latent_data.npy'))
    latent_labels.append(np.load('644044_5__latent_labels.npy'))
    latent_data.append(np.load('672756_5__latent_data.npy'))
    latent_labels.append(np.load('672756_5__latent_labels.npy'))
    latent_data.append(np.load('695768_5__latent_data.npy'))
    latent_labels.append(np.load('695768_5__latent_labels.npy'))
    latent_data.append(np.load('713239_5__latent_data.npy'))
    latent_labels.append(np.load('713239_5__latent_labels.npy'))
    latent_data.append(np.load('727654_5__latent_data.npy'))
    latent_labels.append(np.load('727654_5__latent_labels.npy'))
    latent_data.append(np.load('771354_5__latent_data.npy'))
    latent_labels.append(np.load('771354_5__latent_labels.npy'))
    latent_data.append(np.load('865363_5__latent_data.npy'))
    latent_labels.append(np.load('865363_5__latent_labels.npy'))
    latent_data.append(np.load('871762_5__latent_data.npy'))
    latent_labels.append(np.load('871762_5__latent_labels.npy'))
    latent_data.append(np.load('894673_5__latent_data.npy'))
    latent_labels.append(np.load('894673_5__latent_labels.npy'))
    latent_data.append(np.load('896879_5__latent_data.npy'))
    latent_labels.append(np.load('896879_5__latent_labels.npy'))
    latent_data.append(np.load('898176_5__latent_data.npy'))
    latent_labels.append(np.load('898176_5__latent_labels.npy'))
    latent_data.append(np.load('899885_5__latent_data.npy'))
    latent_labels.append(np.load('899885_5__latent_labels.npy'))
    latent_data.append(np.load('901038_5__latent_data.npy'))
    latent_labels.append(np.load('901038_5__latent_labels.npy'))
    latent_data.append(np.load('901139_5__latent_data.npy'))
    latent_labels.append(np.load('901139_5__latent_labels.npy'))
    latent_data.append(np.load('901442_5__latent_data.npy'))
    latent_labels.append(np.load('901442_5__latent_labels.npy'))
    latent_data.append(np.load('904044_5__latent_data.npy'))
    latent_labels.append(np.load('904044_5__latent_labels.npy'))
    latent_data.append(np.load('907656_5__latent_data.npy'))
    latent_labels.append(np.load('907656_5__latent_labels.npy'))

    assert len(latent_data) == len(latent_labels)
    data_num = len(latent_data)
    mean_dist_matrix = np.zeros((data_num, data_num), dtype='float')
    least_dist_matrix = np.zeros((data_num, data_num), dtype='float')
    biggest_dist_matrix = np.zeros((data_num, data_num), dtype='float')

    i = 0
    for data_i in range(data_num):
        for data_j in range(data_num):
            print(i)
            if data_i != data_j:
                tractography_i = latent_data[data_i]
                tractography_j = latent_data[data_j]
                least_dist_sum = 0
                min_least_dist = np.inf
                max_least_dist = 0
                tractography_i_num = tractography_i.shape[0]
                for data_i_str_i in range(tractography_i_num):
                    least_dist = get_least_dist(streamline_data=tractography_i[data_i_str_i],
                                                tractography_data=tractography_j)
                    if min_least_dist > least_dist: min_least_dist = least_dist
                    if max_least_dist < least_dist: max_least_dist = least_dist
                    least_dist_sum += least_dist
                mean_dist_matrix[data_i, data_j] = least_dist_sum / tractography_i_num
                least_dist_matrix[data_i, data_j] = min_least_dist
                biggest_dist_matrix[data_i, data_j] = max_least_dist

            i += 1

    np.save('mean_dist_matrix.npy', mean_dist_matrix)
    np.save('least_dist_matrix.npy', least_dist_matrix)
    np.save('biggest_dist_matrix.npy', biggest_dist_matrix)


def get_dist_plot(dist_matrix, case):
    """
       Draws plots for distance between one subject with another.
    """
    subj_num = 50
    min_dist_array = np.zeros(subj_num)
    avg_dist_array = np.zeros(subj_num)
    max_dist_array = np.zeros(subj_num)
    for row_i in range(subj_num):
        min_dist_array[row_i] = np.sort(dist_matrix[row_i])[1]
        avg_dist_array[row_i] = np.sum(dist_matrix[row_i]) / (subj_num - 1)
        max_dist_array[row_i] = np.max(dist_matrix[row_i])

    plt.figure()
    plt.plot(min_dist_array, 'r', label='min')
    plt.plot(avg_dist_array, 'g', label='avg')
    plt.plot(max_dist_array, 'b', label='max')
    plt.legend()
    plt.title("Plot for " + case + " Distance between One Subject with Another")
    plt.xlabel('Subject i')
    plt.ylabel('Distance')
    plt.tight_layout()
    plt.savefig(case + "non-normalized_plot_for_distance.png")
    plt.show()

    max_distance = np.sqrt(4 * 32)
    plt.figure()
    plt.plot(min_dist_array / max_distance, 'r', label='min')
    plt.plot(avg_dist_array / max_distance, 'g', label='avg')
    plt.plot(max_dist_array / max_distance, 'b', label='max')
    plt.legend()
    plt.title("Plot for " + case + " Distance between One Subject with Another")
    plt.xlabel('Subject i')
    plt.ylabel('Distance')
    plt.tight_layout()
    plt.savefig(case + "normalized_plot_for_distance.png")
    plt.show()

    data_list = []
    for row_i in range(subj_num):
        data_list.append(dist_matrix[row_i] / max_distance)

    plt.style.use('seaborn-ticks')
    fig = plt.figure(figsize=(10, 7))
    fig.suptitle("Boxplot for " + case + " Distance between One Subject with Another")
    ax = fig.add_subplot(111)
    plt.boxplot(data_list)
    plt.xlabel('Subject i')
    plt.ylabel('Distance')
    plt.savefig(case + "boxplot_for_distance.png")
    plt.show()


def main():
    if False:
        do_visualization_experiments(latent_data_file='Motor__5__latent_data.npy',
                                     latent_labels_file='Motor__5__latent_labels.npy')
        do_visualization_experiments(latent_data_file='865363_5__latent_data.npy',
                                     latent_labels_file='865363_5__latent_labels.npy')
        do_visualization_experiments(latent_data_file='871762_5__latent_data.npy',
                                     latent_labels_file='871762_5__latent_labels.npy')
        do_visualization_experiments(latent_data_file='894673_5__latent_data.npy',
                                     latent_labels_file='894673_5__latent_labels.npy')
        do_visualization_experiments(latent_data_file='896879_5__latent_data.npy',
                                     latent_labels_file='896879_5__latent_labels.npy')

    if False:
        # measure_alignment()
        if True:
            mean_dist_matrix = np.load('mean_dist_matrix.npy')
            least_dist_matrix = np.load('least_dist_matrix.npy')
            biggest_dist_matrix = np.load('biggest_dist_matrix.npy')
            print("mean_dist_matrix", mean_dist_matrix.shape)
            print("least_dist_matrix", least_dist_matrix.shape)
            print("biggest_dist_matrix", biggest_dist_matrix.shape)
            get_dist_plot(dist_matrix=mean_dist_matrix, case='Mean')
            get_dist_plot(dist_matrix=least_dist_matrix, case='Least')
            get_dist_plot(dist_matrix=biggest_dist_matrix, case='Biggest')

    if False:
        visualize_pdf_experiments(latent_data_file='Motor___latent_data.npy')
        visualize_pdf_experiments(latent_data_file='865363__latent_data.npy')
        visualize_pdf_experiments(latent_data_file='871762__latent_data.npy')
        visualize_pdf_experiments(latent_data_file='894673__latent_data.npy')
        visualize_pdf_experiments(latent_data_file='307127__latent_data.npy')

    MotorSensory_Normal = np.load('MotorSensory_Normal_0.7_10_10_10_consistent_measure.npy')
    MotorSensory_Sphere = np.load('MotorSensory_Sphere_0.7_10_10_10_consistent_measure.npy')
    print("MotorSensory_Normal", MotorSensory_Normal)
    print("MotorSensory_Sphere", MotorSensory_Sphere)

    # first 10 subjects; 70% keep; cluster number 100; reference set [10, 10, 128, 3]






if __name__ == '__main__':
    main()
