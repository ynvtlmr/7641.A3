import pandas as pd
import numpy as np
import timeit
from helpers import cluster_acc, get_abspath, save_dataset, save_array
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture as GMM
from collections import defaultdict
from sklearn.metrics import adjusted_mutual_info_score as ami
from sklearn.metrics.cluster import contingency_matrix
from sklearn.metrics.cluster import homogeneity_score, completeness_score
import matplotlib
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

matplotlib.use('agg')
warnings.filterwarnings('ignore')


def get_one_hot(targets, nb_classes):
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape) + [nb_classes])


def clustering_experiment(X, y, name, clusters, rdir):
    """Generate results CSVs for given datasets using the K-Means and EM
    clustering algorithms.

    Args:
        X (Numpy.Array): Attributes.
        y (Numpy.Array): Labels.
        name (str): Dataset name.
        clusters (list[int]): List of k values.
        rdir (str): Output directory.

    """
    sse = defaultdict(dict)  # sum of squared errors
    logl = defaultdict(dict)  # log-likelihood
    bic = defaultdict(dict)  # BIC for EM
    aic = defaultdict(dict)  # AIC for EM
    aic = defaultdict(dict)  # AIC for EM
    silhouette = defaultdict(dict)  # silhouette score
    acc = defaultdict(lambda: defaultdict(dict))  # accuracy scores
    adjmi = defaultdict(lambda: defaultdict(dict))  # adjusted mutual info
    homo = defaultdict(lambda: defaultdict(dict))  # adjusted mutual info
    km = KMeans(random_state=0)  # K-Means
    gmm = GMM(random_state=0)  # Gaussian Mixture Model (EM)

    # start loop for given values of k
    print('DATESET: %s' % name)
    for k in clusters:
        print('K: %s' % k)
        km.set_params(n_clusters=k)
        gmm.set_params(n_components=k)
        km.fit(X)
        gmm.fit(X)

        # calculate SSE, log-likelihood, accuracy, and adjusted mutual info
        sse[k][name] = km.score(X)
        logl[k][name] = gmm.score(X)
        acc[k][name]['km'] = cluster_acc(y, km.predict(X))
        acc[k][name]['gmm'] = cluster_acc(y, gmm.predict(X))
        adjmi[k][name]['km'] = ami(y, km.predict(X))
        adjmi[k][name]['gmm'] = ami(y, gmm.predict(X))

        homo[k][name]['km'] = homogeneity_score(y, km.predict(X))
        homo[k][name]['gmm'] = homogeneity_score(y, gmm.predict(X))

        # calculate silhouette score for K-Means
        km_silhouette = silhouette_score(X, km.predict(X))
        silhouette[k][name] = km_silhouette

        # calculate BIC for EM
        bic[k][name] = gmm.bic(X)
        aic[k][name] = gmm.aic(X)

    # generate output dataframes
    sse = (-pd.DataFrame(sse)).T
    sse.rename(columns={name: 'sse'}, inplace=True)
    logl = pd.DataFrame(logl).T
    logl.rename(columns={name: 'log-likelihood'}, inplace=True)
    bic = pd.DataFrame(bic).T
    bic.rename(columns={name: 'bic'}, inplace=True)
    aic = pd.DataFrame(aic).T
    aic.rename(columns={name: 'aic'}, inplace=True)
    silhouette = pd.DataFrame(silhouette).T
    silhouette.rename(columns={name: 'silhouette_score'}, inplace=True)
    acc = pd.Panel(acc)
    acc = acc.loc[:, :, name].T.rename(
        lambda x: '{}_acc'.format(x), axis='columns')
    adjmi = pd.Panel(adjmi)
    adjmi = adjmi.loc[:, :, name].T.rename(
        lambda x: '{}_adjmi'.format(x), axis='columns')
    homo = pd.Panel(homo)
    homo = homo.loc[:, :, name].T.rename(
        lambda x: '{}_homo'.format(x), axis='columns')

    # concatenate all results
    dfs = (sse, silhouette, logl, bic, aic, acc, adjmi, homo)
    metrics = pd.concat(dfs, axis=1)
    print(metrics)
    resfile = get_abspath('{}_train_metrics.csv'.format(name), rdir)
    metrics.to_csv(resfile, index_label='k')


def clustering_experiment_test(X, y, X_test, y_test, name, clusters, rdir):
    """Generate results CSVs for given datasets using the K-Means and EM
    clustering algorithms.

    Args:
        X (Numpy.Array): Attributes.
        y (Numpy.Array): Labels.
        name (str): Dataset name.
        clusters (list[int]): List of k values.
        rdir (str): Output directory.

    """
    sse = defaultdict(dict)  # sum of squared errors
    logl = defaultdict(dict)  # log-likelihood
    sse_test = defaultdict(dict)  # sum of squared errors
    logl_test = defaultdict(dict)  # log-likelihood
    # bic = defaultdict(dict)  # BIC for EM
    # aic = defaultdict(dict)  # AIC for EM
    # aic = defaultdict(dict)  # AIC for EM
    # silhouette = defaultdict(dict)  # silhouette score
    # acc = defaultdict(lambda: defaultdict(dict))  # accuracy scores
    # adjmi = defaultdict(lambda: defaultdict(dict))  # adjusted mutual info
    # homo = defaultdict(lambda: defaultdict(dict))  # adjusted mutual info
    km = KMeans(random_state=0)  # K-Means
    gmm = GMM(random_state=0)  # Gaussian Mixture Model (EM)

    # start loop for given values of k
    print('DATESET: %s' % name)
    for k in clusters:
        print('K: %s' % k)
        km.set_params(n_clusters=k)
        gmm.set_params(n_components=k)
        km.fit(X)
        gmm.fit(X)

        # calculate SSE, log-likelihood, accuracy, and adjusted mutual info
        sse[k][name] = km.score(X)
        logl[k][name] = gmm.score(X)
        sse_test[k][name] = km.score(X_test)
        logl_test[k][name] = gmm.score(X_test)

        # acc[k][name]['km'] = cluster_acc(y, km.predict(X))
        # acc[k][name]['gmm'] = cluster_acc(y, gmm.predict(X))
        # adjmi[k][name]['km'] = ami(y, km.predict(X))
        # adjmi[k][name]['gmm'] = ami(y, gmm.predict(X))
        #
        # homo[k][name]['km'] = homogeneity_score(y, km.predict(X))
        # homo[k][name]['gmm'] = homogeneity_score(y, gmm.predict(X))

        # calculate silhouette score for K-Means
        # km_silhouette = silhouette_score(X, km.predict(X))
        # silhouette[k][name] = km_silhouette
        #
        # # calculate BIC for EM
        # bic[k][name] = gmm.bic(X)
        # aic[k][name] = gmm.aic(X)

    # generate output dataframes
    sse = (-pd.DataFrame(sse)).T
    sse.rename(columns={name: 'sse'}, inplace=True)
    logl = pd.DataFrame(logl).T
    logl.rename(columns={name: 'log-likelihood'}, inplace=True)

    sse_test = (-pd.DataFrame(sse_test)).T
    sse_test.rename(columns={name: 'sse_test'}, inplace=True)
    logl_test = pd.DataFrame(logl_test).T
    logl_test.rename(columns={name: 'log-likelihood_test'}, inplace=True)

    # bic = pd.DataFrame(bic).T
    # bic.rename(columns={name: 'bic'}, inplace=True)
    # aic = pd.DataFrame(aic).T
    # aic.rename(columns={name: 'aic'}, inplace=True)
    # silhouette = pd.DataFrame(silhouette).T
    # silhouette.rename(columns={name: 'silhouette_score'}, inplace=True)
    # acc = pd.Panel(acc)
    # acc = acc.loc[:, :, name].T.rename(
    #     lambda x: '{}_acc'.format(x), axis='columns')
    # adjmi = pd.Panel(adjmi)
    # adjmi = adjmi.loc[:, :, name].T.rename(
    #     lambda x: '{}_adjmi'.format(x), axis='columns')
    # homo = pd.Panel(homo)
    # homo = homo.loc[:, :, name].T.rename(
    #     lambda x: '{}_homo'.format(x), axis='columns')

    # concatenate all results
    dfs = (sse, logl, sse_test, logl_test)
    metrics = pd.concat(dfs, axis=1)
    print(metrics)
    resfile = get_abspath('{}_train_test_metrics.csv'.format(name), rdir)
    metrics.to_csv(resfile, index_label='k')


def generate_component_plots(name, rdir, pdir):
    """Generates plots of result files for given dataset.

    Args:
        name (str): Dataset name.
        rdir (str): Input file directory.
        pdir (str): Output directory.

    """
    metrics = pd.read_csv(
        get_abspath('{}_train_metrics.csv'.format(name), rdir)
    )

    # test_metrics = pd.read_csv(
    #     get_abspath('{}_train_test_metrics.csv'.format(name), rdir)
    # )

    # get figure and axes
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=1,
                                             ncols=4,
                                             figsize=(15, 3))

    # plot SSE for K-Means
    k = metrics['k']
    metric = metrics['sse']
    # test_metric = test_metrics['sse_test']
    ax1.plot(k, metric, marker='o', markersize=5, color='g')
    # ax1.plot(k, test_metric, marker='o', markersize=5, color='r')
    ax1.set_title('K-Means SSE ({})'.format(name), fontsize=14)
    ax1.set_ylabel('Sum of squared error')
    ax1.set_xlabel('Number of clusters (k)')
    ax1.grid(color='grey', linestyle='dotted')

    # plot Silhoutte Score for K-Means
    metric = metrics['silhouette_score']
    ax2.plot(k, metric, marker='o', markersize=5, color='b')
    ax2.set_title('K-Means Avg Silhouette Score ({})'.format(name))
    ax2.set_ylabel('Mean silhouette score')
    ax2.set_xlabel('Number of clusters (k)')
    ax2.grid(color='grey', linestyle='dotted')

    # plot log-likelihood for EM
    metric = metrics['log-likelihood']
    # test_metric = test_metrics['log-likelihood_test']
    ax3.plot(k, metric, marker='o', markersize=5, color='r')
    # ax3.plot(k, test_metric, marker='o', markersize=5, color='r')
    ax3.set_title('GMM Log-likelihood ({})'.format(name), fontsize=14)
    ax3.set_ylabel('Log-likelihood')
    ax3.set_xlabel('Number of clusters (k)')
    ax3.grid(color='grey', linestyle='dotted')

    # plot BIC for EM
    metric = metrics['bic']
    ax4.plot(k, metric, marker='o', markersize=5, color='k')
    aic = metrics['aic']
    ax4.plot(k, aic, marker='o', markersize=5, color='g')
    ax4.set_title('GMM BIC/AIC ({})'.format(name), fontsize=14)
    ax4.legend(loc='best')
    ax4.set_ylabel('BIC')
    ax4.set_xlabel('Number of clusters (k)')
    ax4.grid(color='grey', linestyle='dotted')

    # change layout size, font size and width between subplots
    fig.tight_layout()
    for ax in fig.axes:
        ax_items = [ax.xaxis.label, ax.yaxis.label]
        ax.title.set_fontsize(12)
        for item in ax_items + ax.get_xticklabels() + ax.get_yticklabels():
            item.set_fontsize(8)
    plt.subplots_adjust(wspace=0.3)

    # save figure
    plotpath = get_abspath('{}_components.png'.format(name), pdir)
    plt.savefig(plotpath)
    plt.close()


def generate_validation_plots(name, rdir, pdir):
    """Generates plots of validation metrics (accuracy, adjusted mutual info)
    for both datasets.

    Args:
        name (str): Dataset name.
        rdir (str): Input file directory.
        pdir (str): Output directory.

    """
    metrics = pd.read_csv(
        get_abspath('{}_train_metrics.csv'.format(name), rdir)
    )

    # get figure and axes
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(9, 4))

    # plot accuracy
    k = metrics['k']
    km = metrics['km_acc']
    gmm = metrics['gmm_acc']
    ax1.plot(k, km, marker='o', markersize=5, color='b', label='K-Means')
    ax1.plot(k, gmm, marker='o', markersize=5, color='g', label='GMM')
    ax1.set_title('Cluster Accuracy ({})'.format(name))
    ax1.set_ylabel('%')
    ax1.set_xlabel('Number of clusters (k)')
    ax1.grid(color='grey', linestyle='dotted')
    ax1.legend(loc='best')

    # plot adjusted mutual info
    km = metrics['km_adjmi']
    gmm = metrics['gmm_adjmi']
    ax2.plot(k, km, marker='o', markersize=5, color='r', label='K-Means')
    ax2.plot(k, gmm, marker='o', markersize=5, color='k', label='GMM')
    ax2.set_title('Adjusted Mutual Info ({})'.format(name))
    ax2.set_ylabel('Adjusted mutual information score')
    ax2.set_xlabel('Number of clusters (k)')
    ax2.grid(color='grey', linestyle='dotted')
    ax2.legend(loc='best')

    # plot adjusted.homogeneity_score
    km = metrics['km_homo']
    gmm = metrics['gmm_homo']
    ax3.plot(k, km, marker='o', markersize=5, color='r', label='K-Means')
    ax3.plot(k, gmm, marker='o', markersize=5, color='k', label='GMM')
    ax3.set_title('Homogeneity ({})'.format(name))
    ax3.set_ylabel('Score')
    ax3.set_xlabel('Number of clusters (k)')
    ax3.grid(color='grey', linestyle='dotted')
    ax3.legend(loc='best')

    # change layout size, font size and width between subplots
    fig.tight_layout()
    for ax in fig.axes:
        ax_items = [ax.xaxis.label, ax.yaxis.label]
        ax.title.set_fontsize(12)
        for item in ax_items + ax.get_xticklabels() + ax.get_yticklabels():
            item.set_fontsize(8)
    plt.subplots_adjust(wspace=0.3)

    # save figure
    plotpath = get_abspath('{}_validation.png'.format(name), pdir)
    plt.savefig(plotpath)
    plt.close()


def get_cluster_data(X, y, name, km_k, gmm_k, rdir, pdir, perplexity=50):
    """Generates 2D dataset that contains cluster labels for K-Means and GMM,
    as well as the class labels for the given dataset.

    Args:
        X (Numpy.Array): Attributes.
        y (Numpy.Array): Labels.
        name (str): Dataset name.
        perplexity (int): Perplexity parameter for t-SNE.
        km_k (int): Number of clusters for K-Means.
        gmm_k (int): Number of components for GMM.
        rdir (str): Folder to save results CSV.

    """
    print('get_cluster_data: %s' % name)
    # generate 2D X dataset
    X2D = TSNE(n_iter=5000, perplexity=perplexity).fit_transform(X)

    # get cluster labels using best k
    km = KMeans(random_state=0).set_params(n_clusters=km_k)
    gmm = GMM(random_state=0).set_params(n_components=gmm_k)
    km_contingency_matrix = contingency_matrix(y, km.fit(X).labels_)
    gm_contingency_matrix = contingency_matrix(y, gmm.fit(X).predict(X))
    print(km.cluster_centers_)
    generate_contingency_matrix(
        km_contingency_matrix,
        gm_contingency_matrix,
        name,
        pdir,
    )
    km_cl = np.atleast_2d(km.fit(X2D).labels_).T
    gmm_cl = np.atleast_2d(gmm.fit(X2D).predict(X2D)).T
    y = np.atleast_2d(y).T

    # create concatenated dataset
    cols = ['x1', 'x2', 'km', 'gmm', 'class']
    df = pd.DataFrame(np.hstack((X2D, km_cl, gmm_cl, y)), columns=cols)

    # save as CSV
    filename = '{}_2D.csv'.format(name)
    save_dataset(df, filename, sep=',', subdir=rdir, header=True)


def generate_contingency_matrix(kmeans_contigency,
                                gmm_continigency, name, pdir):
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(14, 3))
    ax1 = sns.heatmap(
        kmeans_contigency,
        linewidths=.5, cmap="coolwarm",
        ax=ax1)
    ax1.set_title('K-Means Clusters ({})'.format(name))
    ax1.set_xlabel('Cluster')
    ax1.set_ylabel('True label')

    ax2 = sns.heatmap(
        gmm_continigency,
        linewidths=.5, cmap="coolwarm",
        ax=ax2)
    ax2.set_title('GMM Clusters ({})'.format(name))
    ax2.set_xlabel('Cluster')
    ax2.set_ylabel('True label')

    fig.tight_layout()
    for ax in fig.axes:
        ax_items = [ax.title, ax.xaxis.label, ax.yaxis.label]
        for item in ax_items + ax.get_xticklabels() + ax.get_yticklabels():
            item.set_fontsize(8)
    plt.subplots_adjust(wspace=0.3)

    fig.suptitle('Contigency Plot - {}'.format(name), fontsize=12)
    # save figure
    plotdir = pdir
    plotpath = get_abspath('{}_contingecy.png'.format(name), plotdir)
    plt.savefig(plotpath)
    plt.close()


def generate_cluster_plots(df, name, pdir):
    """Visualizes clusters using pre-processed 2D dataset.

    Args:
        df (Pandas.DataFrame): Dataset containing attributes and labels.
        name (str): Dataset name.
        pdir (str): Output folder for plots.

    """
    # get cols
    x1 = df['x1']
    x2 = df['x2']
    km = df['km']
    gmm = df['gmm']
    c = df['class']

    # plot cluster scatter plots
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(14, 3))
    ax1.scatter(x1, x2, marker='o', s=20, c=km, cmap='rainbow')
    ax1.set_title('K-Means Clusters ({})'.format(name))
    ax1.set_ylabel('x1')
    ax1.set_xlabel('x2')
    ax1.grid(color='grey', linestyle='dotted')

    ax2.scatter(x1, x2, marker='o', s=20, c=gmm, cmap='rainbow')
    ax2.set_title('GMM Clusters ({})'.format(name))
    ax2.set_ylabel('x1')
    ax2.set_xlabel('x2')
    ax2.grid(color='grey', linestyle='dotted')

    # change color map depending on dataset
    ax3.scatter(x1, x2, marker='o', s=20, c=c, cmap='rainbow')
    ax3.set_title('Class Labels ({})'.format(name))
    ax3.set_ylabel('x1')
    ax3.set_xlabel('x2')
    ax3.grid(color='grey', linestyle='dotted')

    # change layout size, font size and width between subplots
    fig.tight_layout()
    for ax in fig.axes:
        ax_items = [ax.title, ax.xaxis.label, ax.yaxis.label]
        for item in ax_items + ax.get_xticklabels() + ax.get_yticklabels():
            item.set_fontsize(12)
    plt.subplots_adjust(wspace=0.3)

    # save figure
    plotdir = pdir
    plotpath = get_abspath('{}_clusters.png'.format(name), plotdir)
    plt.savefig(plotpath)
    plt.close()


def nn_cluster_datasets(X, name, km_k, gmm_k):
    """Generates datasets for ANN classification by appending cluster label to
    original dataset.

    Args:
        X (Numpy.Array): Original attributes.
        name (str): Dataset name.
        km_k (int): Number of clusters for K-Means.
        gmm_k (int): Number of components for GMM.

    """
    km = KMeans(random_state=0).set_params(n_clusters=km_k)
    gmm = GMM(random_state=0).set_params(n_components=gmm_k)
    km.fit(X)
    gmm.fit(X)

    # add cluster labels to original attributes
    km_x = np.concatenate((X, km.labels_[:, None]), axis=1)
    gmm_x = np.concatenate((X, gmm.predict(X)[:, None]), axis=1)

    # save results
    resdir = 'results/NN'

    kmfile = get_abspath('{}_km_only_labels.csv'.format(name), resdir)
    gmmfile = get_abspath('{}_gmm_only_labels.csv'.format(name), resdir)
    save_array(array=km.labels_[:, None], filename=kmfile, subdir=resdir)
    save_array(array=gmm.predict(X)[:, None], filename=gmmfile, subdir=resdir)

    kmfile = get_abspath('{}_km_labels.csv'.format(name), resdir)
    gmmfile = get_abspath('{}_gmm_labels.csv'.format(name), resdir)
    save_array(array=km_x, filename=kmfile, subdir=resdir)
    save_array(array=gmm_x, filename=gmmfile, subdir=resdir)


def main():
    """Run code to generate clustering results.

    """
    parser = argparse.ArgumentParser(description='Optional app description')
    parser.add_argument('--generate', nargs='?',
                        help='generate clustering experiment')
    args = parser.parse_args()
    print("Argument values:")
    print(args)

    print('Running base clustering experiments')
    start_time = timeit.default_timer()

    rdir = 'results/clustering'
    pdir = 'plots/clustering'

    train_df = pd.read_csv('../data/optdigits_train.csv', header=None)
    digits_y = train_df.iloc[:, -1:].values.flatten()
    digits_X = train_df.iloc[:, :-1].values

    train_df = pd.read_csv('../data/abalone_train.csv', header=None)
    abalone_y = train_df.iloc[:, -1:].values.flatten()
    abalone_X = train_df.iloc[:, :-1].values

    test_df = pd.read_csv('../data/optdigits_test.csv', header=None)
    digits_y_test = test_df.iloc[:, -1:].values.flatten()
    digits_X_test = test_df.iloc[:, :-1].values

    test_df = pd.read_csv('../data/abalone_test.csv', header=None)
    abalone_y_test = test_df.iloc[:, -1:].values.flatten()
    abalone_X_test = test_df.iloc[:, :-1].values

    # run clustering experiments
    clusters = range(2, 50)
    if args.generate:
        # run clustering experiment
        clustering_experiment(digits_X, digits_y, 'digits', clusters, rdir=rdir)
        clustering_experiment(abalone_X, abalone_y, 'abalone', clusters, rdir=rdir)

        # run clustering experiment with testing data
        clustering_experiment_test(digits_X, digits_y, digits_X_test, digits_y_test,
                                   'digits', clusters, rdir=rdir)
        clustering_experiment_test(abalone_X, abalone_y, abalone_X_test, abalone_y_test,
                                   'abalone', clusters, rdir=rdir)

        # generate component plots
        generate_component_plots(name='digits', rdir=rdir, pdir=pdir)
        generate_component_plots(name='abalone', rdir=rdir, pdir=pdir)

        # generate validation plots (relative performance of clustering)
        generate_validation_plots(name='digits', rdir=rdir, pdir=pdir)
        generate_validation_plots(name='abalone', rdir=rdir, pdir=pdir)
    else:
        get_cluster_data(
            digits_X, digits_y, 'digits',
            km_k=10, gmm_k=10, rdir=rdir, pdir=pdir,
        )
        get_cluster_data(
            abalone_X, abalone_y, 'abalone',
            km_k=3, gmm_k=3, rdir=rdir, pdir=pdir,
        )

        # generate 2D data for cluster visualization
        # generate validation plots (relative performance of clustering)
        df_digits_2D = pd.read_csv(get_abspath('digits_2D.csv', rdir))
        generate_cluster_plots(df_digits_2D, name='digits', pdir=pdir)

        df_abalone_2d = pd.read_csv(get_abspath('abalone_2D.csv', rdir))
        generate_cluster_plots(df_abalone_2d, name='abalone', pdir=pdir)

    # calculate and print running time
    end_time = timeit.default_timer()
    elapsed = end_time - start_time
    print("Completed clustering experiments in {} seconds".format(elapsed))


if __name__ == '__main__':
    main()
