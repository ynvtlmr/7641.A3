import pandas as pd
import numpy as np
import timeit
import argparse
from clustering import get_cluster_data, generate_validation_plots
from clustering import clustering_experiment, generate_cluster_plots
from clustering import generate_component_plots, nn_cluster_datasets
from sklearn.preprocessing import StandardScaler
from helpers import get_abspath, save_array
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn


def pca_experiment(X, name, dims, evp):
    """Run PCA on specified dataset and saves dataset with components that
    explain at least 85% of total variance or 2 components which ever is larger

    Args:
        X (Numpy.Array): Attributes.
        name (str): Dataset name.
        dims (int): Number of components.
        evp (float): Explained variance percentage threshold.

    """
    pca = PCA(random_state=0, svd_solver='full', n_components=dims)

    comps = pca.fit_transform(
        StandardScaler().fit_transform(X)
    )  # get principal components

    # cumulative explained variance greater than threshold
    r = range(1, dims + 1)
    ev = pd.Series(pca.explained_variance_, index=r, name='ev')
    evr = pd.Series(pca.explained_variance_ratio_, index=r, name='evr')
    evrc = evr.rename('evr_cum').cumsum()
    res = comps[:, :max(evrc.where(evrc > evp).idxmin(), 2)]
    error = []
    for _ in range(1, dims+1):
        print(_)
        pca = PCA(random_state=0, svd_solver='full', n_components=_)
        comps = pca.fit_transform(
            StandardScaler().fit_transform(X)
        )  # get principal components
        data_reduced = np.dot(X, pca.components_.T)
        error.append(((X - np.dot(data_reduced, pca.components_)) ** 2).mean())

    evars = pd.concat((ev, evr, evrc), axis=1)
    evars['loss'] = error

    # save results as CSV
    resdir = 'results/PCA'
    evfile = get_abspath('{}_variances.csv'.format(name), resdir)
    resfile = get_abspath('{}_projected.csv'.format(name), resdir)
    save_array(array=res, filename=resfile, subdir=resdir)
    evars.to_csv(evfile, index_label='n')


def generate_variance_plot(name, evp):
    """Plots explained variance and cumulative explained variance ratios as a
    function of principal components.

    Args:
        name (str): Dataset name.
        evp (float): Explained variance percentage threshold.

    """
    resdir = 'results/PCA'
    df = pd.read_csv(get_abspath('{}_variances.csv'.format(name), resdir))

    # get figure and axes
    fig, (ax, ax1) = plt.subplots(nrows=1, ncols=2, figsize=(5, 3))

    # plot explained variance and cumulative explain variance ratios
    x = df['n']
    evr = df['evr']
    evr_cum = df['evr_cum']
    ax.plot(x, evr, marker='.', color='b', label='EVR')
    ax.plot(x, evr_cum, marker='.', color='g', label='Cumulative EVR')
    vmark = evr_cum.where(evr_cum > evp).idxmin() + 1
    fig.suptitle('PCA Explained Variance by PC ({})'.format(name))
    ax.set_title(
        '{:.2%} Cumulative Variance \n Explained by {} Components'.format(
             evr_cum[vmark-1], vmark
        )
    )
    ax.set_ylabel('Explained Variance')
    ax.set_xlabel('Principal Component')
    ax.axvline(x=vmark, linestyle='--', color='r')
    ax.grid(color='grey', linestyle='dotted')
    loss = df['loss']
    ax1.plot(x, loss, marker='.', color='r')
    ax1.set_title('PCA Mean Loss ({})'.format(name))
    ax1.set_ylabel('Mean loss')
    ax1.set_xlabel('# Components')

    # change layout size, font size and width
    fig.tight_layout()
    for ax in fig.axes:
        ax_items = [ax.title, ax.xaxis.label, ax.yaxis.label]
        for item in ax_items + ax.get_xticklabels() + ax.get_yticklabels():
            item.set_fontsize(8)

    # save figure
    plotdir = 'plots/PCA'
    plotpath = get_abspath('{}_explvar.png'.format(name), plotdir)
    plt.savefig(plotpath)
    plt.close()


def run_clustering(digits_y, abalone_y, rdir, pdir, experiment=False):
    """Re-run clustering experiments on datasets after dimensionality
    reduction.

    Args:
        digits_y (Numpy.Array): Labels for digits.
        abalone_y(Numpy.Array): Labels for abalones.
        rdir (str): Input file directory.
        pdir (str): Output directory.

    """

    start_time = timeit.default_timer()
    abalone_X = np.loadtxt(
        get_abspath('abalone_projected.csv', rdir),
        delimiter=','
    )

    digits_X = np.loadtxt(
        get_abspath('digits_projected.csv', rdir),
        delimiter=','
    )

    rdir = rdir + '/clustering'
    pdir = pdir + '/clustering'

    if experiment:
        # re-run clustering experiments after applying PCA
        clusters = range(2, 51)
        clustering_experiment(abalone_X, abalone_y,
                              'abalone', clusters, rdir=rdir)
        clustering_experiment(digits_X, digits_y, 'digits',
                              clusters, rdir=rdir)

        # generate component plots (metrics to choose size of k)
        generate_component_plots(name='abalone', rdir=rdir, pdir=pdir)
        generate_component_plots(name='digits', rdir=rdir, pdir=pdir)

        # # generate validation plots (relative performance of clustering)
        generate_validation_plots(name='abalone', rdir=rdir, pdir=pdir)
        generate_validation_plots(name='digits', rdir=rdir, pdir=pdir)

        return

    # generate 2D data for cluster visualization
    get_cluster_data(
        abalone_X, abalone_y, 'abalone',
        km_k=9, gmm_k=12, rdir=rdir, pdir=pdir,
    )
    get_cluster_data(
        digits_X, digits_y, 'digits',
        km_k=20, gmm_k=12, rdir=rdir, pdir=pdir,
    )
    # generate validation plots (relative performance of clustering)
    generate_cluster_plots(
        pd.read_csv(get_abspath('abalone_2D.csv', rdir)),
        name='abalone',
        pdir=pdir
    )
    generate_cluster_plots(
        pd.read_csv(get_abspath('digits_2D.csv', rdir)),
        name='digits',
        pdir=pdir
    )


def main():
    """Run code to generate results.

    """
    parser = argparse.ArgumentParser(description='Optional app description')
    parser.add_argument('--cluster_exp', nargs='?',
                        help='generate clustering experiment')
    parser.add_argument('--dimension', nargs='?',
                        help='generate clustering experiment')
    args = parser.parse_args()
    print("Argument values:")
    print(args)
    # split data into X and y
    train_df = pd.read_csv('../data/optdigits_train.csv', header=None)
    digits_y = train_df.iloc[:, -1:].as_matrix().flatten()
    digits_X = train_df.iloc[:, :-1].as_matrix()

    train_df = pd.read_csv('../data/abalone_train.csv', header=None)
    abalone_y = train_df.iloc[:, -1:].as_matrix().flatten()
    abalone_X = train_df.iloc[:, :-1].as_matrix()

    digits_dim = digits_X.shape[1]
    abalone_dim = abalone_X.shape[1]
    rdir = 'results/PCA'
    pdir = 'plots/PCA'

    print('Running PCA experiments')
    start_time = timeit.default_timer()
    if args.dimension:
        # set explained variance threshold
        evp = 0.90

        # generate PCA results
        pca_experiment(digits_X, 'digits', dims=digits_dim, evp=evp)
        pca_experiment(abalone_X, 'abalone', dims=abalone_dim, evp=evp)

        # generate PCA explained variance plots
        generate_variance_plot('digits', evp=evp)
        generate_variance_plot('abalone', evp=evp)
        end_time = timeit.default_timer()
        elapsed = end_time - start_time
        print("Completed PCA experiments in {} seconds".format(elapsed))
    else:
        # re-run clustering experiments
        run_clustering(
            digits_y,
            abalone_y,
            rdir,
            pdir,
            experiment=args.cluster_exp
        )

    # calculate and print running time
    end_time = timeit.default_timer()
    elapsed = end_time - start_time
    print("Completed PCA experiments in {} seconds".format(elapsed))


if __name__ == '__main__':
    main()
