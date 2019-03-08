from collections import defaultdict
from itertools import product
import timeit
import argparse
import pandas as pd
import numpy as np
from clustering import get_cluster_data, generate_validation_plots
from clustering import clustering_experiment, generate_cluster_plots
from clustering import generate_component_plots
from helpers import get_abspath, save_array
from helpers import reconstruction_error, pairwise_dist_corr
from sklearn.random_projection import SparseRandomProjection
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn


def rp_experiment(X, y, name, dims):
    """Run Randomized Projections on specified dataset and saves reconstruction
    error and pairwise distance correlation results as CSV file.

    Args:
        X (Numpy.Array): Attributes.
        name (str): Dataset name.
        dims (list(int)): List of component number values.

    """
    re = defaultdict(dict)
    pdc = defaultdict(dict)

    for i, dim in product(range(10), dims):
        rp = SparseRandomProjection(random_state=i, n_components=dim)
        rp.fit(X)
        re[dim][i] = reconstruction_error(rp, X)
        pdc[dim][i] = pairwise_dist_corr(rp.transform(X), X)

    re = pd.DataFrame(pd.DataFrame(re).T.mean(axis=1))
    re.rename(columns={0: 'recon_error'}, inplace=True)
    pdc = pd.DataFrame(pd.DataFrame(pdc).T.mean(axis=1))
    pdc.rename(columns={0: 'pairwise_dc'}, inplace=True)
    metrics = pd.concat((re, pdc), axis=1)

    # save results as CSV
    resdir = 'results/RP'
    resfile = get_abspath('{}_metrics.csv'.format(name), resdir)
    metrics.to_csv(resfile, index_label='n')


def save_rp_results(X, name, dims):
    """Run RP and save projected dataset as CSV.

    Args:
        X (Numpy.Array): Attributes.
        name (str): Dataset name.
        dims (int): Number of components.

    """
    # transform data using ICA
    rp = SparseRandomProjection(random_state=0, n_components=dims)
    res = rp.fit_transform(X)

    # save results file
    resdir = 'results/RP'
    resfile = get_abspath('{}_projected.csv'.format(name), resdir)
    save_array(array=res, filename=resfile, subdir=resdir)


def generate_plots(name):
    """Plots reconstruction error and pairwise distance correlation as a
    function of number of components.

    Args:
        name (str): Dataset name.

    """
    resdir = 'results/RP'
    df = pd.read_csv(get_abspath('{}_metrics.csv'.format(name), resdir))

    # get figure and axes
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(8, 3))

    # plot metrics
    x = df['n']
    re = df['recon_error']
    pdc = df['pairwise_dc']
    ax1.plot(x, re, marker='.', color='g')
    ax1.set_title('Reconstruction Error ({})'.format(name))
    ax1.set_ylabel('Reconstruction error')
    ax1.set_xlabel('# Components')
    ax1.grid(color='grey', linestyle='dotted')

    ax2.plot(x, pdc, marker='.', color='b')
    ax2.set_title('Pairwise Distance Correlation ({})'.format(name))
    ax2.set_ylabel('Pairwise distance correlation')
    ax2.set_xlabel('# Components')
    ax2.grid(color='grey', linestyle='dotted')

    # change layout size, font size and width
    fig.tight_layout()
    for ax in fig.axes:
        ax_items = [ax.title, ax.xaxis.label, ax.yaxis.label]
        for item in ax_items + ax.get_xticklabels() + ax.get_yticklabels():
            item.set_fontsize(8)

    # save figure
    plotdir = 'plots/RP'
    plotpath = get_abspath('{}_metrics.png'.format(name), plotdir)
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
    print('Running base clustering experiments RP')
    start_time = timeit.default_timer()

    digits_path = get_abspath('digits_projected.csv', rdir)
    abalone_path = get_abspath('abalone_projected.csv', rdir)
    digits_X = np.loadtxt(digits_path, delimiter=',')
    abalone_X = np.loadtxt(abalone_path, delimiter=',')
    rdir = rdir + '/clustering'
    pdir = pdir + '/clustering'
    # re-run clustering experiments after applying PCA
    if experiment:
        clusters = range(2, 51)
        clustering_experiment(
            digits_X, digits_y, 'digits', clusters, rdir=rdir
        )
        clustering_experiment(
            abalone_X, abalone_y, 'abalone', clusters, rdir=rdir
        )

        # generate component plots (metrics to choose size of k)
        generate_component_plots(name='digits', rdir=rdir, pdir=pdir)
        generate_component_plots(name='abalone', rdir=rdir, pdir=pdir)

        # # generate validation plots (relative performance of clustering)
        generate_validation_plots(name='digits', rdir=rdir, pdir=pdir)
        generate_validation_plots(name='abalone', rdir=rdir, pdir=pdir)
        return

    # # generate 2D data for cluster visualization
    get_cluster_data(
        digits_X, digits_y, 'digits',
        km_k=10, gmm_k=10, rdir=rdir, pdir=pdir,
    )
    get_cluster_data(
        abalone_X, abalone_y, 'abalone',
        km_k=3, gmm_k=3, rdir=rdir, pdir=pdir
    )
    # generate validation plots (relative performance of clustering)
    df_digits_2D = pd.read_csv(get_abspath('digits_2D.csv', rdir))
    generate_cluster_plots(df_digits_2D, name='digits', pdir=pdir)

    df_abalone_2D = pd.read_csv(
        get_abspath('abalone_2D.csv', rdir)
    )
    generate_cluster_plots(
        df_abalone_2D, name='abalone', pdir=pdir
    )
    end_time = timeit.default_timer()
    elapsed = end_time - start_time
    print("Completed clustering experiments in {} seconds".format(elapsed))


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
    print('Running ICA experiments')
    print('Running RP experiments')
    start_time = timeit.default_timer()

    train_df = pd.read_csv('../data/optdigits_train.csv', header=None)
    digits_y = train_df.iloc[:, -1:].as_matrix().flatten()
    digits_X = train_df.iloc[:, :-1].as_matrix()

    train_df = pd.read_csv('../data/abalone_train.csv', header=None)
    abalone_y = train_df.iloc[:, -1:].as_matrix().flatten()
    abalone_X = train_df.iloc[:, :-1].as_matrix()

    digits_dim = digits_X.shape[1]
    abalone_dim = abalone_X.shape[1]
    rdir = 'results/RP'
    pdir = 'plots/RP'

    if args.dimension:
        # generate RP results
        rp_experiment(digits_X, digits_y, 'digits',
                      dims=range(1, digits_dim + 1))
        rp_experiment(abalone_X, abalone_y, 'abalone',
                      dims=range(1, abalone_dim + 1))

        # generate RP explained variance plots
        generate_plots(name='digits')
        generate_plots(name='abalone')

        # save RP results with best # of components
        save_rp_results(digits_X, 'digits', dims=30)
        save_rp_results(abalone_X, 'abalone', dims=4)
    else:
        # # re-run clustering experiments
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
    print("Completed RP experiments in {} seconds".format(elapsed))


if __name__ == '__main__':
    main()
