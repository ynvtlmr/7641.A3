import pandas as pd
import numpy as np
import timeit
from clustering import get_cluster_data, generate_validation_plots
from clustering import clustering_experiment, generate_cluster_plots
from clustering import generate_component_plots
from helpers import get_abspath, save_array
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FastICA
from scipy.stats import kurtosis, kurtosistest
import argparse
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn

import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")

def ica_experiment(X, name, dims, max_iter=5000, tol=1e-04):
    """Run ICA on specified dataset and saves mean kurtosis results as CSV
    file.

    Args:
        X (Numpy.Array): Attributes.
        name (str): Dataset name.
        dims (list(int)): List of component number values.

    """
    ica = FastICA(random_state=0, max_iter=max_iter, tol=tol)
    kurt = []
    loss = []

    X = StandardScaler().fit_transform(X)
    for dim in dims:
        print(dim)
        ica.set_params(n_components=dim)
        tmp = ica.fit_transform(X)
        df = pd.DataFrame(tmp)
        df = df.kurt(axis=0)
        kurt.append(kurtosistest(tmp).statistic.mean())
        proj = ica.inverse_transform(tmp)
        loss.append(((X - proj)**2).mean())

    res = pd.DataFrame({"kurtosis": kurt, "loss": loss})

    # save results as CSV
    resdir = 'results/ICA'
    resfile = get_abspath('{}_kurtosis.csv'.format(name), resdir)
    res.to_csv(resfile, index_label='n')


def save_ica_results(X, name, dims, max_iter=5000, tol=1e-04):
    """Run ICA and save projected dataset as CSV.

    Args:
        X (Numpy.Array): Attributes.
        name (str): Dataset name.
        dims (int): Number of components.

    """
    # transform data using ICA
    ica = FastICA(
        random_state=0, max_iter=max_iter,
        n_components=dims, tol=tol
    )
    res = ica.fit_transform(X)

    # save results file
    resdir = 'results/ICA'
    resfile = get_abspath('{}_projected.csv'.format(name), resdir)
    save_array(array=res, filename=resfile, subdir=resdir)


def generate_kurtosis_plot(name):
    """Plots mean kurtosis as a function of number of components.

    Args:
        name (str): Dataset name.

    """
    resdir = 'results/ICA'
    df = pd.read_csv(get_abspath('{}_kurtosis.csv'.format(name), resdir))

    # get figure and axes
    fig, (ax, ax1) = plt.subplots(nrows=1, ncols=2, figsize=(7, 3))

    # plot explained variance and cumulative explain variance ratios
    x = df['n']
    kurt = df['kurtosis']
    ax.plot(x, kurt, marker='.', color='g')
    ax.set_title('ICA Mean Kurtosis ({})'.format(name))
    ax.set_ylabel('Mean Kurtosis')
    ax.set_xlabel('# Components')
    ax.grid(color='grey', linestyle='dotted')
    loss = df['loss']
    ax1.plot(x, loss, marker='.', color='r')
    ax1.set_title('ICA Mean Loss ({})'.format(name))
    ax1.set_ylabel('Mean loss')
    ax1.set_xlabel('# Components')

    # change layout size, font size and width
    fig.tight_layout()
    for ax in fig.axes:
        ax_items = [ax.title, ax.xaxis.label, ax.yaxis.label]
        for item in ax_items + ax.get_xticklabels() + ax.get_yticklabels():
            item.set_fontsize(8)

    # save figure
    plotdir = 'plots/ICA'
    plotpath = get_abspath('{}_kurtosis.png'.format(name), plotdir)
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
    digits_path = get_abspath('digits_projected.csv', rdir)
    abalone_path = get_abspath('abalone_projected.csv', rdir)
    digits_X = np.loadtxt(digits_path, delimiter=',')
    abalone_X = np.loadtxt(abalone_path, delimiter=',')
    rdir = rdir + '/clustering'
    pdir = pdir + '/clustering'

    print('Running base clustering experiments ICA')
    start_time = timeit.default_timer()

    # run clustering experiments
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

    # generate 2D data for cluster visualization
    get_cluster_data(
        digits_X, digits_y, 'digits',
        km_k=20, gmm_k=12,
        rdir=rdir, pdir=pdir,
    )
    get_cluster_data(
        abalone_X, abalone_y, 'abalone',
        km_k=4, gmm_k=10, rdir=rdir, pdir=pdir,
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
    """
    Run code to generate results.

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
    start_time = timeit.default_timer()

    train_df = pd.read_csv('../data/optdigits_train.csv', header=None)
    digits_y = train_df.iloc[:, -1:].as_matrix().flatten()
    digits_X = train_df.iloc[:, :-1].as_matrix()

    train_df = pd.read_csv('../data/abalone_train.csv', header=None)
    abalone_y = train_df.iloc[:, -1:].as_matrix().flatten()
    abalone_X = train_df.iloc[:, :-1].as_matrix()

    digits_dim = digits_X.shape[1]
    abalone_dim = abalone_X.shape[1]
    rdir = 'results/ICA'
    pdir = 'plots/ICA'

    if args.dimension:
        # generate ICA results
        print('ICA digits')
        ica_experiment(
            digits_X, 'digits',
            dims=range(1, digits_dim-1),
            max_iter=100000,
            tol=0.01,
        )
        print('ICA abalone')
        ica_experiment(
            abalone_X, 'abalone',
            max_iter=100000,
            tol=0.05,
            dims=range(1, abalone_dim-1)
        )

        # generate ICA kurtosis plots
        print('ICA plots abalone')
        generate_kurtosis_plot('abalone')
        print('ICA plots digits')
        generate_kurtosis_plot('digits')

        # save ICA results with best # of components
        save_ica_results(digits_X, 'digits', dims=22)
        save_ica_results(abalone_X, 'abalone', dims=5)

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
    print("Completed ICA experiments in {} seconds".format(elapsed))


if __name__ == '__main__':
    main()
