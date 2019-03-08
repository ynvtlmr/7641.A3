import pandas as pd
import numpy as np
import timeit
import argparse
from clustering import get_cluster_data, generate_validation_plots
from clustering import clustering_experiment, generate_cluster_plots
from clustering import generate_component_plots
from helpers import get_abspath, save_array
from sklearn.ensemble import RandomForestClassifier
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn


def rf_experiment(X, y, name, theta):
    """Run RF on specified dataset
    and saves feature importance metrics and best
    results CSV.

    Args:
        X (Numpy.Array): Attributes.
        y (Numpy.Array): Labels.
        name (str): Dataset name.
        theta (float): Min cumulative information gain threshold.

    """
    rfc = RandomForestClassifier(
        n_estimators=100, class_weight='balanced', random_state=0)
    fi = rfc.fit(X, y).feature_importances_

    # get feature importance and sort by value in descending order
    i = [i + 1 for i in range(len(fi))]
    fi = pd.DataFrame({'importance': fi, 'feature': i})
    fi.sort_values('importance', ascending=False, inplace=True)
    fi['i'] = i
    cumfi = fi['importance'].cumsum()
    fi['cumulative'] = cumfi

    # generate dataset that meets cumulative feature importance threshold
    idxs = fi.loc[:cumfi.where(cumfi > theta).idxmin(), :]
    idxs = list(idxs.index)
    reduced = X[:, idxs]

    # save results as CSV
    resdir = 'results/RF'
    fifile = get_abspath('{}_fi.csv'.format(name), resdir)
    resfile = get_abspath('{}_projected.csv'.format(name), resdir)
    save_array(array=reduced, filename=resfile, subdir=resdir)
    fi.to_csv(fifile, index_label=None)


def generate_fi_plot(name, theta):
    """Plots feature importance and cumulative feature importance values sorted
    by feature index.

    Args:
        name (str): Dataset name.
        theta (float): Explained variance percentage threshold.

    """
    resdir = 'results/RF'
    df = pd.read_csv(get_abspath('{}_fi.csv'.format(name), resdir))

    # get figure and axes
    fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(
        7 if name == 'abalone' else 12, 3
    ))

    # plot explained variance and cumulative explain variance ratios
    ax2 = ax1.twinx()
    x = df['i']
    fi = df['importance']
    cumfi = df['cumulative']

    ax1.bar(x, height=fi, color='b', tick_label=df['feature'], align='center')
    ax2.plot(x, cumfi, color='r', label='Cumulative Info Gain')
    fig.suptitle('Feature Importance ({})'.format(name))
    ax1.set_title(
        '{:.2%} Explained variance percentage by {} Features'.format(
            cumfi.loc[cumfi.where(cumfi > theta).idxmin()],
            cumfi.where(cumfi > theta).idxmin() + 1,
        )
    )
    ax1.set_ylabel('Gini Gain')
    ax2.set_ylabel('Cumulative Gini Gain')
    ax1.set_xlabel('Feature Index')
    ax2.axhline(y=theta, linestyle='--', color='r')
    ax1.grid(b=None)
    ax2.grid(b=None)

    # change layout size, font size and width
    fig.tight_layout()
    for ax in fig.axes:
        ax_items = [ax.title, ax.xaxis.label, ax.yaxis.label]
        for item in ax_items + ax.get_xticklabels() + ax.get_yticklabels():
            item.set_fontsize(8)

    # save figure
    plotdir = 'plots/RF'
    plotpath = get_abspath('{}_fi.png'.format(name), plotdir)
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
    digits_path = get_abspath('digits_projected.csv', rdir)
    abalone_path = get_abspath('abalone_projected.csv', rdir)
    digits_X = np.loadtxt(digits_path, delimiter=',')
    abalone_X = np.loadtxt(abalone_path, delimiter=',')
    rdir = rdir + '/clustering'
    pdir = pdir + '/clustering'
    # re-run clustering experiments after applying PCA
    if experiment:
        clusters = range(2, 51)
        clustering_experiment(digits_X, digits_y,
                              'digits', clusters, rdir=rdir)
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
        km_k=10, gmm_k=10, rdir=rdir, pdir=pdir,
    )
    get_cluster_data(
        abalone_X, abalone_y, 'abalone',
        km_k=3, gmm_k=3, rdir=rdir, pdir=pdir,
    )

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
    print('Running RF experiments')
    start_time = timeit.default_timer()

    train_df = pd.read_csv('../data/optdigits_train.csv', header=None)
    digits_y = train_df.iloc[:, -1:].as_matrix().flatten()
    digits_X = train_df.iloc[:, :-1].as_matrix()

    train_df = pd.read_csv('../data/abalone_train.csv', header=None)
    abalone_y = train_df.iloc[:, -1:].as_matrix().flatten()
    abalone_X = train_df.iloc[:, :-1].as_matrix()
    theta = 0.80

    # split data into X and y
    digits_dim = digits_X.shape[1]
    abalone_dim = abalone_X.shape[1]
    rdir = 'results/RF'
    pdir = 'plots/RF'
    if args.dimension:
        # set cumulative feature importance threshold

        # generate RF results
        rf_experiment(digits_X, digits_y, 'digits', theta=theta)
        rf_experiment(abalone_X, abalone_y, 'abalone', theta=theta)

        # generate RF feature importance plots
        generate_fi_plot('digits', theta=theta)
        generate_fi_plot('abalone', theta=theta)

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
    print("Completed RF experiments in {} seconds".format(elapsed))


if __name__ == '__main__':
    main()
