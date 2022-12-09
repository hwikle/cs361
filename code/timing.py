import timeit
import argparse
import pickle

import numpy as np
from matplotlib import pyplot as plt

from heap import heap_sort
from util import rand_list

MICROSECS_PER_SEC = 1_000_000
POLY_EVAL_POINTS = 100


def test_runtimes(num_trials: int, num_executions: int, min_n: int, max_n: int, num_input_sizes: int, num_dvals: int,
                  max_dval: int, write_fig: bool) -> None:
    """
    Empirically test the runtime of d-ary heapsort by taking the minimum value
    of the specified number of trials, where each trial gives the average runtime over the
    specified number of executions. Also fit curves to the results and plot them with the data.
    """

    start_val = np.log(2) / np.log(max_dval)  # start with d = 2

    d_vals = max_dval ** np.linspace(start_val, 1, num_dvals)

    d_vals = list(map(lambda x: int(round(x)), d_vals))  # make them integers
    d_vals = np.array(sorted(list(set(d_vals))))  # remove duplicate values

    # Namespace hack for timeit
    namespace = locals()
    namespace['heap_sort'] = heap_sort
    namespace['rand_list'] = rand_list

    n_vals = 2 ** np.linspace(np.log2(min_n), np.log2(max_n), num_input_sizes)
    n_vals = list(map(int, n_vals))  # make them integers
    n_vals = sorted(list(set(n_vals)))  # remove duplicate values
    n_vals = np.array(n_vals)

    fit_xs = np.linspace(0.001, max_n + 20, POLY_EVAL_POINTS)
    list_copy_t = np.ndarray(n_vals.shape)

    times = np.ndarray((len(d_vals), len(n_vals)))
    d_fits = {}

    for i, d in enumerate(d_vals):
        heapsort_ts = np.ndarray(n_vals.shape)

        for j, n in enumerate(n_vals):
            namespace['n'] = n
            namespace['d'] = d

            print(f'Testing list size {n} for d={d}...')
            print(f'\tTesting heap sort runtime...')
            heapsort_ts[j] = min(timeit.repeat("heap_sort(rlist, d, copy=True)", setup="rlist=sorted(rand_list(n))",
                                               repeat=num_trials,
                                               number=num_executions,
                                               globals=namespace)) / num_executions

        # Convert to microseconds for readability

        heapsort_ts *= MICROSECS_PER_SEC
        times[i, :] = heapsort_ts

        print("Computing interpolating polynomials...")

        heapsort_coeffs = np.polyfit(d * n_vals * np.log(n_vals) / np.log(d), heapsort_ts, 1)
        heapsort_fit = np.poly1d(heapsort_coeffs)
        d_fits[d] = heapsort_fit

    # Serialize and save out data for optional later use
    with open('data.pickle', 'wb') as fout:
        pickle.dump((times, d_fits), fout)

    for k, v in d_fits.items():
        print(f"{k}: {v}")

    print('Finished testing. Plotting data...')

    colormap = plt.cm.YlOrRd(np.linspace(0.3, 1, len(d_vals)))

    for i, d in enumerate(d_vals):
        plt.plot(n_vals, times[i, :], "o", color=colormap[i])
        plt.plot(fit_xs, d_fits[d](d * fit_xs * np.log(fit_xs) / np.log(d)), "--", label=f"d = {d}", color=colormap[i])

    plt.ylabel("List Size")
    plt.ylabel("Time (microsecs)")
    plt.title('Runtime Comparison: d-ary Heaps')
    plt.legend()

    if write_fig:
        with open("d-ary_heapsort.png", 'wb') as fout:
            plt.savefig(fout, dpi=500)
    else:
        plt.show()


if __name__ == '__main__':
    # Create command-line interface
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--trials", type=int, default=3, help="Number of trials to run")
    parser.add_argument("-x", "--executions", type=int, default=10, help="Number of executions per trial")
    parser.add_argument("-m", "--min_input_size", type=int, default=2 ** 8, help="Minimum list size")
    parser.add_argument("-M", "--max_input_size", type=int, default=2 ** 14, help="Maximum list size")
    parser.add_argument("-s", "--num_input_sizes", type=int, default=10, help="Number of input sizes")
    parser.add_argument("-d", "--num_d_values", type=int, default=5, help="Number of d-values to test")
    parser.add_argument("-D", "--max_d_val", type=int, default=1024, help="Maximum d-value to test")
    parser.add_argument("-w", "--write_fig", action="store_true", dest="write_fig", help="Whether to save the figure "
                                                                                         "to a file")
    parser.set_defaults(write_fig=False)

    args = parser.parse_args()

    test_runtimes(args.trials, args.executions, args.min_input_size, args.max_input_size, args.num_input_sizes,
                  args.num_d_values, args.max_d_val, args.write_fig)
