import timeit
import argparse

import numpy as np
from matplotlib import pyplot as plt

from heap import heap_sort
from util import rand_list

MICROSECS_PER_SEC = 1_000_000
POLY_EVAL_POINTS = 100


def test_runtimes(num_trials: int, num_executions: int, min_n: int, max_n: int, num_input_sizes: int, num_dvals: int,
                  max_dval: int) -> None:
    """
    Empirically test the runtime of merge sort and insertion sort by taking the minimum value
    of the specified number of trials, where each trial gives the average runtimes over the
    specified number of executions. Also fit curves to the results and plot them with the data.

    :param num_trials: number of trials to run; minimum will be taken
    :param num_executions: number of times to execute each algorithm in each trial
    :param max_exponent: exponent (base 2) of the maximum list size
    :param num_input_sizes: number of input sizes for which to test the runtimes
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

    plt.show()


if __name__ == '__main__':
    # Create command-line interface
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--trials", type=int, default=3)
    parser.add_argument("-x", "--executions", type=int, default=10)
    parser.add_argument("-m", "--min_input_size", type=int, default=2**8)
    parser.add_argument("-M", "--max_input_size", type=int, default=2**14)
    parser.add_argument("-s", "--num_input_sizes", type=int, default=10)
    parser.add_argument("-d", "--d_values", type=int, default=5)
    parser.add_argument("-D", "--max_d_val", type=int, default=1000)

    args = parser.parse_args()

    test_runtimes(args.trials, args.executions, args.min_input_size, args.max_input_size, args.num_input_sizes,
                  args.d_values, args.max_d_val)
