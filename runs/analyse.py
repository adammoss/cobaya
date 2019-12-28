from __future__ import print_function
from __future__ import division

import glob
import os
import argparse
import json

import pandas as pd
import numpy as np
import getdist
import getdist.plots


def main(args):

    logzs = []
    dlogzs = []
    nlikes = []

    with open(os.path.join(args.log_root, 'params.json')) as f:
        sampled, derived = json.load(f)

    sampled_names = []
    sampled_labels = []
    triangle_names = []
    plot_names = []
    add_to_triangle_plot = True

    for k, v in sampled.items():
        print(k, v)
        sampled_names.append(k)
        sampled_labels.append(v['latex'])
        if k == 'A_planck':
            add_to_triangle_plot = False
        if k == 'A_planck':
            add_to_triangle_plot = False
        if add_to_triangle_plot:
            triangle_names.append(k)
        plot_names.append(k)

    derived_names = []
    derived_labels = []
    for k, v in derived.items():
        derived_names.append(k)
        derived_labels.append(v['latex'])

    print(sampled_names)
    print(derived_names)

    for ix, log_dir in enumerate(glob.glob(args.log_root + '/run*')):

        with open(os.path.join(log_dir, 'info', 'params.txt')) as f:
            data = json.load(f)

        if os.path.exists(os.path.join(log_dir, 'chains', 'chain.txt')):

            chain = np.loadtxt(os.path.join(log_dir, 'chains', 'chain.txt'))
            idx = np.where(chain[:, 0] >  1.0e-10)
            chain = chain[idx]
            np.savetxt(os.path.join(log_dir, 'chains', 'chain_trim.txt'), chain, fmt='%.5e')
            names = ['p%i' % i for i in range(int(data['num_params']))]
            labels = [r'x_{%i}' % i for i in range(int(data['num_params']))]
            for i in range(len(sampled_names)):
                names[i] = sampled_names[i]
                labels[i] = sampled_labels[i]
            for i in range(len(derived_names)):
                names[i + len(sampled_names)] = derived_names[i]
                labels[i + len(sampled_names)] = derived_labels[i]
            files = getdist.chains.chainFiles(os.path.join(log_dir, 'chains', 'chain_trim.txt'))
            if data['sampler'] == 'nested':
                mc = getdist.MCSamples(os.path.join(log_dir, 'chains', 'chain_trim.txt'), names=names, labels=labels,
                                       ignore_rows=0.0, sampler='nested')
            else:
                mc = getdist.MCSamples(os.path.join(log_dir, 'chains', 'chain_trim.txt'), names=names, labels=labels,
                                       ignore_rows=0.3)
            try:
                mc.readChains(files)
            except:
                continue
            print(mc.getMargeStats())
            print(mc.getLikeStats())

            if not args.no_plot:
                g = getdist.plots.getSubplotPlotter()
                g.plots_1d(mc, plot_names, nx=8);
                g.export(os.path.join(os.path.join(log_dir, 'plots', '1d.png')))
                g = getdist.plots.getSubplotPlotter()
                g.triangle_plot(mc, triangle_names, filled=True)
                g.export(os.path.join(os.path.join(log_dir, 'plots', 'triangle.png')))

        if os.path.exists(os.path.join(log_dir, 'results', 'final.csv')):
            results = pd.read_csv(os.path.join(log_dir, 'results', 'final.csv'))
            print(results)
            logzs.append(results['logz'])
            dlogzs.append(results['logzerr'])
            nlikes.append(results['ncall'])

        if len(logzs) > 1:
            print()
            print(r'Log Z: $%4.2f \pm %4.2f$' % (np.mean(logzs), np.std(logzs)))
            print(r'Log Z error estimate: $%4.2f \pm %4.2f$' % (np.mean(dlogzs), np.std(dlogzs)))
            print(r'N_like: $%.0f \pm %.0f$' % (np.mean(nlikes), np.std(nlikes)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--log_root', type=str, default='')
    parser.add_argument('-no_plot', action='store_true')

    args = parser.parse_args()
    main(args)
