#!/usr/bin/env python3

"""
Run the underrepresented-population rescue simulation scenarios.

Demonstrates that PRScsx-MT can "rescue" prediction accuracy for an
underrepresented population (EAS, n=500) by borrowing signal through
two information channels:

  1. Cross-population: EUR -> EAS  (via rho_pop)
  2. Cross-trait:      trait0 -> trait1 (via rg)

When both channels are active the EAS trait-1 estimates receive a
"two-hop" boost that single-trait PRScsx cannot exploit.

Scenario axes
-------------
A. Varying rg  (0.0, 0.3, 0.6, 0.8) with EAS n=500, rho_pop=0.8
   -> isolates cross-trait contribution
B. Varying EAS n  (500, 2000, 5000) with rg=0.6, rho_pop=0.8
   -> shows rescue magnitude shrinks as direct power grows
C. Lower rho_pop  (0.4) with rg=0.6, EAS n=500
   -> shows rescue requires reasonable cross-pop transferability
D. Asymmetric h2  (0.5, 0.1) with rg=0.6, EAS n=500
   -> low-h2 trait benefits most from borrowing

Usage
-----
  python run_underrep_rescue.py                          # all scenarios
  python run_underrep_rescue.py --scenarios underrep_rg0.6_eas500  # single
  python run_underrep_rescue.py --n_snp 2000 --n_iter 1000         # larger sim
"""

import os
import sys
import json
import argparse
import csv
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

import compare_methods


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run underrepresented-population rescue scenarios.')
    parser.add_argument('--n_snp', type=int, default=1000,
                        help='Number of SNPs (default: 1000)')
    parser.add_argument('--n_iter', type=int, default=500,
                        help='MCMC iterations (default: 500)')
    parser.add_argument('--n_burnin', type=int, default=None,
                        help='Burn-in iterations (default: n_iter//2)')
    parser.add_argument('--block_size', type=int, default=50,
                        help='LD block size (default: 50)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--out_dir', type=str, default=None,
                        help='Output directory (default: underrep_rescue_output)')
    parser.add_argument('--scenarios', type=str, nargs='+', default=None,
                        help='Run only named scenarios')
    args = parser.parse_args()
    if args.n_burnin is None:
        args.n_burnin = args.n_iter // 2
    if args.out_dir is None:
        args.out_dir = os.path.join(script_dir, 'underrep_rescue_output')
    return args


def load_scenarios(filter_names=None):
    """Load scenarios from the JSON config file."""
    path = os.path.join(script_dir, 'scenarios_underrep_rescue.json')
    with open(path) as f:
        scenarios = json.load(f)
    if filter_names is not None:
        names = set(filter_names)
        scenarios = [s for s in scenarios if s['name'] in names]
        if not scenarios:
            print('ERROR: no matching scenarios for: %s' % ', '.join(filter_names))
            sys.exit(1)
    return scenarios


def print_rescue_analysis(all_results):
    """Print analysis focused on the EAS (underrepresented pop) rescue effect."""
    print('\n' + '=' * 95)
    print('UNDERREPRESENTED-POPULATION RESCUE ANALYSIS')
    print('Focus: EAS (pop index 1) accuracy gain from multi-trait modeling')
    print('=' * 95)

    header = '%-35s %-6s %-6s %-8s %-12s %-12s %-10s' % (
        'Scenario', 'Pop', 'Trait', 'n_gwas', 'corr_single', 'corr_MT', 'delta')
    print(header)
    print('-' * 95)

    for res in all_results:
        for row in res['rows']:
            delta_str = '%+.4f' % row['delta'] if not np.isnan(row['delta']) else 'N/A'
            corr_s = '%.4f' % row['corr_prscsx'] if not np.isnan(row['corr_prscsx']) else 'N/A'
            corr_m = '%.4f' % row['corr_mt'] if not np.isnan(row['corr_mt']) else 'N/A'
            # Highlight EAS rows
            marker = ' <--' if row['pop'] == 'EAS' else ''
            print('%-35s %-6s %-6d %-8d %-12s %-12s %-10s%s' % (
                row['scenario'], row['pop'], row['trait'], row['n_gwas'],
                corr_s, corr_m, delta_str, marker))

    print('-' * 95)

    # Summarize EAS-specific gains
    print('\nEAS-ONLY SUMMARY (underrepresented population):')
    print('-' * 65)
    print('%-35s %-8s %-12s %-12s' % ('Scenario', 'Trait', 'corr_gain', 'rel_gain%'))
    print('-' * 65)
    for res in all_results:
        for row in res['rows']:
            if row['pop'] == 'EAS' and not np.isnan(row['delta']):
                base = row['corr_prscsx']
                rel = (row['delta'] / base * 100) if base > 0 else float('nan')
                rel_str = '%+.1f%%' % rel if not np.isnan(rel) else 'N/A'
                print('%-35s %-8d %+.4f      %s' % (
                    row['scenario'], row['trait'], row['delta'], rel_str))
    print('-' * 65)
    print()


def main():
    args = parse_args()
    scenarios = load_scenarios(args.scenarios)

    print('\n' + '=' * 70)
    print('UNDERREPRESENTED-POPULATION RESCUE SIMULATIONS')
    print('  n_snp=%d  n_iter=%d  n_burnin=%d  seed=%d' % (
        args.n_snp, args.n_iter, args.n_burnin, args.seed))
    print('  %d scenario(s) to run' % len(scenarios))
    print('=' * 70)

    # Clean and create output directory
    os.makedirs(args.out_dir, exist_ok=True)

    all_results = []
    for scenario in scenarios:
        result = compare_methods.run_scenario(
            scenario,
            n_snp=args.n_snp,
            n_iter=args.n_iter,
            n_burnin=args.n_burnin,
            block_size=args.block_size,
            base_out_dir=args.out_dir,
            seed=args.seed,
        )
        all_results.append(result)

    # Standard summary table
    compare_methods.print_summary_table(all_results)

    # Rescue-focused analysis
    print_rescue_analysis(all_results)

    # Save CSV
    csv_path = os.path.join(args.out_dir, 'underrep_rescue_results.csv')
    compare_methods.save_results_csv(all_results, csv_path)

    print('All output saved to: %s' % args.out_dir)


if __name__ == '__main__':
    main()
