#!/usr/bin/env python3

"""
Compare PRS accuracy: PRScsx (single-trait) vs PRScsx_mt (multi-trait).

Simulates two correlated traits with asymmetric sample sizes to demonstrate
the benefit of multi-trait modeling when one trait is underpowered.
"""

import os
import sys
import shutil
import argparse
import time
import csv
import json
import numpy as np

# Add paths for both implementations
script_dir = os.path.dirname(os.path.abspath(__file__))
prscsx_dir = os.path.join(os.path.dirname(script_dir), 'PRScsx')
sys.path.insert(0, script_dir)
sys.path.insert(0, prscsx_dir)

import simulate_mt
import mcmc_gtb_mt
import parse_genet_mt

# Import PRScsx modules with aliases to avoid conflicts
import importlib.util
spec_mcmc = importlib.util.spec_from_file_location("mcmc_gtb_orig", os.path.join(prscsx_dir, "mcmc_gtb.py"))
mcmc_gtb_orig = importlib.util.module_from_spec(spec_mcmc)
spec_mcmc.loader.exec_module(mcmc_gtb_orig)

spec_parse = importlib.util.spec_from_file_location("parse_genet_orig", os.path.join(prscsx_dir, "parse_genet.py"))
parse_genet_orig = importlib.util.module_from_spec(spec_parse)
spec_parse.loader.exec_module(parse_genet_orig)


def read_effect_file(filepath):
    """Read posterior effect size file."""
    betas = []
    snps = []
    with open(filepath) as ff:
        for line in ff:
            parts = line.strip().split()
            snps.append(parts[1])
            betas.append(float(parts[5]))
    return np.array(betas), snps


def read_true_effects(filepath):
    """Read true effect file and return dict mapping SNP -> beta."""
    snp_beta = {}
    with open(filepath) as ff:
        next(ff)  # skip header
        for line in ff:
            parts = line.strip().split()
            snp_beta[parts[0]] = float(parts[1])
    return snp_beta


def run_prscsx_single_trait(ref_dir, bim_prefix, sst_file, n_gwas, pop_label,
                             out_dir, out_name, n_iter=500, n_burnin=250, seed=42):
    """Run original PRScsx on a single trait."""
    os.makedirs(out_dir, exist_ok=True)
    chrom = 1
    n_pop = len(pop_label)

    # Parse reference and validation data
    ref_dict = parse_genet_orig.parse_ref(ref_dir + '/snpinfo_mult_1kg_hm3', chrom, '1kg')
    vld_dict = parse_genet_orig.parse_bim(bim_prefix, chrom)

    # Parse summary statistics for each population
    sst_dict = {}
    for pp in range(n_pop):
        sst_dict[pp] = parse_genet_orig.parse_sumstats(
            ref_dict, vld_dict, sst_file[pp], pop_label[pp], n_gwas[pp])

    # Parse LD blocks
    ld_blk = {}
    blk_size = {}
    for pp in range(n_pop):
        ld_blk[pp], blk_size[pp] = parse_genet_orig.parse_ldblk(
            ref_dir, sst_dict[pp], pop_label[pp], chrom, '1kg')

    # Align LD blocks
    snp_dict, beta_dict, frq_dict, idx_dict = parse_genet_orig.align_ldblk(
        ref_dict, vld_dict, sst_dict, n_pop, chrom)

    # Run MCMC
    mcmc_gtb_orig.mcmc(
        1, 0.5, None, snp_dict, beta_dict, frq_dict, idx_dict,
        n_gwas, ld_blk, blk_size,
        n_iter, n_burnin, 5, pop_label, chrom,
        out_dir, out_name, 'FALSE', 'FALSE', seed)


def run_prscsx_mt(ref_dir, bim_prefix, sst_files, n_gwas, pop_labels,
                  out_dir, out_name, n_trait=2, n_iter=500, n_burnin=250, seed=42):
    """Run PRScsx_mt on multiple traits."""
    os.makedirs(out_dir, exist_ok=True)
    chrom = 1
    n_pop = len(pop_labels)

    # Parse reference and validation data
    ref_dict = parse_genet_mt.parse_ref(ref_dir + '/snpinfo_mult_1kg_hm3', chrom, '1kg')
    vld_dict = parse_genet_mt.parse_bim(bim_prefix, chrom)

    # Parse summary statistics for each (pop, trait) pair
    sst_dict = {}
    n_gwas_dict = {}
    for pp in range(n_pop):
        for tt in range(n_trait):
            sst_dict[(pp, tt)] = parse_genet_mt.parse_sumstats(
                ref_dict, vld_dict, sst_files[(pp, tt)],
                pop_labels[pp], n_gwas[(pp, tt)])
            n_gwas_dict[(pp, tt)] = n_gwas[(pp, tt)]

    # Parse LD blocks (shared across traits within each population)
    ld_blk = {}
    blk_size = {}
    for pp in range(n_pop):
        ld_blk[pp], blk_size[pp] = parse_genet_mt.parse_ldblk(
            ref_dir, sst_dict[(pp, 0)], pop_labels[pp], chrom, '1kg')

    # Align LD blocks for multi-trait
    snp_dict, beta_dict, frq_dict, idx_dict = parse_genet_mt.align_ldblk_mt(
        ref_dict, vld_dict, sst_dict, n_pop, n_trait, chrom)

    # Run MCMC
    mcmc_gtb_mt.mcmc(
        1, 0.5, None, snp_dict, beta_dict, frq_dict, idx_dict,
        n_gwas_dict, ld_blk, blk_size,
        n_iter, n_burnin, 5, pop_labels, chrom,
        out_dir, out_name, 'FALSE', 'FALSE', seed,
        n_trait=n_trait, rho_e=None, lambda_psi=1.0)


def evaluate_accuracy(out_dir, out_name, data_dir, pop_labels, n_trait, method_name, is_mt=False, single_trait_idx=None):
    """Evaluate estimated effects against true effects.

    Parameters
    ----------
    single_trait_idx : int or None
        For single-trait PRScsx runs, specifies which trait's true effects to compare against.
        If None (default for multi-trait), uses tt from the loop.
    """
    results = {}
    n_pop = len(pop_labels)

    for pp in range(n_pop):
        for tt in range(n_trait):
            # Build filename based on method
            if is_mt:
                eff_file = '%s/%s_%s_trait%d_pst_eff_a1_b0.5_phiauto_chr1.txt' % (
                    out_dir, out_name, pop_labels[pp], tt)
            else:
                eff_file = '%s/%s_%s_pst_eff_a1_b0.5_phiauto_chr1.txt' % (
                    out_dir, out_name, pop_labels[pp])

            if not os.path.isfile(eff_file):
                print('  WARNING: %s not found' % eff_file)
                continue

            # Read estimated effects
            beta_est, snps_est = read_effect_file(eff_file)

            # Read true effects - use single_trait_idx if provided (for single-trait PRScsx)
            true_trait_idx = single_trait_idx if single_trait_idx is not None else tt
            true_file = '%s/true_effects_%s_trait%d.txt' % (data_dir, pop_labels[pp], true_trait_idx)
            true_snp_beta = read_true_effects(true_file)

            # Align true effects to estimated SNP order
            beta_true_aligned = np.array([true_snp_beta.get(s, 0.0) for s in snps_est])

            # Compute correlation
            if np.std(beta_est) > 0 and np.std(beta_true_aligned) > 0:
                corr = np.corrcoef(beta_est, beta_true_aligned)[0, 1]
            else:
                corr = 0.0

            results[(pp, tt)] = {
                'corr': corr,
                'n_snp': len(beta_est),
                'pop': pop_labels[pp],
                'trait': true_trait_idx
            }

    return results


DEFAULT_POP_LABELS = ['EUR', 'EAS', 'AFR', 'SAS', 'AMR']

DEFAULT_SCENARIOS = [
    {'name': '2pop_2trait_rg0.2', 'n_pop': 2, 'n_trait': 2, 'rg': 0.2, 'n_gwas_pattern': 'asymmetric'},
    {'name': '2pop_2trait_rg0.5', 'n_pop': 2, 'n_trait': 2, 'rg': 0.5, 'n_gwas_pattern': 'asymmetric'},
    {'name': '2pop_2trait_rg0.8', 'n_pop': 2, 'n_trait': 2, 'rg': 0.8, 'n_gwas_pattern': 'asymmetric'},
    {'name': '2pop_3trait_rg0.5', 'n_pop': 2, 'n_trait': 3, 'rg': 0.5, 'n_gwas_pattern': 'asymmetric'},
    {'name': '2pop_5trait_rg0.5', 'n_pop': 2, 'n_trait': 5, 'rg': 0.5, 'n_gwas_pattern': 'asymmetric'},
    {'name': '3pop_2trait_rg0.5', 'n_pop': 3, 'n_trait': 2, 'rg': 0.5, 'n_gwas_pattern': 'asymmetric'},
    {'name': '2pop_2trait_rg0.5_sym', 'n_pop': 2, 'n_trait': 2, 'rg': 0.5, 'n_gwas_pattern': 'symmetric'},
]


def build_n_gwas(n_pop, n_trait, pattern, n_large=20000, n_small=5000):
    """Generate the [n_pop][n_trait] sample size list from a named pattern.

    Parameters
    ----------
    n_pop : int
    n_trait : int
    pattern : str
        'asymmetric' — trait 0 gets n_large, rest get n_small.
        'symmetric'  — all traits get n_large.
    n_large, n_small : int
        GWAS sample sizes for well-powered and underpowered traits.

    Returns
    -------
    list of lists
        n_gwas[pop][trait]
    """
    n_gwas = []
    for _ in range(n_pop):
        row = []
        for tt in range(n_trait):
            if pattern == 'symmetric':
                row.append(n_large)
            else:  # asymmetric
                row.append(n_large if tt == 0 else n_small)
        n_gwas.append(row)
    return n_gwas


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Compare PRScsx (single-trait) vs PRScsx_mt (multi-trait) across scenarios.')

    parser.add_argument('--n_snp', type=int, default=1000,
                        help='Number of SNPs to simulate (default: 1000)')
    parser.add_argument('--n_iter', type=int, default=500,
                        help='Number of MCMC iterations (default: 500)')
    parser.add_argument('--n_burnin', type=int, default=None,
                        help='Number of burn-in iterations (default: n_iter//2)')
    parser.add_argument('--block_size', type=int, default=50,
                        help='LD block size (default: 50)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--out_dir', type=str, default=None,
                        help='Output directory (default: comparison_output in script dir)')
    parser.add_argument('--scenarios', type=str, nargs='+', default=None,
                        help='Filter scenarios by name (e.g. --scenarios 2pop_2trait_rg0.8)')
    parser.add_argument('--scenario_file', type=str, default=None,
                        help='Load custom scenarios from a JSON file')
    parser.add_argument('--n_large', type=int, default=20000,
                        help='GWAS sample size for well-powered traits (default: 20000)')
    parser.add_argument('--n_small', type=int, default=5000,
                        help='GWAS sample size for underpowered traits (default: 5000)')

    args = parser.parse_args()
    if args.n_burnin is None:
        args.n_burnin = args.n_iter // 2
    if args.out_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        args.out_dir = os.path.join(base_dir, 'comparison_output')
    return args


def get_scenarios(args):
    """Return the list of scenario dicts to run."""
    if args.scenario_file is not None:
        with open(args.scenario_file) as f:
            scenarios = json.load(f)
    else:
        scenarios = DEFAULT_SCENARIOS

    if args.scenarios is not None:
        names = set(args.scenarios)
        scenarios = [s for s in scenarios if s['name'] in names]
        if not scenarios:
            print('ERROR: no matching scenarios found for: %s' % ', '.join(args.scenarios))
            sys.exit(1)

    return scenarios


def run_scenario(scenario, n_snp, n_iter, n_burnin, block_size, base_out_dir, seed,
                 n_large=20000, n_small=5000):
    """Run a single comparison scenario and return results.

    Returns
    -------
    dict with keys 'scenario', 'rows' (list of per-pop-trait dicts), 'time_prscsx', 'time_mt'
    """
    name = scenario['name']
    n_pop = scenario['n_pop']
    n_trait = scenario['n_trait']
    rg = scenario['rg']
    rho_pop = scenario.get('rho_pop', 0.8)  # default to 0.8 for backward compatibility

    # Resolve n_gwas
    if 'n_gwas' in scenario:
        n_gwas = scenario['n_gwas']
    else:
        n_gwas = build_n_gwas(n_pop, n_trait, scenario['n_gwas_pattern'],
                              n_large=n_large, n_small=n_small)

    pop_labels = DEFAULT_POP_LABELS[:n_pop]
    n_causal = max(n_snp // 10, 5)
    h2 = scenario.get('h2', [0.5] * n_trait)

    scenario_dir = os.path.join(base_out_dir, name)
    data_dir = os.path.join(scenario_dir, 'simulated_data')

    print('\n' + '=' * 70)
    print('SCENARIO: %s' % name)
    print('  n_pop=%d  n_trait=%d  rg=%.2f  n_snp=%d  pattern=%s'
          % (n_pop, n_trait, rg, n_snp, scenario.get('n_gwas_pattern', 'literal')))
    print('  rho_pop = %s' % rho_pop)
    print('  h2 = %s' % h2)
    print('  n_gwas = %s' % n_gwas)
    print('=' * 70)

    # --- Step 1: Simulate data ---
    print('\n  Simulating data ...')
    simulate_mt.simulate_mt(
        n_snp=n_snp, n_causal=n_causal, n_pop=n_pop, n_trait=n_trait,
        n_gwas=n_gwas, block_size=block_size, ld_decay=0.5,
        rg=rg, rho_pop=rho_pop, h2=h2,
        pop=pop_labels, out_dir=data_dir, chrom=1, seed=seed
    )

    # --- Step 2: PRScsx (single-trait) per trait ---
    print('  Running PRScsx (single-trait) ...')
    t0 = time.time()
    prscsx_results = {}
    for tt in range(n_trait):
        sst_files_trait = [
            os.path.join(data_dir, 'sst_%s_trait%d.txt' % (pop_labels[pp], tt))
            for pp in range(n_pop)
        ]
        n_gwas_trait = [n_gwas[pp][tt] for pp in range(n_pop)]
        out_dir_trait = os.path.join(scenario_dir, 'prscsx_trait%d' % tt)

        run_prscsx_single_trait(
            ref_dir=data_dir,
            bim_prefix=os.path.join(data_dir, 'sim_data'),
            sst_file=sst_files_trait,
            n_gwas=n_gwas_trait,
            pop_label=pop_labels,
            out_dir=out_dir_trait,
            out_name='prscsx_t%d' % tt,
            n_iter=n_iter, n_burnin=n_burnin, seed=seed
        )

        res = evaluate_accuracy(
            out_dir_trait, 'prscsx_t%d' % tt, data_dir, pop_labels,
            n_trait=1, method_name='PRScsx', is_mt=False, single_trait_idx=tt
        )
        for (pp, _), metrics in res.items():
            prscsx_results[(pp, tt)] = metrics
    time_prscsx = time.time() - t0

    # --- Step 3: PRScsx_mt (multi-trait) ---
    print('  Running PRScsx_mt (multi-trait) ...')
    t0 = time.time()
    sst_files_mt = {}
    n_gwas_mt = {}
    for pp in range(n_pop):
        for tt in range(n_trait):
            sst_files_mt[(pp, tt)] = os.path.join(
                data_dir, 'sst_%s_trait%d.txt' % (pop_labels[pp], tt))
            n_gwas_mt[(pp, tt)] = n_gwas[pp][tt]

    out_dir_mt = os.path.join(scenario_dir, 'prscsx_mt')
    run_prscsx_mt(
        ref_dir=data_dir,
        bim_prefix=os.path.join(data_dir, 'sim_data'),
        sst_files=sst_files_mt,
        n_gwas=n_gwas_mt,
        pop_labels=pop_labels,
        out_dir=out_dir_mt,
        out_name='prscsx_mt',
        n_trait=n_trait,
        n_iter=n_iter, n_burnin=n_burnin, seed=seed
    )

    prscsx_mt_results = evaluate_accuracy(
        out_dir_mt, 'prscsx_mt', data_dir, pop_labels,
        n_trait=n_trait, method_name='PRScsx_mt', is_mt=True
    )
    time_mt = time.time() - t0

    # --- Collect rows ---
    rows = []
    for pp in range(n_pop):
        for tt in range(n_trait):
            corr_single = prscsx_results.get((pp, tt), {}).get('corr', np.nan)
            corr_multi = prscsx_mt_results.get((pp, tt), {}).get('corr', np.nan)
            delta = corr_multi - corr_single if not (np.isnan(corr_single) or np.isnan(corr_multi)) else np.nan
            rows.append({
                'scenario': name,
                'pop': pop_labels[pp],
                'trait': tt,
                'n_gwas': n_gwas[pp][tt],
                'corr_prscsx': corr_single,
                'corr_mt': corr_multi,
                'delta': delta,
            })

    return {'scenario': name, 'rows': rows, 'time_prscsx': time_prscsx, 'time_mt': time_mt}


def print_summary_table(all_results):
    """Print a formatted summary table across all scenarios."""
    print('\n' + '=' * 90)
    print('SUMMARY: PRScsx (single-trait) vs PRScsx_mt (multi-trait)')
    print('=' * 90)
    header = '%-25s %-6s %-6s %-8s %-12s %-12s %-8s' % (
        'Scenario', 'Pop', 'Trait', 'n_gwas', 'corr_PRScsx', 'corr_MT', 'delta')
    print(header)
    print('-' * 90)

    for res in all_results:
        for row in res['rows']:
            delta_str = '%+.4f' % row['delta'] if not np.isnan(row['delta']) else 'N/A'
            corr_s = '%.4f' % row['corr_prscsx'] if not np.isnan(row['corr_prscsx']) else 'N/A'
            corr_m = '%.4f' % row['corr_mt'] if not np.isnan(row['corr_mt']) else 'N/A'
            print('%-25s %-6s %-6d %-8d %-12s %-12s %-8s' % (
                row['scenario'], row['pop'], row['trait'], row['n_gwas'],
                corr_s, corr_m, delta_str))

    print('-' * 90)

    # Timing summary
    print('\nTiming (seconds):')
    for res in all_results:
        print('  %-25s  PRScsx: %7.1f   PRScsx_mt: %7.1f' % (
            res['scenario'], res['time_prscsx'], res['time_mt']))
    print('=' * 90 + '\n')


def save_results_csv(all_results, filepath):
    """Write all results rows to a CSV file."""
    fieldnames = ['scenario', 'pop', 'trait', 'n_gwas', 'corr_prscsx', 'corr_mt', 'delta']
    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for res in all_results:
            for row in res['rows']:
                writer.writerow(row)
    print('Results written to %s' % filepath)


def main():
    args = parse_args()

    print('\n' + '=' * 70)
    print('COMPARISON: PRScsx (single-trait) vs PRScsx_mt (multi-trait)')
    print('  n_snp=%d  n_iter=%d  n_burnin=%d  seed=%d' % (
        args.n_snp, args.n_iter, args.n_burnin, args.seed))
    print('=' * 70)

    scenarios = get_scenarios(args)

    # Clean up previous output
    if os.path.exists(args.out_dir):
        shutil.rmtree(args.out_dir)
    os.makedirs(args.out_dir)

    all_results = []
    for scenario in scenarios:
        result = run_scenario(
            scenario,
            n_snp=args.n_snp,
            n_iter=args.n_iter,
            n_burnin=args.n_burnin,
            block_size=args.block_size,
            base_out_dir=args.out_dir,
            seed=args.seed,
            n_large=args.n_large,
            n_small=args.n_small,
        )
        all_results.append(result)

    print_summary_table(all_results)

    csv_path = os.path.join(args.out_dir, 'results_summary.csv')
    save_results_csv(all_results, csv_path)

    print('Output files saved to: %s' % args.out_dir)
    print('=' * 70 + '\n')


if __name__ == '__main__':
    main()
