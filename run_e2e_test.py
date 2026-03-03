#!/usr/bin/env python3

"""
End-to-end test for PRS-CSx-MT.

Tests:
1. T=1 single-trait mode (backward compatibility with PRS-CSx)
2. T=2 independent traits (rg=0)
3. T=2 correlated traits (rg=0.8)
4. T=2 with sample overlap correction

"""

import os
import sys
import shutil
import numpy as np

# Add current dir to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import simulate_mt
import mcmc_gtb_mt as mcmc_gtb
import parse_genet_mt as parse_genet
import gigrnd


def generate_test_data(out_dir, scenario_name, pop_labels=None, n_snp=200, n_causal=20, n_pop=2,
                       n_trait=1, n_gwas=None, rg=0.0, rho_pheno=0.0,
                       n_overlap=None, block_size=50, seed=42):
    """Generate test data for a specific scenario."""
    print('\n' + '='*60)
    print('GENERATING TEST DATA: %s' % scenario_name)
    print('='*60)

    if n_gwas is None:
        n_gwas = [[10000]*n_trait for _ in range(n_pop)]

    sim = simulate_mt.simulate_mt(
        n_snp=n_snp, n_causal=n_causal, n_pop=n_pop, n_trait=n_trait,
        n_gwas=n_gwas, block_size=block_size, ld_decay=0.5,
        rg=rg, rho_pop=0.8, h2=[0.5]*n_trait,
        rho_pheno=rho_pheno, n_overlap=n_overlap,
        pop=pop_labels, out_dir=out_dir, chrom=1, seed=seed
    )

    print('Test data written to: %s' % out_dir)
    return sim


def run_prscsx_mt(ref_dir, bim_prefix, sst_files, n_gwas, pop_labels,
                  out_dir, out_name, n_trait=1, rho_pheno=None,
                  n_overlap=None, lambda_psi=1.0, n_iter=500,
                  n_burnin=250, seed=42):
    """Run PRS-CSx-MT programmatically (without subprocess)."""
    print('\n' + '-'*40)
    print('RUNNING PRS-CSx-MT: %s' % out_name)
    print('-'*40)

    os.makedirs(out_dir, exist_ok=True)

    n_pop = len(pop_labels)
    chrom = 1

    # Determine reference panel type
    if os.path.isfile(ref_dir + '/snpinfo_mult_1kg_hm3'):
        ref = '1kg'
        ref_dict = parse_genet.parse_ref(ref_dir + '/snpinfo_mult_1kg_hm3', chrom, ref)
    else:
        print('ERROR: reference panel not found in %s' % ref_dir)
        return None

    vld_dict = parse_genet.parse_bim(bim_prefix, chrom)

    if n_trait == 1:
        # Single-trait mode
        sst_dict = {}
        for pp in range(n_pop):
            sst_dict[pp] = parse_genet.parse_sumstats(
                ref_dict, vld_dict, sst_files[pp], pop_labels[pp], n_gwas[pp])

        ld_blk = {}
        blk_size = {}
        for pp in range(n_pop):
            ld_blk[pp], blk_size[pp] = parse_genet.parse_ldblk(
                ref_dir, sst_dict[pp], pop_labels[pp], chrom, ref)

        snp_dict, beta_dict, frq_dict, idx_dict = parse_genet.align_ldblk(
            ref_dict, vld_dict, sst_dict, n_pop, chrom)

        mcmc_gtb.mcmc(
            1, 0.5, None, snp_dict, beta_dict, frq_dict, idx_dict,
            n_gwas, ld_blk, blk_size,
            n_iter, n_burnin, 5, pop_labels, chrom,
            out_dir, out_name, 'FALSE', 'FALSE', seed,
            n_trait=1)

    else:
        # Multi-trait mode
        sst_dict = {}
        n_gwas_dict = {}
        for pp in range(n_pop):
            for tt in range(n_trait):
                sst_dict[(pp, tt)] = parse_genet.parse_sumstats(
                    ref_dict, vld_dict, sst_files[(pp, tt)],
                    pop_labels[pp], n_gwas[(pp, tt)])
                n_gwas_dict[(pp, tt)] = n_gwas[(pp, tt)]

        ld_blk = {}
        blk_size = {}
        for pp in range(n_pop):
            ld_blk[pp], blk_size[pp] = parse_genet.parse_ldblk(
                ref_dir, sst_dict[(pp, 0)], pop_labels[pp], chrom, ref)

        snp_dict, beta_dict, frq_dict, idx_dict = parse_genet.align_ldblk_mt(
            ref_dict, vld_dict, sst_dict, n_pop, n_trait, chrom)

        # Build rho_e if overlap specified
        rho_e = None
        if rho_pheno is not None and n_overlap is not None:
            from PRScsx_mt import build_rho_e
            rho_e = build_rho_e(rho_pheno, n_overlap, n_gwas_dict, n_pop, n_trait)
            print('... sample overlap correction enabled ...')
            for pp in range(n_pop):
                print('    Pop %d rho_e matrix:' % pp)
                print(rho_e[pp])

        mcmc_gtb.mcmc(
            1, 0.5, None, snp_dict, beta_dict, frq_dict, idx_dict,
            n_gwas_dict, ld_blk, blk_size,
            n_iter, n_burnin, 5, pop_labels, chrom,
            out_dir, out_name, 'FALSE', 'FALSE', seed,
            n_trait=n_trait, rho_e=rho_e, lambda_psi=lambda_psi)

    return True


def read_effect_file(filepath):
    """Read posterior effect size file."""
    betas = []
    snps = []
    with open(filepath) as ff:
        for line in ff:
            ll = line.strip().split()
            snps.append(ll[1])
            betas.append(float(ll[5]))
    return np.array(betas), snps


def read_true_effects(filepath):
    """Read true effect file."""
    betas = []
    with open(filepath) as ff:
        header = next(ff)
        for line in ff:
            ll = line.strip().split()
            betas.append(float(ll[1]))
    return np.array(betas)


def evaluate_results(out_dir, out_name, data_dir, pop_labels, n_trait, phi_str='phiauto'):
    """Evaluate estimated effects against true effects."""
    print('\n' + '-'*40)
    print('EVALUATION')
    print('-'*40)

    results = {}
    n_pop = len(pop_labels)

    for pp in range(n_pop):
        for tt in range(n_trait):
            # Read estimated effects
            if n_trait == 1:
                eff_file = '%s/%s_%s_pst_eff_a1_b0.5_%s_chr1.txt' % (out_dir, out_name, pop_labels[pp], phi_str)
            else:
                eff_file = '%s/%s_%s_trait%d_pst_eff_a1_b0.5_%s_chr1.txt' % (out_dir, out_name, pop_labels[pp], tt, phi_str)

            if not os.path.isfile(eff_file):
                print('WARNING: output file not found: %s' % eff_file)
                continue

            beta_est, snps_est = read_effect_file(eff_file)

            # Read true effects
            true_file = '%s/true_effects_%s_trait%d.txt' % (data_dir, pop_labels[pp], tt)
            beta_true_all = read_true_effects(true_file)

            # Map true effects to estimated SNP order
            # Read the true file SNP names
            true_snps = []
            with open(true_file) as ff:
                next(ff)
                for line in ff:
                    true_snps.append(line.strip().split()[0])

            true_snp_map = {s: b for s, b in zip(true_snps, beta_true_all)}
            beta_true_aligned = np.array([true_snp_map[s] for s in snps_est])

            # Compute metrics
            corr = np.corrcoef(beta_est, beta_true_aligned)[0, 1] if np.std(beta_est) > 0 else 0.0
            mse = np.mean((beta_est - beta_true_aligned)**2)

            key = (pp, tt)
            results[key] = {'corr': corr, 'mse': mse, 'n_snp': len(beta_est)}

            print('  Pop %d (%s), Trait %d: corr=%.4f, mse=%.6f, n_snp=%d' %
                  (pp, pop_labels[pp], tt, corr, mse, len(beta_est)))

    return results


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    test_base = os.path.join(base_dir, 'test_output')

    # Clean up previous test output
    if os.path.exists(test_base):
        shutil.rmtree(test_base)

    pop_labels = ['EUR', 'EAS']
    n_pop = 2
    n_snp = 200
    n_causal = 20
    block_size = 50
    n_iter = 500
    n_burnin = 250
    seed = 42

    all_results = {}

    # ========================================
    # TEST 1: T=1 Single-Trait (backward compat)
    # ========================================
    scenario = 'test1_single_trait'
    data_dir = os.path.join(test_base, scenario, 'data')
    out_dir = os.path.join(test_base, scenario, 'output')

    sim = generate_test_data(
        data_dir, 'T=1 Single Trait', pop_labels=pop_labels,
        n_snp=n_snp, n_causal=n_causal, n_pop=n_pop, n_trait=1,
        n_gwas=[[10000], [10000]], block_size=block_size, seed=seed)

    sst_files = [
        os.path.join(data_dir, 'sst_%s_trait0.txt' % pop_labels[pp])
        for pp in range(n_pop)
    ]

    run_prscsx_mt(
        ref_dir=data_dir,
        bim_prefix=os.path.join(data_dir, 'sim_data'),
        sst_files=sst_files,
        n_gwas=[10000, 10000],
        pop_labels=pop_labels,
        out_dir=out_dir,
        out_name='test1',
        n_trait=1,
        n_iter=n_iter, n_burnin=n_burnin, seed=seed)

    all_results['T=1'] = evaluate_results(out_dir, 'test1', data_dir, pop_labels, 1)

    # ========================================
    # TEST 2: T=2 Independent Traits (rg=0)
    # ========================================
    scenario = 'test2_independent_traits'
    data_dir = os.path.join(test_base, scenario, 'data')
    out_dir = os.path.join(test_base, scenario, 'output')

    sim = generate_test_data(
        data_dir, 'T=2 Independent Traits (rg=0)', pop_labels=pop_labels,
        n_snp=n_snp, n_causal=n_causal, n_pop=n_pop, n_trait=2,
        n_gwas=[[10000, 10000], [10000, 10000]],
        rg=0.0, block_size=block_size, seed=seed)

    sst_files_mt = {}
    n_gwas_mt = {}
    for pp in range(n_pop):
        for tt in range(2):
            sst_files_mt[(pp, tt)] = os.path.join(data_dir, 'sst_%s_trait%d.txt' % (pop_labels[pp], tt))
            n_gwas_mt[(pp, tt)] = 10000

    run_prscsx_mt(
        ref_dir=data_dir,
        bim_prefix=os.path.join(data_dir, 'sim_data'),
        sst_files=sst_files_mt,
        n_gwas=n_gwas_mt,
        pop_labels=pop_labels,
        out_dir=out_dir,
        out_name='test2',
        n_trait=2,
        n_iter=n_iter, n_burnin=n_burnin, seed=seed)

    all_results['T=2 rg=0'] = evaluate_results(out_dir, 'test2', data_dir, pop_labels, 2)

    # ========================================
    # TEST 3: T=2 Correlated Traits (rg=0.8)
    # ========================================
    scenario = 'test3_correlated_traits'
    data_dir = os.path.join(test_base, scenario, 'data')
    out_dir = os.path.join(test_base, scenario, 'output')

    # Use asymmetric sample sizes: trait 1 well-powered, trait 2 underpowered
    sim = generate_test_data(
        data_dir, 'T=2 Correlated Traits (rg=0.8), asymmetric n', pop_labels=pop_labels,
        n_snp=n_snp, n_causal=n_causal, n_pop=n_pop, n_trait=2,
        n_gwas=[[20000, 5000], [20000, 5000]],
        rg=0.8, block_size=block_size, seed=seed)

    sst_files_mt = {}
    n_gwas_mt = {}
    for pp in range(n_pop):
        for tt in range(2):
            sst_files_mt[(pp, tt)] = os.path.join(data_dir, 'sst_%s_trait%d.txt' % (pop_labels[pp], tt))
            n_gwas_mt[(pp, tt)] = [20000, 5000][tt]

    run_prscsx_mt(
        ref_dir=data_dir,
        bim_prefix=os.path.join(data_dir, 'sim_data'),
        sst_files=sst_files_mt,
        n_gwas=n_gwas_mt,
        pop_labels=pop_labels,
        out_dir=out_dir,
        out_name='test3',
        n_trait=2,
        n_iter=n_iter, n_burnin=n_burnin, seed=seed)

    all_results['T=2 rg=0.8'] = evaluate_results(out_dir, 'test3', data_dir, pop_labels, 2)

    # ========================================
    # TEST 4: T=2 with Sample Overlap
    # ========================================
    scenario = 'test4_sample_overlap'
    data_dir = os.path.join(test_base, scenario, 'data')
    out_dir = os.path.join(test_base, scenario, 'output')

    sim = generate_test_data(
        data_dir, 'T=2 with Sample Overlap (rg=0.5, rho_pheno=0.3)', pop_labels=pop_labels,
        n_snp=n_snp, n_causal=n_causal, n_pop=n_pop, n_trait=2,
        n_gwas=[[10000, 10000], [10000, 10000]],
        rg=0.5, rho_pheno=0.3, n_overlap=[5000, 5000],
        block_size=block_size, seed=seed)

    sst_files_mt = {}
    n_gwas_mt = {}
    for pp in range(n_pop):
        for tt in range(2):
            sst_files_mt[(pp, tt)] = os.path.join(data_dir, 'sst_%s_trait%d.txt' % (pop_labels[pp], tt))
            n_gwas_mt[(pp, tt)] = 10000

    run_prscsx_mt(
        ref_dir=data_dir,
        bim_prefix=os.path.join(data_dir, 'sim_data'),
        sst_files=sst_files_mt,
        n_gwas=n_gwas_mt,
        pop_labels=pop_labels,
        out_dir=out_dir,
        out_name='test4',
        n_trait=2,
        rho_pheno=0.3,
        n_overlap=[5000, 5000],
        n_iter=n_iter, n_burnin=n_burnin, seed=seed)

    all_results['T=2 overlap'] = evaluate_results(out_dir, 'test4', data_dir, pop_labels, 2)

    # ========================================
    # SUMMARY
    # ========================================
    print('\n' + '='*60)
    print('SUMMARY OF ALL TESTS')
    print('='*60)
    for scenario_name, results in all_results.items():
        print('\n%s:' % scenario_name)
        for (pp, tt), metrics in sorted(results.items()):
            print('  Pop %d, Trait %d: corr=%.4f, mse=%.6f' %
                  (pp, tt, metrics['corr'], metrics['mse']))

    # Verify key expectations
    print('\n' + '='*60)
    print('VERIFICATION CHECKS')
    print('='*60)

    # Check 1: All tests produced output
    all_passed = True
    for name, res in all_results.items():
        if len(res) == 0:
            print('FAIL: %s produced no output' % name)
            all_passed = False
        else:
            print('PASS: %s produced output for %d pop-trait pairs' % (name, len(res)))

    # Check 2: Correlations should be positive (estimating the right direction)
    for name, res in all_results.items():
        for key, metrics in res.items():
            if metrics['corr'] < 0:
                print('WARN: %s Pop %d Trait %d has negative correlation (%.4f)' %
                      (name, key[0], key[1], metrics['corr']))
            elif metrics['corr'] > 0.1:
                print('PASS: %s Pop %d Trait %d has positive correlation (%.4f)' %
                      (name, key[0], key[1], metrics['corr']))

    # Check 3: T=2 rg=0.8 should help the underpowered trait
    if 'T=2 rg=0.8' in all_results and 'T=1' in all_results:
        print('\nCorrelated traits should help underpowered trait 1:')
        for pp in range(n_pop):
            if (pp, 1) in all_results.get('T=2 rg=0.8', {}):
                corr_mt = all_results['T=2 rg=0.8'][(pp, 1)]['corr']
                print('  Pop %d Trait 1 (underpowered, n=5000): corr=%.4f' % (pp, corr_mt))

    print('\n' + '='*60)
    if all_passed:
        print('ALL TESTS COMPLETED SUCCESSFULLY')
    else:
        print('SOME TESTS HAD ISSUES — see details above')
    print('='*60)


if __name__ == '__main__':
    main()
