#!/usr/bin/env python

"""
PRS-CSx-MT: Multi-Trait Extension of PRS-CSx.

Jointly models multiple correlated traits across populations using coupled continuous shrinkage priors
with trait-specific shrinkage modulators. When a single trait is specified, reduces exactly to PRS-CSx.

References: T Ge, CY Chen, Y Ni, YCA Feng, JW Smoller. Polygenic Prediction via Bayesian Regression and Continuous Shrinkage Priors.
           Nature Communications, 10:1776, 2019.

           Ruan Y, Lin YF, Feng YCA, Chen CY, Lam M, Guo Z, Stanley Global Asia Initiatives, He L, Sawa A, Martin AR, Qin S, Huang H, Ge T.
           Improving Polygenic Prediction in Ancestrally Diverse Populations.
           Nature Genetics, 54:573-580, 2022.

Usage:
python PRScsx_mt.py --ref_dir=PATH_TO_REFERENCE --bim_prefix=VALIDATION_BIM_PREFIX
                    --sst_file=SUM_STATS_FILES --n_gwas=GWAS_SAMPLE_SIZES
                    --pop=POPULATION --out_dir=OUTPUT_DIR --out_name=OUTPUT_FILE_PREFIX
                    [--a=PARAM_A --b=PARAM_B --phi=PARAM_PHI
                     --n_iter=MCMC_ITERATIONS --n_burnin=MCMC_BURNIN --thin=MCMC_THINNING_FACTOR
                     --chrom=CHROM --meta=META_FLAG --write_pst=WRITE_POSTERIOR_SAMPLES --seed=SEED
                     --rho_pheno=PHENOTYPIC_CORRELATION --n_overlap=OVERLAP_SAMPLE_SIZE
                     --lambda_psi=TRAIT_MODULATOR_HYPERPARAMETER]

Multi-trait format:
  --sst_file: trait1_pop1,trait1_pop2;trait2_pop1,trait2_pop2  (semicolon separates traits)
  --n_gwas:   n_trait1_pop1,n_trait1_pop2;n_trait2_pop1,n_trait2_pop2
  --rho_pheno: phenotypic correlation between traits (single float for T=2)
  --n_overlap: overlap sample size per population (comma-separated)
  --lambda_psi: hyperparameter for trait-specific modulators (default 1.0; cross_trait=mult only)
  --cross_trait: cross-trait sharing scheme — mult (default, multiplicative psi_trait),
                 mvcs (per-ancestry multivariate continuous shrinkage, signed cross-trait R),
                 kron (separable Kronecker R_trait (x) R_anc, signed across traits & ancestries)

Single-trait format (backward compatible with PRS-CSx):
  --sst_file: pop1_sst,pop2_sst
  --n_gwas:   n_pop1,n_pop2

"""


import os
import sys
import getopt
import multiprocessing
import numpy as np

import parse_genet_mt as parse_genet
import mcmc_gtb_mt as mcmc_gtb
import gigrnd


def build_rho_e(rho_pheno, n_overlap, n_gwas_per_pop, n_pop, n_trait):
    """
    Construct per-population error correlation matrices from phenotypic correlation
    and sample overlap information.

    Parameters
    ----------
    rho_pheno : float or None
        Phenotypic correlation between traits.
    n_overlap : list or None
        Number of overlapping samples per population.
    n_gwas_per_pop : dict
        Sample sizes keyed by (pp, tt).
    n_pop : int
        Number of populations.
    n_trait : int
        Number of traits.

    Returns
    -------
    rho_e : dict or None
        Per-population error correlation matrices (n_trait x n_trait), keyed by pp.
        None if no overlap correction needed.
    """
    if rho_pheno is None or n_overlap is None or n_trait == 1:
        return None

    rho_e = {}
    for pp in range(n_pop):
        R = np.eye(n_trait)
        for tt1 in range(n_trait):
            for tt2 in range(tt1+1, n_trait):
                # Error correlation = rho_pheno * sqrt(n_overlap^2 / (n1 * n2))
                n1 = n_gwas_per_pop[(pp, tt1)]
                n2 = n_gwas_per_pop[(pp, tt2)]
                n_ov = n_overlap[pp]
                rho = rho_pheno * n_ov / np.sqrt(n1 * n2)
                R[tt1, tt2] = rho
                R[tt2, tt1] = rho
        rho_e[pp] = R
    return rho_e


def parse_param():
    long_opts_list = ['ref_dir=', 'bim_prefix=', 'sst_file=', 'a=', 'b=', 'phi=', 'n_gwas=', 'pop=',
                      'n_iter=', 'n_burnin=', 'thin=', 'out_dir=', 'out_name=', 'chrom=', 'meta=', 'write_pst=', 'seed=',
                      'rho_pheno=', 'n_overlap=', 'lambda_psi=', 'cross_trait=', 'n_jobs=', 'help']

    param_dict = {'ref_dir': None, 'bim_prefix': None, 'sst_file': None, 'a': 1, 'b': 0.5, 'phi': None, 'n_gwas': None, 'pop': None,
                  'n_iter': None, 'n_burnin': None, 'thin': 5, 'out_dir': None, 'out_name': None, 'chrom': range(1,23),
                  'meta': 'FALSE', 'write_pst': 'FALSE', 'seed': None,
                  'rho_pheno': None, 'n_overlap': None, 'lambda_psi': 1.0, 'cross_trait': 'mult', 'n_jobs': 1}

    print('\n')

    if len(sys.argv) > 1:
        try:
            opts, args = getopt.getopt(sys.argv[1:], "h", long_opts_list)
        except:
            print('* Option not recognized.')
            print('* Use --help for usage information.\n')
            sys.exit(2)

        for opt, arg in opts:
            if opt == "-h" or opt == "--help":
                print(__doc__)
                sys.exit(0)
            elif opt == "--ref_dir": param_dict['ref_dir'] = arg
            elif opt == "--bim_prefix": param_dict['bim_prefix'] = arg
            elif opt == "--sst_file": param_dict['sst_file'] = arg
            elif opt == "--a": param_dict['a'] = float(arg)
            elif opt == "--b": param_dict['b'] = float(arg)
            elif opt == "--phi": param_dict['phi'] = float(arg)
            elif opt == "--n_gwas": param_dict['n_gwas'] = arg
            elif opt == "--pop": param_dict['pop'] = arg.split(',')
            elif opt == "--n_iter": param_dict['n_iter'] = int(arg)
            elif opt == "--n_burnin": param_dict['n_burnin'] = int(arg)
            elif opt == "--thin": param_dict['thin'] = int(arg)
            elif opt == "--out_dir": param_dict['out_dir'] = arg
            elif opt == "--out_name": param_dict['out_name'] = arg
            elif opt == "--chrom": param_dict['chrom'] = arg.split(',')
            elif opt == "--meta": param_dict['meta'] = arg.upper()
            elif opt == "--write_pst": param_dict['write_pst'] = arg.upper()
            elif opt == "--seed": param_dict['seed'] = int(arg)
            elif opt == "--rho_pheno": param_dict['rho_pheno'] = float(arg)
            elif opt == "--n_overlap": param_dict['n_overlap'] = list(map(int, arg.split(',')))
            elif opt == "--lambda_psi": param_dict['lambda_psi'] = float(arg)
            elif opt == "--cross_trait": param_dict['cross_trait'] = arg.lower()
            elif opt == "--n_jobs": param_dict['n_jobs'] = int(arg)
    else:
        print(__doc__)
        sys.exit(0)

    # Validate required parameters
    if param_dict['ref_dir'] == None:
        print('* Please specify the directory to the reference panel using --ref_dir\n')
        sys.exit(2)
    elif param_dict['bim_prefix'] == None:
        print('* Please specify the directory and prefix of the bim file for the target (validation/testing) dataset using --bim_prefix\n')
        sys.exit(2)
    elif param_dict['sst_file'] == None:
        print('* Please provide at least one summary statistics file using --sst_file\n')
        sys.exit(2)
    elif param_dict['n_gwas'] == None:
        print('* Please provide the sample size of the GWAS using --n_gwas\n')
        sys.exit(2)
    elif param_dict['pop'] == None:
        print('* Please specify the population of the GWAS sample using --pop\n')
        sys.exit(2)
    elif param_dict['out_dir'] == None:
        print('* Please specify the output directory using --out_dir\n')
        sys.exit(2)
    elif param_dict['out_name'] == None:
        print('* Please specify the prefix of the output file using --out_name\n')
        sys.exit(2)

    # Parse multi-trait sst_file and n_gwas
    # Format: trait1_pop1,trait1_pop2;trait2_pop1,trait2_pop2
    sst_raw = param_dict['sst_file']
    n_gwas_raw = param_dict['n_gwas']

    if ';' in sst_raw:
        # Multi-trait mode
        sst_traits = sst_raw.split(';')
        n_gwas_traits = n_gwas_raw.split(';')
        n_trait = len(sst_traits)

        sst_file_mt = {}  # (pp, tt) -> filename
        n_gwas_mt = {}    # (pp, tt) -> sample size
        n_pop = len(param_dict['pop'])

        for tt, (sst_t, n_t) in enumerate(zip(sst_traits, n_gwas_traits)):
            sst_pops = sst_t.split(',')
            n_pops = list(map(int, n_t.split(',')))
            if len(sst_pops) != n_pop or len(n_pops) != n_pop:
                print('* Length of sst_file and n_gwas for trait %d does not match number of populations\n' % tt)
                sys.exit(2)
            for pp in range(n_pop):
                sst_file_mt[(pp, tt)] = sst_pops[pp]
                n_gwas_mt[(pp, tt)] = n_pops[pp]

        param_dict['sst_file_mt'] = sst_file_mt
        param_dict['n_gwas_mt'] = n_gwas_mt
        param_dict['n_trait'] = n_trait
        # Also store flat lists for display
        param_dict['sst_file'] = [sst_file_mt[(pp, tt)] for tt in range(n_trait) for pp in range(n_pop)]
        param_dict['n_gwas'] = [n_gwas_mt[(pp, tt)] for tt in range(n_trait) for pp in range(n_pop)]
    else:
        # Single-trait mode (backward compatible)
        param_dict['sst_file'] = sst_raw.split(',')
        param_dict['n_gwas'] = list(map(int, n_gwas_raw.split(',')))
        param_dict['n_trait'] = 1
        n_pop = len(param_dict['pop'])

        if (len(param_dict['sst_file']) != len(param_dict['n_gwas']) or
            len(param_dict['sst_file']) != n_pop):
            print('* Length of sst_file, n_gwas and pop does not match\n')
            sys.exit(2)

    if param_dict['cross_trait'] not in ('mult', 'mvcs', 'kron'):
        print('* --cross_trait must be one of: mult, mvcs, kron\n')
        sys.exit(2)

    n_pop = len(param_dict['pop'])
    if param_dict['n_iter'] == None or param_dict['n_burnin'] == None:
        param_dict['n_iter'] = n_pop*1000
        param_dict['n_burnin'] = n_pop*500

    # Validate overlap parameters
    if param_dict['n_overlap'] is not None:
        if len(param_dict['n_overlap']) != n_pop:
            print('* Length of n_overlap does not match number of populations\n')
            sys.exit(2)
        if param_dict['rho_pheno'] is None:
            print('* rho_pheno is required when n_overlap is specified\n')
            sys.exit(2)

    for key in param_dict:
        print('--%s=%s' % (key, param_dict[key]))

    print('\n')
    return param_dict


def process_chrom(args):
    """Process a single chromosome. Designed to run in a worker process."""
    chrom, param_dict, rho_e, ref = args
    chrom = int(chrom)
    n_pop = len(param_dict['pop'])
    n_trait = param_dict['n_trait']

    print('##### process chromosome %d #####' % chrom)

    if ref == '1kg':
        ref_dict = parse_genet.parse_ref(param_dict['ref_dir'] + '/snpinfo_mult_1kg_hm3', chrom, ref)
    else:
        ref_dict = parse_genet.parse_ref(param_dict['ref_dir'] + '/snpinfo_mult_ukbb_hm3', chrom, ref)

    vld_dict = parse_genet.parse_bim(param_dict['bim_prefix'], chrom)

    # Per-chromosome seed: offset by chrom so parallel runs are independent
    seed = param_dict['seed']
    if seed is not None:
        seed = seed + chrom

    if n_trait == 1:
        sst_dict = {}
        for pp in range(n_pop):
            sst_dict[pp] = parse_genet.parse_sumstats(ref_dict, vld_dict, param_dict['sst_file'][pp], param_dict['pop'][pp], param_dict['n_gwas'][pp])

        ld_blk = {}
        blk_size = {}
        for pp in range(n_pop):
            ld_blk[pp], blk_size[pp] = parse_genet.parse_ldblk(param_dict['ref_dir'], sst_dict[pp], param_dict['pop'][pp], chrom, ref)

        snp_dict, beta_dict, frq_dict, idx_dict = parse_genet.align_ldblk(ref_dict, vld_dict, sst_dict, n_pop, chrom)

        mcmc_gtb.mcmc(param_dict['a'], param_dict['b'], param_dict['phi'], snp_dict, beta_dict, frq_dict, idx_dict, param_dict['n_gwas'], ld_blk, blk_size,
            param_dict['n_iter'], param_dict['n_burnin'], param_dict['thin'], param_dict['pop'], chrom,
            param_dict['out_dir'], param_dict['out_name'], param_dict['meta'], param_dict['write_pst'], seed,
            n_trait=1)

    else:
        sst_file_mt = param_dict['sst_file_mt']
        n_gwas_mt = param_dict['n_gwas_mt']

        sst_dict = {}
        for pp in range(n_pop):
            for tt in range(n_trait):
                sst_dict[(pp, tt)] = parse_genet.parse_sumstats(
                    ref_dict, vld_dict, sst_file_mt[(pp, tt)],
                    param_dict['pop'][pp], n_gwas_mt[(pp, tt)])

        ld_blk = {}
        blk_size = {}
        for pp in range(n_pop):
            ld_blk[pp], blk_size[pp] = parse_genet.parse_ldblk(
                param_dict['ref_dir'], sst_dict[(pp, 0)],
                param_dict['pop'][pp], chrom, ref)

        snp_dict, beta_dict, frq_dict, idx_dict = parse_genet.align_ldblk_mt(
            ref_dict, vld_dict, sst_dict, n_pop, n_trait, chrom)

        mcmc_gtb.mcmc(param_dict['a'], param_dict['b'], param_dict['phi'], snp_dict, beta_dict, frq_dict, idx_dict,
            n_gwas_mt, ld_blk, blk_size,
            param_dict['n_iter'], param_dict['n_burnin'], param_dict['thin'], param_dict['pop'], chrom,
            param_dict['out_dir'], param_dict['out_name'], param_dict['meta'], param_dict['write_pst'], seed,
            n_trait=n_trait, rho_e=rho_e, lambda_psi=param_dict['lambda_psi'], cross_trait=param_dict['cross_trait'])

    print('\n')


def main():
    param_dict = parse_param()
    n_pop = len(param_dict['pop'])
    n_trait = param_dict['n_trait']

    if n_pop == 1:
        print('*** only %d discovery population detected ***\n' % n_pop)
    else:
        print('*** %d discovery populations detected ***\n' % n_pop)

    if n_trait > 1:
        print('*** %d traits detected — multi-trait mode ***\n' % n_trait)

    # Build rho_e for sample overlap correction
    rho_e = None
    if n_trait > 1:
        rho_e = build_rho_e(
            param_dict['rho_pheno'],
            param_dict['n_overlap'],
            param_dict.get('n_gwas_mt', {}),
            n_pop, n_trait
        )

    # Detect reference panel type once
    if os.path.isfile(param_dict['ref_dir'] + '/snpinfo_mult_1kg_hm3'):
        ref = '1kg'
    elif os.path.isfile(param_dict['ref_dir'] + '/snpinfo_mult_ukbb_hm3'):
        ref = 'ukbb'
    else:
        print('* Reference panel not found in %s\n' % param_dict['ref_dir'])
        sys.exit(2)

    chroms = [int(c) for c in param_dict['chrom']]
    n_jobs = min(param_dict['n_jobs'], len(chroms))
    args = [(chrom, param_dict, rho_e, ref) for chrom in chroms]

    if n_jobs == 1:
        for a in args:
            process_chrom(a)
    else:
        print('*** running %d chromosomes in parallel (n_jobs=%d) ***\n' % (len(chroms), n_jobs))
        with multiprocessing.Pool(processes=n_jobs) as pool:
            pool.map(process_chrom, args)


if __name__ == '__main__':
    main()
