import mdsine2 as md2
import pickle
import os
import time
import argparse
from mdsine2.names import STRNAMES
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from pathlib import Path
import seaborn as sns
from .base import CLIModule
from mdsine2.logger import logger
from mdsine2.names import STRNAMES


# Create the semi-synthetic system
def setup_semisynthetic_system(mcmc):
    # Make the semisynthetic system
    semi_syn = md2.synthetic.make_semisynthetic(
        mcmc, min_bayes_factor=10, name='semisynth', set_times=True)

    # make subject names
    semi_syn.set_subjects(['subj-{}'.format(i + 1) for i in range(4)])

    # Set the timepoints
    semi_syn.set_timepoints(times=np.arange(77))
    semi_syn.times

    # Forward simulate
    pv = md2.model.MultiplicativeGlobal(0.01 ** 2)  # 5% process variation
    semi_syn.generate_trajectories(dt=0.01, init_dist=md2.variables.Uniform(low=2, high=15),
                                   processvar=pv)

    return semi_syn


# Create semi-synthetic data object--with selected noise levels--from synethic system
def make_semisynthetic_object(a0_level, a1_level, qpcr_level, semi_syn):
    # Simulate noise and create object
    study = semi_syn.simulateMeasurementNoise(
        a0=a0_level, a1=a1_level, qpcr_noise_scale=qpcr_level, approx_read_depth=75000,
        name='semi-synth')
    return study


# run inference with new synthetic object -- DONE
def run_inference(study, basepath, seed, burnin, n_samples, checkpoint):
    #Inference paramaters and settings
    params = md2.config.MDSINE2ModelConfig(
        basepath=basepath,
        seed=seed,
        burnin=burnin,
        n_samples=n_samples,
        checkpoint=checkpoint,
        negbin_a0=1e-10, negbin_a1=0.001
    )

    params.INITIALIZATION_KWARGS[STRNAMES.CLUSTERING]['value_option'] = 'no-clusters'
    mcmc_semisyn = md2.initialize_graph(params=params, graph_name=study.name, subjset=study)

    clustering = mcmc_semisyn.graph[STRNAMES.CLUSTERING_OBJ]
    print(clustering.coclusters.value.shape)
    print(len(clustering))

    mcmc_syn = md2.run_graph(mcmc_semisyn, crash_if_error=True)

    return mcmc_syn

class SyntheticCLI(CLIModule):
    def __init__(self, subcommand="synthetic"):
        super().__init__(
            subcommand=subcommand,
            docstring=__doc__
        )

    def create_parser(self, parser: argparse.ArgumentParser):
        parser.add_argument(
            '--input', '-i', type=str, dest='input',
            required=True,
            help='This is the dataset to do inference with.'
        )

        parser.add_argument(
            '--negbin', type=str, dest='negbin', nargs='+',
            required=True,
            help='If there is a single argument, then this is the MCMC object that was run to ' \
                 'learn a0 and a1. If there are two arguments passed, these are the a0 and a1 ' \
                 'of the negative binomial dispersion parameters. Example: ' \
                 '--negbin /path/to/negbin/mcmc.pkl. Example: ' \
                 '--negbin 0.0025 0.025'
        )
        parser.add_argument(
            '--seed', '-s', type=int, dest='seed',
            required=True,
            help='This is the seed to initialize the inference with'
        )
        parser.add_argument(
            '--a0-level', '-a0', type=float, dest='a0_level',
            required=True,
            help='This is the a0-noise-level for the synthetic data-generation parameter'
        )
        parser.add_argument(
            '--a1-level', '-a1', type=float, dest='a1_level',
            required=True,
            help='This is the a1-noise-level for the synthetic data-generation parameter'
        )
        parser.add_argument(
            '--qpcr-level', '-qpcr', type=float, dest='qpcr_level',
            required=True,
            help='This is the qpcr-noise-level for the synthetic data-generation parameter'
        )
        parser.add_argument(
            '--burnin', '-nb', type=int, dest='burnin',
            required=True,
            help='How many burn-in Gibb steps for Markov Chain Monte Carlo (MCMC)'
        )
        parser.add_argument(
            '--n-samples', '-ns', type=int, dest='n_samples',
            required=True,
            help='Total number Gibb steps to perform during MCMC inference'
        )
        parser.add_argument(
            '--checkpoint', '-c', type=int, dest='checkpoint',
            required=True,
            help='How often to write the posterior to disk. Note that `--burnin` and ' \
                 '`--n-samples` must be a multiple of `--checkpoint` (e.g. checkpoint = 100, ' \
                 'n_samples = 600, burnin = 300)'
        )
        # parser.add_argument(
        #     '--multiprocessing', '-mp', type=int, dest='mp',
        #     help='If 1, run the inference with multiprocessing. Else run on a single process',
        #     default=0
        # )
        parser.add_argument(
            '--rename-study', type=str, dest='rename_study',
            required=False, default=None,
            help='Specify the name of the study to set'
        )
        parser.add_argument(
            '--basepath', '--output-basepath', '-b', type=str, dest='basepath',
            required=True,
            help='This is folder to save the output of inference'
        )

    def main(self, args: argparse.Namespace):
        print("Started to run main within synthetic.py")
        # 1) load dataset
        logger.info('Loading dataset {}'.format(args.input))
        mcmc = md2.BaseMCMC.load(args.input) #might be study.load instead of BaseMCMC
        study1 = md2.Study.load(args.input)
        if args.rename_study is not None:
            if args.rename_study.lower() != 'none':
                study1.name = args.rename_study

        md2.seed(args.seed)

        # 2) Load the model parameters
        os.makedirs(args.basepath, exist_ok=True)
        basepath = os.path.join(args.basepath, study1.name)
        os.makedirs(basepath, exist_ok=True)

        # 3) Make semi_syn object and save to pickle for downstream metrics
        semi_syn = setup_semisynthetic_system(mcmc)
        pickle_loc_ss = basepath + "/semi_syn_" + args.seed + ".pkl"
        pickle.dump(semi_syn, open(pickle_loc_ss, "wb"))  #You might need something with basepath here

        # 4) Create study object based on synthetic paramaters; later used in inference
        study = make_semisynthetic_object(args.a0_level, args.a1_level, args.qpcr_level, semi_syn)
        taxa = semi_syn.taxa
        print(len(taxa))

        #5) Run inference on study object and save to pickle file for downstream metrics
        mcmc_syn = run_inference(study, basepath, args.seed, args.burnin, args.n_samples, args.checkpoint)
        pickle_loc_ms = basepath + "/mcmc_syn_" + args.seed + ".pkl" #This might make double MCMC files; check to see if it does after we run it
        pickle.dump(mcmc_syn, open(pickle_loc_ms, "wb")) #This might make double MCMC files; check to see if it does after we run it


# -----------------------REMOVE THIS LATER------------------------
# MCMC_dir = "/PHShome/as1010/MDSINE2_Paper/analysis/output/gibson/mdsine2_as1010/fixed_clustering/healthy-seed0-strong-sparse/mcmc.pkl"
# mcmc = md2.BaseMCMC.load(MCMC_dir)
# # mcmc = md2.BaseMCMC.load(args.dset_fileloc)
# semi_syn = setup_semisynthetic_system(mcmc)
# pickle.dump(semi_syn, open("/PHShome/as1010/synthetic_new_arash/semi_syn.pkl", "wb"))
# # pickle.dump(semi_syn, "/PHShome/as1010/synthetic_new_arash/semi_syn.pkl")
# study = make_semisynthetic_object(.000001, .000003, .0001, semi_syn)
# taxa = semi_syn.taxa
# print(len(taxa))
# # mcmc_syn = run_inference(study)
# # plot_relRMSE(mcmc_syn, semi_syn, study)
# # calculate_RMSEgrowth(mcmc_syn, semi_syn)