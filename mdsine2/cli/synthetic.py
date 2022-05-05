
lsfstr = '''#!/bin/bash
#BSUB -J {jobname}
#BSUB -o {stdout_loc}
#BSUB -e {stderr_loc}
#BSUB -q {queue}
#BSUB -n {cpus}
#BSUB -M {mem}
#BSUB -R rusage[mem={mem}]
echo '---PROCESS RESOURCE LIMITS---'
ulimit -a
echo '---SHARED LIBRARY PATH---'
echo $LD_LIBRARY_PATH
echo '---APPLICATION SEARCH PATH:---'
echo $PATH
echo '---LSF Parameters:---'
printenv | grep '^LSF'
echo '---LSB Parameters:---'
printenv | grep '^LSB'
echo '---LOADED MODULES:---'
module list
echo '---SHELL:---'
echo $SHELL
echo '---HOSTNAME:---'
hostname
echo '---GROUP MEMBERSHIP (files are created in the first group listed):---'
groups
echo '---DEFAULT FILE PERMISSIONS (UMASK):---'
umask
echo '---CURRENT WORKING DIRECTORY:---'
pwd
echo '---DISK SPACE QUOTA---'
df .
echo '---TEMPORARY SCRATCH FOLDER ($TMPDIR):---'
echo $TMPDIR
# Load the environment
module load anaconda/4.8.2
source activate {environment_name}
cd {code_basepath}



# Run the metrics file 
# -------------
mdsine2 metrics\
    --input-mcmc-low {input_mcmc_low} \
    --input-mcmc-med {input_mcmc_med} \
    --input-mcmc-high {input_mcmc_high} \
    --input-semi-syn {input_semi_syn} \
    --seed {seed} \
    --rename-study {rename_study} \
    --output-basepath {basepath} \
        

'''


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


# Create semi-synthetic data object--with selected noise levels--from synthetic system
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
        parser.add_argument('--a0-level-low', '-a0l', type=float, dest='a0_level_low',
                            help='This is the a0-noise-level low for the synthetic data-generation parameter')
        parser.add_argument('--a1-level-low', '-a1l', type=float, dest='a1_level_low',
                            help='This is the a1-noise-level low for the synthetic data-generation parameter')
        parser.add_argument('--qpcr-level-low', '-qpcrl', type=float, dest='qpcr_level_low',
                            help='This is the qpcr-noise-level low for the synthetic data-generation parameter')
        parser.add_argument('--a0-level-med', '-a0m', type=float, dest='a0_level_med',
                            help='This is the a0-noise-level medium for the synthetic data-generation parameter')
        parser.add_argument('--a1-level-med', '-a1m', type=float, dest='a1_level_med',
                            help='This is the a1-noise-level medium for the synthetic data-generation parameter')
        parser.add_argument('--qpcr-level-med', '-qpcrm', type=float, dest='qpcr_level_med',
                            help='This is the qpcr-noise-level medium for the synthetic data-generation parameter')
        parser.add_argument('--a0-level-high', '-a0h', type=float, dest='a0_level_high',
                            help='This is the a0-noise-level high for the synthetic data-generation parameter')
        parser.add_argument('--a1-level-high', '-a1h', type=float, dest='a1_level_high',
                            help='This is the a1-noise-level high for the synthetic data-generation parameter')
        parser.add_argument('--qpcr-level-high', '-qpcrh', type=float, dest='qpcr_level_high',
                            help='This is the qpcr-noise-level high for the synthetic data-generation parameter')

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

        # Erisone Parameters
        parser.add_argument('--lsf-basepath', '-l', type=str, dest='lsf_basepath',
                            help='This is the basepath to save the lsf files', default='lsf_files/'
        )
        parser.add_argument('--environment-name', dest='environment_name', type=str,
                            help='Name of the conda environment to activate when the job starts')
        parser.add_argument('--code-basepath', type=str, dest='code_basepath',
                            help='Where the `run_cross_validation` script is located')
        parser.add_argument('--queue', '-q', type=str, dest='queue',
                            help='ErisOne queue this job gets submitted to')
        parser.add_argument('--memory', '-mem', type=str, dest='memory',
                            help='Amount of memory to reserve on ErisOne')
        parser.add_argument('--n-cpus', '-cpus', type=str, dest='cpus',
                            help='Number of cpus to reserve on ErisOne')

    def main(self, args: argparse.Namespace):
        print("Started to run main within synthetic.py")
        # 1) load dataset
        logger.info('Loading dataset {}'.format(args.input))
        mcmc = md2.BaseMCMC.load(args.input) #might be study.load instead of BaseMCMC
        study1 = md2.Study.load(args.input)
        if args.rename_study is not None:
            if args.rename_study.lower() != 'none':
                study1.name = args.rename_study #might just want to delete 153-155

        print("This is study name:" + study1.name)
        md2.seed(args.seed)

        # 2) Load the model parameters
        os.makedirs(args.basepath, exist_ok=True)
        basepath = os.path.join(args.basepath, study1.name)
        os.makedirs(basepath, exist_ok=True)

        # 3) Make semi_syn object and save to pickle for downstream metrics
        semi_syn = setup_semisynthetic_system(mcmc)
        pickle_loc_ss = basepath + "/semi_syn_" + str(args.seed) + ".pkl"
        pickle.dump(semi_syn, open(pickle_loc_ss, "wb"))  #You might need something with basepath here

        # 4) Create study object based on synthetic paramaters; later used in inference
        study_low = make_semisynthetic_object(args.a0_level_low, args.a1_level_low, args.qpcr_level_low, semi_syn)
        study_med = make_semisynthetic_object(args.a0_level_med, args.a1_level_med, args.qpcr_level_med, semi_syn)
        study_high = make_semisynthetic_object(args.a0_level_high, args.a1_level_high, args.qpcr_level_high, semi_syn)

        when_true_1 = (study_low == study_med)
        when_true_2 = (study_low == study_high)
        when_true_3 = (study_med == study_high)


        print("THE BOOLEAN RIGHT NOW:", when_true_1, when_true_2, when_true_3)



        taxa = semi_syn.taxa
        print(len(taxa))

        #5) Run inference on study object and save to pickle file for downstream metrics
        mcmc_syn_low = run_inference(study_low, basepath, args.seed, args.burnin, args.n_samples, args.checkpoint)
        pickle_loc_ms_low = basepath + "/mcmc_syn_low_" + str(args.seed) + ".pkl" #This might make double MCMC files; check to see if it does after we run it
        pickle.dump(mcmc_syn_low, open(pickle_loc_ms_low, "wb")) #This might make double MCMC files; check to see if it does after we run it

        mcmc_syn_med = run_inference(study_med, basepath, args.seed, args.burnin, args.n_samples, args.checkpoint)
        pickle_loc_ms_med = basepath + "/mcmc_syn_med_" + str(args.seed) + ".pkl"
        pickle.dump(mcmc_syn_med, open(pickle_loc_ms_med, "wb"))

        mcmc_syn_high = run_inference(study_high, basepath, args.seed, args.burnin, args.n_samples, args.checkpoint)
        pickle_loc_ms_high = basepath + "/mcmc_syn_high_" + str(args.seed) + ".pkl"
        pickle.dump(mcmc_syn_high, open(pickle_loc_ms_high, "wb"))

        ####------LSF Creation Section Starts------#
        # Make the arguments
        jobname = args.rename_study

        lsfdir = args.lsf_basepath

        script_path = os.path.join(lsfdir, 'scripts')
        stdout_loc = os.path.abspath(os.path.join(lsfdir, 'stdout'))
        stderr_loc = os.path.abspath(os.path.join(lsfdir, 'stderr'))
        os.makedirs(script_path, exist_ok=True)
        os.makedirs(stdout_loc, exist_ok=True)
        os.makedirs(stderr_loc, exist_ok=True)
        stdout_loc = os.path.join(stdout_loc, jobname + '.out')
        stderr_loc = os.path.join(stderr_loc, jobname + '.err')

        os.makedirs(lsfdir, exist_ok=True)
        lsfname = os.path.join(script_path, jobname + '.lsf')

        f = open(lsfname, 'w')
        f.write(lsfstr.format( ##ad mcmc_syn input and semi_syn input
            jobname=jobname, stdout_loc=stdout_loc, stderr_loc=stderr_loc,
            environment_name=args.environment_name,
            code_basepath=args.code_basepath, queue=args.queue, cpus=args.cpus,
            mem=args.memory, seed=args.seed, input_mcmc_low=pickle_loc_ms_low, input_mcmc_med=pickle_loc_ms_med, input_mcmc_high=pickle_loc_ms_high,
            input_semi_syn=pickle_loc_ss, rename_study=args.rename_study, basepath=args.basepath))
        f.close()
        command = 'bsub < {}'.format(lsfname)
        print(command)
        print("Successfully got to almost end of synthetic.py")
        os.system(command)

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