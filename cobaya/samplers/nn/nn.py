"""
.. module:: samplers.nnest

:Synopsis: Interface for the NNest nested sampler
:Author: Adam Moss
"""
# Python 2/3 compatibility
from __future__ import absolute_import, division, print_function

# Global
import os
import sys
import numpy as np
import logging
from itertools import chain

# Local
from cobaya.tools import read_dnumber, get_external_function, relative_to_int
from cobaya.sampler import Sampler
from cobaya.mpi import get_mpi_comm
from cobaya.mpi import am_single_or_primary_process, more_than_one_process, sync_processes
from cobaya.log import LoggedError
from cobaya.install import download_github_release
from cobaya.yaml import yaml_dump_file


class nn(Sampler):
    def initialize(self):
        """Imports the NNest sampler and prepares its arguments."""
        if am_single_or_primary_process():  # rank = 0 (MPI master) or None (no MPI)
            self.log.info("Initializing")
        # If path not given, try using general path to modules
        if not self.path and self.path_install:
            self.path = get_path(self.path_install)
        if self.path:
            if am_single_or_primary_process():
                self.log.info("Importing *local* NNest from " + self.path)
                if not os.path.exists(os.path.realpath(self.path)):
                    raise LoggedError(self.log, "The given path does not exist. "
                                                "Try installing NNest with "
                                                "'cobaya-install nnest -m [modules_path]")
            pc_build_path = get_build_path(self.path)
            if not pc_build_path:
                raise LoggedError(self.log, "Either NNest is not in the given folder, "
                                            "'%s', or you have not compiled it.", self.path)
            # Inserting the previously found path into the list of import folders
            sys.path.insert(0, pc_build_path)
        else:
            self.log.info("Importing *global* NNest.")
        try:
            import nnest
            self.nnest = nnest
        except ImportError:
            raise LoggedError(
                self.log, "Couldn't find the NNest python interface. "
                          "Make sure that you have compiled it, and that you either\n"
                          " (a) specify a path (you didn't) or\n"
                          " (b) install the Python interface globally with\n"
                          "     '/path/to/NNest/python setup.py install --user'")
        # Prepare arguments and settings
        self.nDims = self.model.prior.d()
        self.nDerived = (len(self.model.parameterization.derived_params()) +
                         len(self.model.prior) + len(self.model.likelihood._likelihoods))
        if self.logzero is None:
            self.logzero = np.nan_to_num(-np.inf)
        if self.max_ndead == np.inf:
            self.max_ndead = -1
        for p in ["nlive", "nprior", "max_ndead"]:
            setattr(self, p, read_dnumber(getattr(self, p), self.nDims, dtype=int))
        # Fill the automatic ones
        if getattr(self, "feedback", None) is None:
            values = {logging.CRITICAL: 0, logging.ERROR: 0, logging.WARNING: 0,
                      logging.INFO: 1, logging.DEBUG: 2}
            self.feedback = values[self.log.getEffectiveLevel()]
        try:
            output_folder = getattr(self.output, "folder")
            output_prefix = getattr(self.output, "prefix") or ""
            self.read_resume = self.resuming
        except AttributeError:
            # dummy output -- no resume!
            self.read_resume = False
            from tempfile import gettempdir
            output_folder = gettempdir()
            if am_single_or_primary_process():
                from random import random
                output_prefix = hex(int(random() * 16 ** 6))[2:]
            else:
                output_prefix = None
            if more_than_one_process():
                output_prefix = get_mpi_comm().bcast(output_prefix, root=0)
        self.base_dir = os.path.join(output_folder, self.base_dir)
        self.file_root = output_prefix
        if am_single_or_primary_process():
            # Creating output folder, if it does not exist (just one process)
            if not os.path.exists(self.base_dir):
                os.makedirs(self.base_dir)
            self.log.info("Storing raw NNest output in '%s'.",
                          self.base_dir)
        # Exploiting the speed hierarchy
        if self.blocking:
            speeds, blocks = self.model.likelihood._check_speeds_of_params(self.blocking)
        else:
            speeds, blocks = self.model.likelihood._speeds_of_params(int_speeds=True)
        blocks_flat = list(chain(*blocks))
        self.ordering = [
            blocks_flat.index(p) for p in self.model.parameterization.sampled_params()]
        self.grade_dims = np.array([len(block) for block in blocks])
        # bugfix: pypolychord's C interface for Fortran does not like int numpy types
        self.grade_dims = [int(x) for x in self.grade_dims]
        # Steps per block
        # NB: num_repeats is ignored by PolyChord when int "grade_frac" given,
        # so needs to be applied by hand.
        # Make sure that speeds are integer, and that the slowest is 1,
        # for a straightforward application of num_repeats
        speeds = relative_to_int(speeds, 1)
        # In num_repeats, `d` is interpreted as dimension of each block
        self.grade_frac = [
            int(speed * read_dnumber(self.num_repeats, dim_block))
            for speed, dim_block in zip(speeds, self.grade_dims)]
        # prior conversion from the hypercube
        bounds = self.model.prior.bounds(
            confidence_for_unbounded=self.confidence_for_unbounded)
        # Check if priors are bounded (nan's to inf)
        inf = np.where(np.isinf(bounds))
        if len(inf[0]):
            params_names = self.model.parameterization.sampled_params()
            params = [params_names[i] for i in sorted(list(set(inf[0])))]
            raise LoggedError(
                self.log, "PolyChord needs bounded priors, but the parameter(s) '"
                          "', '".join(params) + "' is(are) unbounded.")
        locs = bounds[:, 0]
        scales = bounds[:, 1] - bounds[:, 0]
        # This function re-scales the parameters AND puts them in the right order
        self.nn_prior = lambda x: (locs + (np.array(x)[self.ordering] / 2 + 0.5) * scales).tolist()
        # We will need the volume of the prior domain, since PolyChord divides by it
        self.logvolume = np.log(np.prod(scales))
        # Prepare callback function
        if self.callback_function is not None:
            self.callback_function_callable = (
                get_external_function(self.callback_function))
        self.last_point_callback = 0
        self.n_sampled = len(self.model.parameterization.sampled_params())
        self.n_derived = len(self.model.parameterization.derived_params())
        self.n_priors = len(self.model.prior)
        self.n_likes = len(self.model.likelihood._likelihoods)
        # Done!

    def run(self):
        """
        Prepares the posterior function and calls ``PolyChord``'s ``run`` function.
        """

        # Prepare the posterior
        # Don't forget to multiply by the volume of the physical hypercube,
        # since PolyChord divides by it
        def logpost(params_values):
            logposterior, logpriors, loglikes, derived = (
                self.model.logposterior(params_values))
            if len(derived) != len(self.model.parameterization.derived_params()):
                derived = np.full(
                    len(self.model.parameterization.derived_params()), np.nan)
            if len(loglikes) != len(self.model.likelihood._likelihoods):
                loglikes = np.full(
                    len(self.model.likelihood._likelihoods), np.nan)
            derived = list(derived) + list(logpriors) + list(loglikes)
            return (
                max(logposterior + self.logvolume, 0.99 * self.pc_settings.logzero), derived)

        sync_processes()
        if am_single_or_primary_process():
            self.log.info("Sampling!")

        nn = self.nnest.NestedSampler(self.nDims,
                                      logpost,
                                      transform=self.nn_prior,
                                      append_run_num=False,
                                      log_dir=self.base_dir,
                                      num_live_points=self.nlive)

        nn.run(dlogz=self.precision_criterion,
               mcmc_batch_size=1)


# Installation routines ##################################################################

# Name of the PolyChord repo and version to download
pc_repo_name = "adammoss/nnest"
pc_repo_version = "0.4"


def get_path(path):
    return os.path.realpath(
        os.path.join(path, "code", pc_repo_name[pc_repo_name.find("/") + 1:]))


def is_installed(**kwargs):
    if not kwargs["code"]:
        return True
    try:
        import nnest
        return True
    except ImportError:
        return False


def install(path=None, force=False, code=False, data=False, no_progress_bars=False):
    if not code:
        return True
    log = logging.getLogger(__name__.split(".")[-1])
    log.info("Downloading NNest...")
    success = download_github_release(os.path.join(path, "code"), pc_repo_name,
                                      pc_repo_version, no_progress_bars=no_progress_bars,
                                      logger=log)
    if not success:
        log.error("Could not download NNest.")
        return False
    log.info("Compiling (Py)PolyChord...")
    from subprocess import Popen, PIPE
    # Needs to re-define os' PWD,
    # because MakeFile calls it and is not affected by the cwd of Popen
    cwd = os.path.join(path, "code", pc_repo_name[pc_repo_name.find("/") + 1:])
    my_env = os.environ.copy()
    my_env.update({"PWD": cwd})
    process_make = Popen([sys.executable, "setup.py", "build"],
                         cwd=cwd, env=my_env, stdout=PIPE, stderr=PIPE)
    out, err = process_make.communicate()
    if process_make.returncode:
        log.info(out)
        log.info(err)
        log.error("Python build failed!")
        return False
    return True
