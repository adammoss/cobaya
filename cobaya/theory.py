"""
.. module:: theory

:Synopsis: Prototype theory class and theory loader
:Author: Jesus Torrado

If you are using a experimental likelihood, chances are the you will need a theoretical
code to compute the observables needed to compute the likelihood.

This module contains the prototype of the theory code and the loader for the requested
code.

.. note::

   At this moment, of all modules of cobaya, this is the one with the least fixed
   structure. Don't pay much attention to it for now. Just go on to the documentation of
   :doc:`CAMB <theory_camb>` and :doc:`CLASS <theory_class>`.

"""

# Python 2/3 compatibility
from __future__ import division

# Local
from cobaya.conventions import _input_params, _output_params
from cobaya.log import HasLogger
from cobaya.input import HasDefaults

from scipy.integrate import quad
from astropy.utils import isiterable
import numpy as np

# Default options for all subclasses
class_options = {"speed": -1}


# Theory code prototype
class Theory(HasLogger, HasDefaults):
    """Prototype of the theory class."""

    def initialize(self):
        """
        Initializes the theory code: imports the theory code, if it is an external one,
        and makes any necessary preparations.
        """
        pass

    def needs(self, arguments):
        """
        Function to be called by the likelihoods at their initialization,
        to specify their requests.
        Its specific behaviour for a code must be defined.
        """
        pass

    def compute(self, **parameter_values_and_derived_dict):
        """
        Takes a dictionary of parameter values and computes the products needed by the
        likelihood.
        If passed a keyword `derived` with an empty dictionary, it populates it with the
        value of the derived parameters for the present set of sampled and fixed parameter
        values.
        """
        pass

    def close(self):
        """Finalizes the theory code, if something needs to be done
        (releasing memory, etc.)"""
        pass

    def _w_integrand(self, ln1pz):
        a = np.exp(-ln1pz)
        return 1.0 + self.w(a)

    def de_density_scale(self, z):
        if isiterable(z):
            z = np.asarray(z)
            ival = np.array([quad(self._w_integrand, 0, np.log(1 + redshift))[0]
                             for redshift in z])
            return np.exp(3 * ival)
        else:
            ival = quad(self._w_integrand, 0, np.log(1 + z))[0]
            return np.exp(3 * ival)

    # Generic methods: do not touch these

    def __init__(self, info_theory, modules=None, timing=None):
        self.name = self.__class__.__name__
        self.set_logger()
        self.path_install = modules
        # Load info of the code
        for k in info_theory:
            setattr(self, k, info_theory[k])
        # Timing
        self.timing = timing
        self.n = 0
        self.time_avg = 0

        # AJM
        max_z_early = 10000.0
        min_z_early = 500.0
        max_z_late = 3.0
        min_z_late = 0.0
        self.max_a_early = 1 / (1 + min_z_early)
        self.min_a_early = 1 / (1 + max_z_early)
        self.max_a_late = 1 / (1 + min_z_late)
        self.min_a_late = 1 / (1 + max_z_late)

        self.w = lambda a: -1.0
        self.w_min = -2.0
        self.w_max = 1.0
        self.w_bbn = -1.0
        self.w_dark_ages = -1.0
        self.w_early_bins = 0
        self.w_late_bins = 0
        self.omm_test = 0.3

    def d(self):
        """
        Dimension of the input vector.

        NB: Different from dimensionality of the sampling problem, e.g. this may include
        fixed input parameters.
        """
        return len(self.input_params)

    def __exit__(self, exception_type, exception_value, traceback):
        if self.timing:
            self.log.info("Average 'compute' evaluation time: %g s  (%d evaluations)" %
                          (self.time_avg, self.n))
        self.close()
