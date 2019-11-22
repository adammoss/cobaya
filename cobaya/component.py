from cobaya.log import HasLogger
from cobaya.input import HasDefaults
from cobaya.tools import get_class_methods
from cobaya.log import LoggedError
import numpy as np
import time
from collections import OrderedDict


class Timer(object):
    def __init__(self):
        self.n = 0
        self.time_avg = 0
        self.time_sqsum = 0
        self.time_std = np.inf  # not actually used?
        self._start = None
        self._time_func = getattr(time, "perf_counter", time.time)

    def start(self):
        self._start = self._time_func()

    def increment(self, logger=None):
        delta_time = self._time_func() - self._start
        self.n += 1
        # TODO: Protect against caching in first call by discarding first time
        # if too different from second one (maybe check difference in log10 > 1)
        # In that case, take into account that #timed_evals is now (self.n - 1)
        if logger and self.n == 2:
            if delta_time:
                log10diff = np.log10(self.time_avg / delta_time)
                if log10diff > 1:
                    logger.warning(
                        "It seems the first call has done some caching (difference "
                        " of a factor %g). Average timing will not be reliable "
                        "unless many evaluations are carried out.", 10 ** log10diff)
            else:
                logger.warning("Timing returning zero, may be inaccurate. First call may "
                               "have done some caching.")
        self.time_avg = (delta_time + self.time_avg * (self.n - 1)) / self.n
        self.time_sqsum += delta_time ** 2
        if self.n > 1:
            self.time_std = np.sqrt(
                (self.time_sqsum - self.n * self.time_avg ** 2) / (self.n - 1))
        if logger:
            logger.debug("Average evaluation time: %g s", self.time_avg)


class CobayaComponent(HasLogger, HasDefaults):
    """
    Base class for a theory, likelihood or sampler with associated .yaml parameter
    that can set attributes.
    """

    class_options = {}

    def __init__(self, info={}, name=None, timing=None, path_install=None,
                 initialize=True):
        self._name = name or self.get_qualified_class_name()
        self.path_install = path_install
        for k, value in self.class_options.items():
            setattr(self, k, value)
        # set attributes from the info (usually from yaml file)
        for k, value in info.items():
            setattr(self, k, value)
        self.set_logger(name=self._name)
        if timing:
            self.timer = Timer()
        else:
            self.timer = None
        try:
            if initialize:
                self.initialize()
        except AttributeError as e:
            if '_params' in str(e):
                raise LoggedError(self.log, "use 'initialize_with_params' if you need to "
                                            "initialize after input and output parameters"
                                            " are set (%s, %s)", self, e)
            raise

    def get_name(self):
        return self._name

    def __repr__(self):
        return self.get_name()

    def close(self):
        """Finalizes the class, if something needs to be cleaned up."""
        pass

    def get_requirements(self):
        """
        Get a dictionary of requirements (e.g. calculated by a another component)
        :return: dictionary of requirements
        """
        return {}

    def needs(self, **requirements):
        """
        Function to be called specifying any output products that are needed and hence
        should be calculated by this component.
        Requirements is a dictionary of requirement names with optional parameters for
        each.
        """
        pass

    def initialize(self):
        """
        Initializes the class (before getting requirements and before input_params,
        output_params and provider assigned).
        """
        pass

    def initialize_with_params(self):
        """
        Additional initialization after requirements called and input_params and
        output_params hjve been assigned (but provider and needs assigned).
        """
        pass

    def initialize_with_provider(self, provider):
        """
        Final initialization after parameters, provider and needs assigned
        """
        self.provider = provider

    def get_can_provide_methods(self):
        """
        Get a dictionary of quantities X that can be retrieved using get_X methods.

        :return: dictionary of the form {X: get_X method}
        """
        return get_class_methods(self.__class__, not_base=CobayaComponent)

    def get_can_provide_params(self):
        """
        Get a list of derived parameters that this component can calculate.

        :return: list of parameter names
        """
        return []

    def get_allow_agnostic(self):
        """
        Whether it is allowed to pass all unassigned input parameters to this component
        (True) or whether parameters must be explicitly specified.

        :return: True or False
        """
        return False

    def __exit__(self, exception_type, exception_value, traceback):
        if self.timer:
            self.log.info("Average evaluation time for %s: %g s  (%d evaluations)" % (
                self.get_name(), self.timer.time_avg, self.timer.n))
        self.close()


class ComponentCollection(OrderedDict, HasLogger):
    """
    Base class for an ordered dictionary of components (e.g. likelihoods or theories)
    """

    def __init__(self):
        super(ComponentCollection, self).__init__()

    def dump_timing(self):
        timers = [component for component in self.values() if component.timer]
        if timers:
            sep = "\n   "
            self.log.info(
                "Average computation time:" + sep + sep.join(
                    ["%s : %g s (%d evaluations)" % (
                        component.get_name(), component.timer.time_avg, component.timer.n)
                     for component in timers]))

    # Python magic for the "with" statement
    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        for component in self.values():
            component.__exit__(exception_type, exception_value, traceback)


class Provider(object):
    """
    Class used to retrieve computed requirements.
    Just passes on get_X and get_param methods to the correct component.
    """

    def __init__(self, model, requirement_providers):
        self.model = model
        self.requirement_providers = requirement_providers
        self.params = None

    def set_current_input_params(self, params):
        self.params = params

    def get_param(self, param):
        if param in self.params:
            return self.params[param]
        else:
            return self.requirement_providers[param].get_param(param)

    def __getattribute__(self, name):
        if name.startswith('get_'):
            requirement = name[4:]
            if requirement == 'param':
                return object.__getattribute__(self, name)
            else:
                return getattr(self.requirement_providers[requirement], name)
        return object.__getattribute__(self, name)
