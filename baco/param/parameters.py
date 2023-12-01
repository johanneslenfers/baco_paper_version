from abc import abstractmethod
from itertools import permutations, product
from typing import Any, List, Optional, Union, Tuple

import numpy as np
import torch
from numpy.random import beta
from scipy.stats import truncnorm


# PARAMETERS CLASSES ##################
# The input search space supports 6 types of parameters: Categorical, Ordinal, Integer, Real, Permutation and Selection 
class Parameter:
    """
    Parent class of the different parameters.
    """

    def __init__(
            self,
            name: str,
            default: Union[int, float],
            constraints: List[str],
            dependencies: List[str],
    ):
        self.name = name
        self.default = default
        self.constraints = constraints
        self.dependencies = dependencies

        if not self.dependencies:
            self.dependencies = None
        if not self.constraints:
            self.constraints = None

    def get_name(self) -> str:
        return self.name

    def get_default(self) -> Any:
        return self.default

    def get_dependencies(self) -> List[str]:
        return self.dependencies

    def get_constraints(self) -> List[str]:
        return self.constraints

    @abstractmethod
    def convert(
            self,
            input_value: Union[str, float],
            from_type: str,
            to_type: str,
    ) -> Union[str, float]:
        raise NotImplementedError


class RealParameter(Parameter):
    """
    This class defines a real (continuous) parameter.
    """

    def __init__(
            self,
            name: str,
            min_value: float,
            max_value: float,
            default: float,
            probability_distribution: Union[str, List[float], Tuple],
            preferred_discretization,
            constraints: Optional[List[str]] = None,
            dependencies: Optional[List[str]] = None,
            transform: Optional[str] = "none",
    ):
        """
        Initialization method. The possible values for this parameter are between min_value and max_value.

        Input:
             - name: variable name
             - min_value: minimum value.
             - max_value: maximum value.
             - default: default value.
             - preferred_discretization: list of discrete values.
             - probability_distribution: a string describing the probability density function.
             - constraints: not yet implemented for RealParameter.
             - dependencies: not yet implemented for RealParameter.
        """
        if dependencies is None:
            dependencies = []
        if constraints is None:
            constraints = []
        Parameter.__init__(self, name, default, constraints, dependencies)
        self.min_value = min_value
        self.max_value = max_value
        self.preferred_discretization = preferred_discretization
        self.pdf_distribution = None
        self.cdf_distribution = None
        self.transform = transform

        if isinstance(probability_distribution, str):
            self.distribution_name = probability_distribution

        elif isinstance(probability_distribution, list):
            self.distribution_name = "custom_distribution"
            self.probability_distribution = torch.tensor(probability_distribution) / np.sum(probability_distribution)
            self.cdf = np.cumsum(self.probability_distribution)

        elif isinstance(probability_distribution, tuple):
            if not len(probability_distribution[1]) == 2:
                raise Exception("gaussian prior requires two prior_parameters, mean and std.")
            self.distribution_name = probability_distribution[0]
            self.distribution_parameters = probability_distribution[1]

    def sample(self, size=1, uniform=False) -> torch.Tensor:
        """
        Sample from the prior or uniform distribution for this parameter.
        Input:
            - size: the number of sampled random points
            - uniform: sample uniformly
        Returns:
            - the samples
        """
        if self.distribution_name == "uniform" or uniform:
            samples = self.min_value + torch.rand(size) * (self.max_value - self.min_value)
        elif self.distribution_name == "custom_distribution":
            x_probability = torch.tensor(np.random.uniform(0, 1, size))
            samples = torch.tensor(np.interp(x_probability, self.cdf, [self.min_value + x * (self.max_value - self.min_value) / (len(self.cdf) - 1) for x in range(len(self.cdf))]))
        elif self.distribution_name == "gaussian":
            # truncnorm expects limits in number of stds
            a, b = (self.min_value - self.distribution_parameters[0]) / self.distribution_parameters[1], (self.max_value - self.distribution_parameters[0]) / self.distribution_parameters[1]
            samples = torch.tensor(
                truncnorm.rvs(
                    a=a,
                    b=b,
                    loc=self.distribution_parameters[0],
                    scale=self.distribution_parameters[1],
                    size=size,
                )
            )
        else:
            raise Exception("Invalid distribution:", self.distribution_name)
        return samples

    def pdf(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the probability density of a given x under the prior distribution of the parameter.
        Returns:
        - the probability of x
        """
        x = x.view(-1)
        if self.distribution_name == "custom_distribution":
            return np.interp(x, [self.min_value + x * (self.max_value - self.min_value) / (len(self.probability_distribution) - 1) for x in range(len(self.probability_distribution))], self.probability_distribution)
        elif self.distribution_name == "gaussian":
            mean = self.distribution_parameters[0]
            std = self.distribution_parameters[1]
            dist = torch.distributions.Normal(mean, std)
            return torch.exp(dist.log_prob(x)) / (dist.cdf(torch.tensor(self.max_value)) - dist.cdf(torch.tensor(self.min_value)))
        elif self.distribution_name == "uniform":
            return torch.ones_like(x) / (self.max_value - self.min_value)

    def get_size(self) -> float:
        return float("inf")

    def get_discrete_size(self) -> int:
        return len(self.preferred_discretization)

    def get_discrete_values(self) -> int:
        return self.preferred_discretization

    def get_min(self) -> float:
        return self.min_value

    def get_max(self) -> float:
        return self.max_value

    def convert(
            self,
            input_value: Union[str, float],
            from_type: str,
            to_type: str,
    ):

        """
        converts a single value between formats

        Inputs:
            - data: a single value
            - from_type: the format of the input ("string", "internal", "original", "01")
            - to_type: the format of the output ("string", "internal", "original", "01")
        Returns:
            - the converted value
        """
        if from_type == "string":
            intermediate_value = float(input_value)
        elif from_type == "01":
            intermediate_value = input_value * (self.get_max() - self.get_min()) + self.get_min()
        else:
            intermediate_value = input_value

        if to_type == "string":
            return f"{intermediate_value}"
        elif to_type == "01":
            return (intermediate_value - self.get_min()) / (self.get_max() - self.get_min())
        else:
            return intermediate_value


class IntegerParameter(Parameter):
    """
    This class defines an Integer parameter, i.e. an interval of integers from a to b.
    """

    def __init__(
            self,
            name: str,
            min_value: int,
            max_value: int,
            default: int,
            probability_distribution: Union[str, List[float]],
            constraints: Optional[List[str]] = None,
            dependencies: Optional[List[str]] = None,
            transform: Optional[str] = "none",
    ):
        """
        Initialization method. The possible values for this parameter are between min_value and max_value.

        Input:
             - name: name of the variable
             - min_value: minimum value.
             - max_value: maximum value.
             - default: default value.
             - probability_distribution: a string describing the probability density function.
             - constraints: list of constraints as evaluable strings
             - dependencies: list of strings encoding dependencies
        """
        if dependencies is None:
            dependencies = []
        if constraints is None:
            constraints = []
        Parameter.__init__(self, name, default, constraints, dependencies)
        self.min_value = min_value
        self.max_value = max_value
        self.values = torch.arange(min_value, max_value + 1)
        self.val_indices = dict(zip(self.values.to(dtype=torch.long).numpy(), list(range(len(self.values)))))
        self.transform = transform

        if isinstance(probability_distribution, str):
            self.distribution_name = probability_distribution
            if self.distribution_name == "uniform":
                self.distribution = torch.ones(len(self.values)) / len(self.values)
        else:
            self.distribution_name = "custom_distribution"
            self.distribution = torch.tensor(probability_distribution) / np.sum(probability_distribution)
        assert self.distribution_name in (["uniform", "custom_distribution"]), f"invalid distribution {self.distribution_name} for IntegerParameter."

    def sample(self, size=1, uniform=False) -> torch.Tensor:
        """
        Sample from the specific beta distribution defined in the json for this parameter.

        Input:
            - size: the number of sampled random points
        Returns:
            - the random sampled values from the set of available values.
        """
        if self.distribution_name == "uniform" or uniform:
            samples = np.random.choice(self.values, size=size)
        elif self.distribution_name == "custom_distribution":
            samples = np.random.choice(self.values, size=size, p=self.distribution)
        else:
            raise Exception("Invalid distribution", self.distribution_name)
        samples = torch.tensor(samples)
        return samples

    def pdf(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the probability density of a given x under the prior distribution of the parameter.
        Returns:
            - the probability of x
        """
        return self.distribution[x.to(dtype=torch.long) - self.min_value]

    def get_size(self) -> int:
        return self.max_value - self.min_value + 1

    def get_discrete_size(self) -> int:
        return self.get_size()

    def get_values(self) -> List[int]:
        return list(range(self.min_value, self.max_value + 1))

    def get_min(self) -> int:
        return self.min_value

    def get_max(self) -> int:
        return self.max_value

    def convert(
            self,
            input_value: Union[str, float],
            from_type: str,
            to_type: str,
    ):
        """
        converts a single value between formats

        Inputs:
            - data: a single value
            - from_type: the format of the input ("string", "internal", "original", "01")
            - to_type: the format of the output ("string", "internal", "original", "01")
        Returns:
            - the converted value
        """
        if from_type in ["string", "internal"]:
            intermediate_value = int(input_value)
        elif from_type == "01":
            intermediate_value = int(np.floor(input_value * (self.get_max() + 0.999999 - self.get_min()))) + self.get_min()
        else:
            intermediate_value = input_value

        if to_type == "string":
            return f"{intermediate_value}"
        elif to_type == "01":
            return (intermediate_value - self.get_min()) / (self.get_max() - self.get_min())
        else:
            return intermediate_value


class OrdinalParameter(Parameter):
    """
    This class defines an ordinal parameter, i.e. parameters that are numerical and can be ordered using lesser, equal, greater than.
    """

    def __init__(
            self,
            name: str,
            values: List[float],
            default: float,
            probability_distribution: Union[str, List[float]],
            constraints: List[str] = None,
            dependencies: List[str] = None,
            transform: Optional[str] = "none",
    ):
        """
        Initialization method. The possible values for this parameter are defined by the list values.

        Input:
             - name: variable name
             - values: list of possible value for this parameter.
             - default: default value.
             - probability_distribution: a string describing the probability density function or a list of values describing the probability distribution.
             - constraints: list of constraints as evaluable strings
             - dependencies: list of strings encoding dependencies
        """
        if dependencies is None:
            dependencies = []
        if constraints is None:
            constraints = []
        Parameter.__init__(self, name, default, constraints, dependencies)
        self.values = torch.tensor(sorted(values, key=float))  # ascending order
        self.int_ordinal = all([v % 1 == 0 for v in self.values])
        self.val_indices = dict(zip(self.values.to(dtype=torch.long).numpy(), list(range(len(self.values)))))
        self.distribution = []
        self.transform = transform

        if isinstance(probability_distribution, str):
            self.distribution_name = probability_distribution
            self.distribution = torch.ones(len(self.values)) / len(self.values)
        else:
            self.distribution_name = "custom_distribution"
            self.distribution = torch.tensor(probability_distribution)
        assert self.distribution_name in (["uniform", "custom_distribution"]), f"invalid distribution {self.distribution_name} for OrdinalParameter."

    def sample(self, size=1, uniform=False) -> torch.Tensor:
        """
        Sample from the specific beta distribution defined in the json for this parameter.

        Input:
             - size: the number of sampled random points
        Returns:
        - the random sampled values from the set of available values.
        """

        if self.distribution_name == "uniform" or uniform:
            samples = np.random.choice(self.values, size=size)
        elif self.distribution_name == "custom_distribution":
            samples = np.random.choice(self.values, size=size, p=self.distribution)
        else:
            raise Exception("Invalid distribution", self.distribution_name)
        samples = torch.tensor(samples)
        return samples

    def pdf(self, x: Any) -> Union[torch.Tensor]:
        """
        Compute the probability of a given X under the prior distribution of the parameter.
        Returns:
        - the probability of X
        """
        return torch.tensor([self.distribution[self.val_indices[i.item()]] for i in x])

    def get_size(self) -> int:
        return len(self.values)

    def get_discrete_size(self) -> int:
        return self.get_size()

    def get_discrete_values(self) -> List[Any]:
        return self.values

    def get_values(self) -> List[Any]:
        return self.values

    def get_min(self) -> Any:
        if self.get_size() > 0:
            return self.values[0]
        else:
            print("Error: this ordinal parameter doesn't have values. Exit.")
            exit()

    def get_max(self) -> Any:
        if self.get_size() > 0:
            return self.values[-1]
        else:
            print("Error: this ordinal parameter doesn't have values. Exit.")
            exit()

    def convert(
            self,
            input_value: Union[str, float],
            from_type: str,
            to_type: str,
    ):
        """
        converts a single value between formats

        Inputs:
            - data: a single value
            - from_type: the format of the input ("string", "internal", "original", "01")
            - to_type: the format of the output ("string", "internal", "original", "01")
        Returns:
            - the converted value
        """
        if from_type == "string":
            intermediate_value = float(input_value)
        elif from_type == "01":
            intermediate_value = self.values[int(np.floor(input_value * self.get_size() * 0.999999))]
        else:
            intermediate_value = input_value

        if to_type == "string":
            if self.int_ordinal:
                return f"{int(intermediate_value)}"
            else:
                return f"{intermediate_value}"
        elif to_type == "01":
            return self.values.index(intermediate_value) / (self.get_size() - 1)
        else:
            return intermediate_value


class CategoricalParameter(Parameter):
    """
    This class defines a categorical parameter, i.e. parameters like strings and booleans (all encoded as strings, but represented by their indices in the code),
    where the elements cannot be ordered using lesser, equal, greater than
    (or at least it doesn't make sense ordering them like it doesn't make sense to order "true" and "false").

    Warning: Categorical parameters are treated as a sort of Ordinal parameters, this may not work in general.
    """

    def __init__(
            self,
            name: str,
            values: List[str],
            default: str,
            probability_distribution: Union[str, List[float]],
            constraints: Optional[List[str]] = None,
            dependencies: Optional[List[str]] = None,
    ):
        """
        Initialization method. The possible values for this parameter are defined by the list values.

        Input:
             - name: variable name
             - values: list of possible value for this parameter.
             - default: default value.
             - probability_distribution: a string describing the probability density function or a list of values describing the probability distribution.
             - constraints: list of constraints as evaluable strings
             - dependencies: list of strings encoding dependencies
        """

        if dependencies is None:
            dependencies = []
        if constraints is None:
            constraints = []

        print(f"values: {values}")
        print(f"default: {default}")
        Parameter.__init__(self, name, values.index(default), constraints, dependencies)
        self.values = torch.arange(len(values))
        self.string_values = values
        self.val_indices = {i: i for i in self.values}

        if isinstance(probability_distribution, str):
            self.distribution_name = probability_distribution
            if self.distribution_name == "uniform":
                self.distribution = torch.ones(len(self.values)) / len(self.values)
        else:
            self.distribution_name = "custom_distribution"
            self.distribution = torch.tensor(probability_distribution)
        assert self.distribution_name in [
            "uniform",
            "custom_distribution",
        ], "only 'uniform' and 'custom_distribution' are allowed distributions for CategoricalParameter."

    def sample(self, size=1, uniform=False) -> torch.Tensor:
        """
        Select at random following the distribution given in the json.
        Returns:
        - a random number.
             - size: the number of sampled random points
        """
        if self.distribution_name == "uniform" or uniform:
            samples = np.random.choice(self.values, size=size)
        else:
            samples = np.random.choice(self.get_size(), size=size, p=self.distribution)
        samples = torch.tensor(samples)
        return samples

    def pdf(self, x_idx: torch.tensor) -> float:
        """
        Compute the probability of a given X under the prior distribution of the parameter.
        Returns:
        - the probability of X
        """
        return self.distribution[x_idx.to(dtype=torch.long)]

    def get_default(self) -> int:
        if isinstance(self.default, str):
            return self.values.index(self.default)
        else:
            return self.default

    def get_size(self) -> int:
        return len(self.values)

    def get_discrete_size(self) -> int:
        return self.get_size()

    def get_discrete_values(self) -> List[int]:
        return self.get_values()

    def get_values(self) -> List[int]:
        return self.values

    def get_string_values(self) -> List[str]:
        return self.string_values

    def get_int_value(self, str_value: str) -> int:
        return self.string_values.index(str_value)

    def get_original_string(self, idx_value: int) -> str:
        return str(self.string_values[int(idx_value)].encode())

    def convert(
            self,
            input_value: Union[str, float],
            from_type: str,
            to_type: str,
    ):
        """
        converts a single value between formats

        Inputs:
            - data: a single value
            - from_type: the format of the input ("string", "internal", "original", "01")
            - to_type: the format of the output ("string", "internal", "original", "01")
        Returns:
            - the converted value
        """
        if from_type in ["string", "original"]:
            intermediate_value = self.values.index(input_value)
        elif from_type == "01":
            intermediate_value = int(np.floor(input_value * self.get_size() * 0.999999))
        else:
            intermediate_value = int(input_value)

        if to_type in ["string", "original"]:
            return self.values[intermediate_value]
        elif to_type == "01":
            return intermediate_value / (self.get_size() - 1)
        else:
            return intermediate_value


class PermutationParameter(Parameter):
    """
    This class represents permutation variables in different parametrizations.

    When using GPs, each permutation variable is parametrized by a number of other variables.

    For ease of notation, in the following, let n be the number of items in the permutation and
    let sigma(i) be the rank of item i, such that if sigma(i) = i for all i, then it is the
    identity permutation.

    Parametrizations:
        - Spearman:
            The permutation is represented by n variables x_i = sigma(i), i.e., x_i represents
            the position of item i in the permutation. This will yield an exponential kernel with
            Spearman distance.

        - Kendall:
            The permutation is represented by n(n-1)/2 variables x_ij such that x_ij (i < j)
            is equal to 1 iff sigma(i) < sigma(j). This will yield an exponential kernel with
            the Kendall distance, also called the Mallows kernel.

        - Hamming:
            The permutation is represented by n categorical variables x_i = sigma(i). Hence,
            on paper same as the Spearman representation but using categorical variables will
            instead yield the exponential kernel with Hamming distance.

    """

    def __init__(
            self,
            name: str,
            n_elements: int,
            default: List[int],
            parametrization: str,
            constraints: Optional[List[str]] = None,
            dependencies: Optional[List[str]] = None,
    ):
        """
        Initialization method. The possible values for this parameter are defined by the list values.
             - name: variable name
             - n_elements: number of elements in the permutation
             - default: default value.
             - parametrization: how to internally parametrize the permutation. Also defines which kernel is used.
             - constraints: list of constraints as evaluable strings
             - dependencies: list of strings encoding dependencies

        Note that permutations are tuples
        """
        if dependencies is None:
            dependencies = []
        if constraints is None:
            constraints = []
        self.n_elements = n_elements
        self.permutation_values: List[tuple] = [p for p in permutations([x for x in range(self.n_elements)])]
        self.string_values = [f"{tuple(p)}" for p in self.permutation_values]
        self.values = torch.arange(len(self.permutation_values))
        self.default_index = None
        if default:
            self.default_index = self.permutation_values.index(tuple(default))
        Parameter.__init__(self, name, self.default_index, constraints, dependencies)

        self.parametrization = parametrization.lower()
        self.distribution = torch.ones(len(self.values)) / len(self.values)
        self.val_indices = {i.item(): i for i in self.values}  # from internal to index (which are the same for permutations)

        print(f"values: {self.values}")
        print(f"permutation_values: {self.permutation_values}")
        print(f"string_values: {self.string_values}")
        print(f"n_elemens: {self.string_values}")
        print(f"parametrization: {self.parametrization}")
        print(f"distribution: {self.distribution}")

    def parametrize(self, data: List[int]) -> Tuple[List[str], List[List[float]]]:

        """
        Provides a parametrization representation of the variable.

        Input:
             - data: the values of the parameter
        :returns: a list of names for the new variables
                  a list of values. The outer list is over the new variables, and the inner list over the data points.
        """

        if self.parametrization == "spearman":
            return (
                [f"{self.name}_{i}" for i in range(self.n_elements)],
                [
                    [
                        self.permutation_values[int(d)].index(i) / self.n_elements
                        for i in range(self.n_elements)
                    ]
                    for d in data
                ],
            )

        elif self.parametrization == "kendall":
            return (
                [
                    f"{self.name}_{i}_{j}"
                    for i in range(self.n_elements)
                    for j in range(i + 1, self.n_elements)
                ],
                [
                    [
                        self.permutation_values[int(d)][i]
                        < self.permutation_values[int(d)][j]
                        for i in range(self.n_elements)
                        for j in range(i + 1, self.n_elements)
                    ]
                    for d in data
                ],
            )

        elif self.parametrization == "hamming":
            return (
                [
                    f"{self.name}_{i}_{j}"
                    for i in range(self.n_elements)
                    for j in range(self.n_elements)
                ],
                [
                    [self.permutation_values[int(d)][i] == j
                     for i in range(self.n_elements)
                     for j in range(self.n_elements)
                     ]
                    for d in data
                ],
            )

        elif self.parametrization == "naive":
            return (
                [f"{self.name}_{i}" for i in self.values],
                [[int(int(d) == i) for i in self.values] for d in data],
            )

        else:
            raise Exception(
                f"Incorrect permutation parametrization: {self.parametrization}"
            )

    def sample(self, size=1, uniform=False) -> torch.Tensor:
        """
        Select at random following the distribution given in the json.

        Input:
             - size: the number of sampled random points
        Returns:
            - a random number.
        """
        samples = np.random.choice(self.values, size=size)
        samples = torch.tensor(samples)
        return samples

    def pdf(self, x_idx: torch.Tensor) -> float:
        """
        Compute the probability of a given X under the prior distribution of the parameter.
        Returns:
        - the probability of X
        """
        return self.distribution[x_idx.to(dtype=torch.long)]

    def get_default(self) -> int:
        return self.default_index

    def get_size(self) -> int:
        return len(self.values)

    def get_discrete_size(self) -> int:
        return self.get_size()

    def get_discrete_values(self) -> List[int]:
        return self.get_values()

    def get_values(self) -> List[int]:
        return self.values

    def get_permutation_values(self) -> List[Tuple[int]]:
        return self.permutation_values

    def get_int_value(self, permutation: Tuple[int]) -> int:
        return self.permutation_values.index(permutation)

    def get_permutation_value(self, idx_value: int) -> Tuple[int]:
        return self.permutation_values[int(idx_value)]

    def string_to_int(self, string: str) -> int:
        if string[0] == "(":
            return self.get_int_value(tuple(int(x) for x in string[1:-1].split(",")))
        else:
            return self.get_int_value(tuple(int(x) for x in string.split(",")))

    def int_to_string(self, idx_value: int) -> str:
        return f"{tuple(self.get_permutation_value(idx_value))}"

    def convert(
            self,
            input_value: Union[str, float, tuple],
            from_type: str,
            to_type: str,
    ):
        """
        converts a single value between formats

        Inputs:
            - data: a single value
            - from_type: the format of the input ("string", "internal", "original", "01")
            - to_type: the format of the output ("string", "internal", "original", "01")
        Returns:
            - the converted value
        """
        if from_type == "string":
            intermediate_value = self.string_values.index(input_value)
        elif from_type == "original":
            intermediate_value = self.permutation_values.index(input_value)
        elif from_type == "01":
            intermediate_value = int(np.floor(input_value * self.get_size() * 0.999999))
        else:
            intermediate_value = int(input_value)

        if to_type == "string":
            return self.string_values[intermediate_value]
        if to_type == "original":
            return self.permutation_values[intermediate_value]
        elif to_type == "01":
            return intermediate_value / (self.get_size() - 1)
        else:
            return intermediate_value


class VirtualList(list):
    def __init__(self, length: int, function: callable):
        super().__init__()
        self.length = length
        self.function = function

    def __getitem__(self, key):
        if isinstance(key, slice):
            return [function(x) for x in key]
        else:
            return function(key)

    def __len__(self):
        return self.length

# todo adjust description
class SelectionParameter(Parameter):
    """
    This class represents selection variables in different parametrizations.

    When using GPs, each permutation variable is parametrized by a number of other variables.

    For ease of notation, in the following, let n be the number of items in the permutation and
    let sigma(i) be the rank of item i, such that if sigma(i) = i for all i, then it is the
    identity permutation.

    Parametrizations:
        - Spearman:
            The permutation is represented by n variables x_i = sigma(i), i.e., x_i represents
            the position of item i in the permutation. This will yield an exponential kernel with
            Spearman distance.

        - Kendall:
            The permutation is represented by n(n-1)/2 variables x_ij such that x_ij (i < j)
            is equal to 1 iff sigma(i) < sigma(j). This will yield an exponential kernel with
            the Kendall distance, also called the Mallows kernel.

        - Hamming:
            The permutation is represented by n categorical variables x_i = sigma(i). Hence,
            on paper same as the Spearman representation but using categorical variables will
            instead yield the exponential kernel with Hamming distance.

    """

    def __init__(
            self,
            name: str,
            n_elements: int,
            values: List[str],
            default: List[int],
            parametrization: str,
            length: Tuple[int, int],
            constraints: Optional[List[str]] = None,
            dependencies: Optional[List[str]] = None,
    ):
        """
        Initialization method. The possible values for this parameter are defined by the list values.
             - name: variable name
             - n_elements: number of elements in the permutation
             - default: default value.
             - parametrization: how to internally parametrize the permutation. Also defines which kernel is used.
             - constraints: list of constraints as evaluable strings
             - dependencies: list of strings encoding dependencies

        Note that permutations are tuples
        """

        '''
        if dependencies is None:
            dependencies = []
        if constraints is None:
            constraints = []
        '''
        constraints = None
        dependencies = None
        self.n_elements = n_elements

        self.name = name
        self.length = length
        self.symbols = values

        vars = []
        for i in range(length[1]):
            vars.append(CategoricalParameter(name + "_" + str(i), self.symbols, self.symbols[0], "uniform"))
        self.vars = vars

        # print("\n")
        # print("Selection parametr init\n")

        # change this to other values 

        # todo: use length [min, max]

        # all permutations including partial sums (1 - 6)
        # select from list of values

        # self.permutation_values: List[tuple] = [p for p in permutations([x for x in range(self.n_elements)])]

        # for x in range(self.length)
        '''
        self.selection_values: List[tuple] = []
        '''
        # print(f"length: {self.length}")


        # groups of selection  
        # A: 0 1 2 3
        # B: 0 1 2 3
        # C: 0 1 2 3

        # how to combine? 
        # combine all and filter duplicates? 

        # length 1
        # A: 0 1
        # B: 0 1
        # C: 0 1
        # A, B, C

        # length 2
        # A: 0 1 2
        # B: 0 1 2
        # C: 0 1 2
        # permutations: 
        # A: 2 -> AA
        # A: 1 -> AB, AC, BA, CA
        # A: 0 -> BC, CB
        # B: 2 -> BB
        # C: 2 -> CC

        # length 3
        # A: 0 1 2 3
        # B: 0 1 2 3
        # C: 0 1 2 3
        # permutations:  
        # A: 3 -> AAA
        # A: 2 -> AAB, AAC, ABA, ACA, BAA, CAA
        # A: 1 -> ABC, ACB, BAC, BCA, CAB, CBA
        # A: 0 -> BBC, BCB, CBB, CBC, CCB, BBB, CCC

        # A: List[str] = ['A', 'A', 'A']
        # B: List[str] = ['B', 'B', 'B']
        # C: List[str] = ['C', 'C', 'C']
        # perms = permute(A, B, C)

        # A: List[str] = ['A', 'A']
        # B: List[str] = ['B']
        # C: List[str] = ['C']


        # import itertools

        # vals: List[str] = ['A', 'B', 'C']
        # length = 4

        # permutations = []
        # for perm in itertools.product(my_list, repeat=4):
        #     permutations.append(perm)

        # print(permutations)

        # print("Test")


        # vals: List[int] = [1, 2]
 
        # selections = []
        # for sel in permutations(vals, length[1]):
        #     selections.append(sel)

        # print("selections: ")
        # [print(x) for x in selections]


        # selections = []
        
        # print("build selecton values")

        self.values = VirtualList(self.get_size(), lambda i: self.int_to_original(i))
        self.val_indices = VirtualList(self.get_size(), lambda i: torch.tensor([i]))
        
        self.selection_values = []
        for x in range(length[0], length[1] + 1):
            self.selection_values.extend(product(self.symbols, repeat=x))
        
        print(f"Finished: {len(self.selection_values)} elements")

        '''
        print("selection_values: ")
        for value in self.selection_values:
            print(f"type: {type(value)}")
            print(value[0])
            print(f"tuple length: {len(value)}")
            print(f"split: {str(value).split(',')}")
            print(f"len: {len(str(value).split(','))}")
        '''
        
        self.default_index = None
        Parameter.__init__(self, name, self.default_index, constraints, dependencies)

        '''
        self.selection_values = [x[0] if len(str(x).split(',')) == 1 else x for x in self.selection_values]

        # todo add selection part
        # we want to allow permutations including AAA, BBB, CCB

        self.string_values = [f"{s}" for s in self.selection_values]
        # print("torch.arange")
        self.values = torch.arange(len(self.selection_values))
        self.default_index = None
        if default:
            self.default_index = self.selection_values.index(tuple(default))
        Parameter.__init__(self, name, self.default_index, constraints, dependencies)

        self.parametrization = parametrization.lower()
        self.distribution = torch.ones(len(self.values)) / len(self.values)
        self.val_indices = {i.item(): i for i in self.values}  # from internal to index (which are the same for permutations)
        '''


    def parametrize(self, data: List[int]) -> Tuple[List[str], List[List[float]]]:

        """
        Provides a parametrization representation of the variable.

        Input:
             - data: the values of the parameter
        :returns: a list of names for the new variables
                  a list of values. The outer list is over the new variables, and the inner list over the data points.
        """

        print("parameterizei?")

        if self.parametrization == "spearman":
            return (
                [f"{self.name}_{i}" for i in range(self.n_elements)],
                [
                    [
                        self.selection_values[int(d)].index(i) / self.n_elements
                        for i in range(self.n_elements)
                    ]
                    for d in data
                ],
            )

        elif self.parametrization == "kendall":
            return (
                [
                    f"{self.name}_{i}_{j}"
                    for i in range(self.n_elements)
                    for j in range(i + 1, self.n_elements)
                ],
                [
                    [
                        self.selection_values[int(d)][i]
                        < self.selection_values[int(d)][j]
                        for i in range(self.n_elements)
                        for j in range(i + 1, self.n_elements)
                    ]
                    for d in data
                ],
            )

        elif self.parametrization == "hamming":
            return (
                [
                    f"{self.name}_{i}_{j}"
                    for i in range(self.n_elements)
                    for j in range(self.n_elements)
                ],
                [
                    [self.selection_values[int(d)][i] == j
                     for i in range(self.n_elements)
                     for j in range(self.n_elements)
                     ]
                    for d in data
                ],
            )

        elif self.parametrization == "naive":
            return (
                [f"{self.name}_{i}" for i in self.values],
                [[int(int(d) == i) for i in self.values] for d in data],
            )

        else:
            raise Exception(
                f"Incorrect permutation parametrization: {self.parametrization}"
            )

    def sample(self, size=1, uniform=False) -> torch.Tensor:
        """
        Select at random following the distribution given in the json.

        Input:
             - size: the number of sampled random points
        Returns:
            - a random number.
        """

        '''
        samples = np.random.choice(self.values, size=size)
        samples = torch.tensor(samples)
        return samples
        '''
        samples = torch.stack([v.sample(size, uniform) for v in self.vars])
        tensor = torch.transpose(samples, 0, 1)
        return torch.tensor([self.original_to_int(x) for x in tensor])

    def pdf(self, x_idx: torch.Tensor) -> float:
        """
        Compute the probability of a given X under the prior distribution of the parameter.
        Returns:
        - the probability of X
        """
        '''
        return self.distribution[x_idx.to(dtype=torch.long)]
        '''
        return 1 / self.get_size()

    '''
     def get_size(self) -> int:
        return len(self.values)
    '''
    def get_size(self) -> int:
        # return sum([len(self.symbols) ** n for n in range(self.length[0], self.length[1]+1)])
        return len(self.symbols) ** self.length[1]
    
    '''
    def get_default(self) -> int:
        return self.default_index

    def get_discrete_size(self) -> int:
        return self.get_size()

    def get_discrete_values(self) -> List[int]:
        return self.get_values()

    def get_values(self) -> List[int]:
        return self.values

    def get_permutation_values(self) -> List[Tuple[int]]:
        return self.selection_values

    def get_int_value(self, selection: Tuple[int]) -> int:
        return self.selection_values.index(selection)

    def get_permutation_value(self, idx_value: int) -> Tuple[int]:
        return self.selection_values[int(idx_value)]

    def string_to_int(self, string: str) -> int:
        if string[0] == "(":
            return self.get_int_value(tuple(int(x) for x in string[1:-1].split(",")))
        else:
            return self.get_int_value(tuple(int(x) for x in string.split(",")))

    def int_to_string(self, idx_value: int) -> str:
        return f"{tuple(self.get_permutation_value(idx_value))}"

    '''

    def string_to_original(self, input_value: str) -> list[int]:
        symbols = input_value.split(", ")
        return torch.tensor([self.symbols.index(s) for s in symbols])

    def original_to_string(self, input_value: list[int]) -> str:
        return [self.symbols[s] for s in input_value]

    def original_to_int(self, input_value: list[int]) -> int:
        return sum([input_value[i].item() * (len(self.symbols) ** i) for i in range(len(input_value))])

    def int_to_original(self, input_value: int) -> list[int]:
        result = []
        for i in range(len(self.symbols), 0, -1):
            div = len(self.symbols) ** i
            print(str(i) + " " + str(div))
            val = input_value // div
            input_value = input_value % div
            result.append(val)
        return result

    # TODO: adjust this -> create internal representation using ints (enumeration of input strings)
    # (A, B, C) -> (0, 1, 2)
    # create dict to convert? 
    def convert(
            self,
            input_value: Union[str, float, tuple],
            from_type: str,
            to_type: str,
    ):
        """
        converts a single value between formats

        Inputs:
            - data: a single value
            - from_type: the format of the input ("string", "internal", "original", "01")
            - to_type: the format of the output ("string", "internal", "original", "01")
        Returns:
            - the converted value
        """
        print("convert " + str(input_value) + " " + from_type + " " + to_type)

        if from_type == "string":
            original = self.string_to_original(input_value)
        elif from_type == "original":
            original = input_value
        elif from_type == "01":
            original = self.int_to_original(int(round(input_value * (self.get_size()-1))))
        else:
            original = self.int_to_original(int(round(input_value)))
        
        if to_type == "string":
            return self.original_to_string(original)
        elif to_type == "original":
            return original
        elif to_type == "01":
            return self.original_to_int(original) / (self.get_size()-1)
        else:
            return self.original_to_int(original)

        '''
        if from_type == "string":
            intermediate_value = self.string_values.index(input_value)
        elif from_type == "original":
            intermediate_value = self.selection_values.index(input_value)
        elif from_type == "01":
            intermediate_value = int(np.floor(input_value * self.get_size() * 0.999999))
        else:
            intermediate_value = int(input_value)

        if to_type == "string":
            return self.string_values[intermediate_value]
        if to_type == "original":
            return self.selection_values[intermediate_value]
        elif to_type == "01":
            return intermediate_value / (self.get_size() - 1)
        else:
            return intermediate_value
        '''

