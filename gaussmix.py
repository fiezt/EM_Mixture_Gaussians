__author__ = 'tfiez'

from collections import defaultdict
from collections import Counter
import numpy as np
import random
from datetime import datetime
from scipy.misc import logsumexp
import csv
import warnings
import sys
warnings.filterwarnings('ignore')


def load_data(args):
    """Load input data and get the number of components, examples, and features.

    :param args: Command line arguments.
    :return: The data as an array, the number of components, the number of
    examples, and the number of features.
    """

    # Get the number of componets.
    num_components = int(args[1])

    # Open and read the data file.
    with open(args[2]) as f:

        curr_file = []

        for line in f:

            # Strip the line of commas and newlines and convert to floats.
            curr_line = line.rsplit()
            curr_line = [float(val) for val in curr_line]

            # Get the number of examples and features from the first line.
            if len(curr_line) == 2:
                num_examples = int(curr_line[0])
                num_features = int(curr_line[1])
            else:
                curr_file.append(curr_line)

    data = np.vstack(tuple(curr_file))

    return data, num_components, num_examples, num_features


def write_output(args, components, num_components, num_features):
    """Writing the component priors, means, and variances to an output file.

    :param args: Command line arguments.
    :param components: Components which include the prior, mean, and variance.
    :param num_components: Number of components being used.
    :param num_features: Number of features in the data.
    :return: Writes to a file the results.
    """

    # Get the name of the output file and write the data to it.
    with open(args[3], 'w') as output:
        file_writer = csv.writer(output, delimiter=',')

        # Writing the number of components and features first.
        file_writer.writerow([num_components, num_features])

        for i in range(num_components):

            # Writing in order the prior, mean, var, for each feature.
            curr_line = [components[i]['prior']] + \
                        components[i]['mean'].flatten().tolist() + \
                        np.diag(components[i]['var']).flatten().tolist()

            file_writer.writerow(curr_line)


def get_likelihood(components, example, num_components):
    """Calculate the likelihood of the multivariate gaussian. P(x_i|C_j).

    :param components: Components which include the prior, mean, and variance.
    :param example: A data point.
    :param num_components: The number of components.
    :return: The log of the likelihoods for each of the components.
    """

    # Get the number of features.
    num_features = example.shape[0]

    # List to hold the likelihoods for each of the components.
    likelihoods = list()

    # Getting the log of the likelihood for each of the components.
    for i in range(num_components):

        # This is the log of the exponential piece of the Multivariate Gaussian.
        exp_comp = -.5 * np.transpose(example - components[i]['mean']).\
                            dot(np.linalg.pinv(components[i]['var'])).\
                            dot(example - components[i]['mean'])

        # This is the log of the pi component of the Multivariate Gaussian.
        pi_comp = (num_features/2.) * np.log(2*np.pi)

        """
        This is the determinant piece of the Multivariate Gaussian.
        This a protection for when there is just one example for a component
        so the variance is 0, or less than the float min for python
        so I add a protection to not take log of 0.
        """
        if np.linalg.det(components[i]['var']) == 0.0:
            det_comp = sys.float_info.min
        else:
            det_comp = (1 / 2.) * np.log(np.linalg.det(components[i]['var']))
        
        likelihoods.append(float(exp_comp) - pi_comp - det_comp)

    return likelihoods


def get_posteriors(components, example, num_components):
    """Calculate the posterior probabilities for each component. P(C|x).

    :param components: Components which include the prior, mean, and variance.
    :param example: A data point.
    :param num_components: The number of components.
    :return: The posterior probabilities for a data point.
    """

    likelihoods = get_likelihood(components, example, num_components)

    # Calculating the log of P(C)P(x|C) for each component.
    probs = [np.log(components[i]['prior']) + likelihoods[i] for i in range(num_components)]

    # Getting the sum of P(C)P(x|C) over the components using log-sum-exp.
    alpha = logsumexp(probs)

    # Calculating the log of the posterior probabilities.
    posteriors = [probs[i]-alpha for i in range(num_components)]

    # Converting back to probability from log probability.
    sum_posteriors = [np.exp(posterior) for posterior in posteriors]

    return sum_posteriors


def log_likelihood(components, data, num_components, num_examples, num_features):
    """Calculate the complete log likelihood.

    :param components: Components which include the prior, mean, and variance.
    :param data: All the training data provided.
    :param num_components: The number of components for the model.
    :param num_examples: The number of data points in the data.
    :param num_features: The number of features in the data.
    :return: The complete log likelihood sum.
    """

    log_like = 0.0
    for i in range(num_examples):

        # Get the likelihood for one data point for each of the components.
        # This is getting log(P(C)) + log(P(x|C)) for each component.
        probs = [np.log(components[j]['prior']) +
                 get_likelihood(components, np.reshape(data[i], (num_features, 1)),
                num_components)[j] for j in range(num_components)]

        log_like += logsumexp(probs)

    return log_like


def em_mixture(data, num_components, num_examples, num_features):
    """EM mixture algorithm for clustering.

    :param data: All the training data provided.
    :param num_components: The number of components for the model.
    :param num_examples: The number of data points in the data.
    :param num_features: The number of features in the data.
    :return: The components which include the priors, mean, and variance.
    """

    components = dict()

    posteriors = np.zeros((num_examples, num_components))

    log_like_new = float('-inf')

    random.seed(datetime.now())

    """
    Priors are initialized to a uniform distribution, the means are
    initialized to random data points, and variances are initialized to
    a fraction of the range of each variable.
    """
    mean_inits = [random.randint(0, num_examples-1) for i in range(num_components)]
    for i in range(num_components):
        components[i] = {'prior': 1/float(num_components),
                         'mean': np.reshape(data[mean_inits[i]], (num_features, 1)),
                         'var': np.diag(np.ptp(data, axis=0)/(i + 1))}

    for iterations in range(100):
        # E Step getting the posterior probabilities (the unobserved variables).
        for i in range(num_examples):
            row = np.reshape(data[i], (num_features, 1))
            posteriors[i] = get_posteriors(components, row, num_components)

        # M step getting the new parameter values to maximize
        # the probability of the data, both observed and estimated.
        for i in range(num_components):
            components[i]['prior'] = np.mean(posteriors[:,i])

            components[i]['mean'] = sum(data[j] * posteriors[j,i] for j in range(num_examples))\
                                    /np.sum(posteriors[:,i])

            components[i]['var'] = np.diag(sum((data[j] - components[i]['mean'])**2
                                    * posteriors[j, i] for j in range(num_examples))/np.sum(posteriors[:,i]))

            components[i]['mean'] = np.reshape(components[i]['mean'], (num_features, 1))

        log_like_old = log_like_new
        log_like_new = log_likelihood(components, data, num_components, num_examples, num_features)

        # Convergence condition for EM, log like increasing by less than 0.1%.
        if (abs(log_like_new - log_like_old)/abs(log_like_old) * 100) < 0.1:
            return components

    return components


def main():
    """Read in data and perform EM on data, before writing component information."""

    args = sys.argv

    data, num_components, num_examples, num_features = load_data(args)

    components = em_mixture(data, num_components, num_examples, num_features)

    write_output(args, components, num_components, num_features)


if __name__ == '__main__':
    main()


