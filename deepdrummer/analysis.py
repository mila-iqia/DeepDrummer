import pickle

import numpy as np
import numpy.testing as npt

import scipy
import scipy.optimize
import scipy.stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pylab

"""
{"params": {"batch_size": 8, "dropout": 0.15, "end_time": 1588003140.4521294, "fotf": false, "lr": 0.0005, "num_conv": 64, "num_mlp": 128, "phase": 1, "phase1_count": 100, "save_interval": 50, "start_time": 1588002123.921673, "trials_per_model": 25, "weight_decay": 0}, "user_email": "guillaume.alain.umontreal@gmail.com", "user_name": "Guillaume Alain", "start_time": 1588002123.921673, "end_time": 1588003140.4521294, "phase2_ratings": [[0, "bad"], [0, "bad"], [0, "bad"], [0, "bad"], [0, "bad"], [0, "bad"], [0, "bad"], [0, "bad"], [0, "bad"], [0, "bad"], [0, "bad"], [0, "good"], [0, "good"], [0, "bad"], [0, "good"], [0, "bad"], [0, "bad"], [0, "bad"], [0, "bad"], [0, "bad"], [0, "bad"], [0, "bad"], [0, "bad"], [0, "good"], [0, "good"], [50, "good"], [50, "bad"], [50, "bad"], [50, "bad"], [50, "bad"], [50, "bad"], [50, "bad"], [50, "bad"], [50, "bad"], [50, "good"], [50, "bad"], [50, "bad"], [50, "good"], [50, "bad"], [50, "good"], [50, "good"], [50, "good"], [50, "bad"], [50, "bad"], [50, "bad"], [50, "bad"], [50, "bad"], [50, "bad"], [50, "bad"], [50, "bad"], [100, "good"], [100, "good"], [100, "bad"], [100, "good"], [100, "good"], [100, "good"], [100, "good"], [100, "bad"], [100, "bad"], [100, "bad"], [100, "bad"], [100, "bad"], [100, "bad"], [100, "good"], [100, "bad"], [100, "good"], [100, "good"], [100, "good"], [100, "good"], [100, "good"], [100, "bad"], [100, "good"], [100, "bad"], [100, "bad"], [100, "bad"]]}
"""


"""
First let's expose some methods that will be used by the web interface.
These could be thought as the "porcelain".
"""

def analyze_phase_2(results, histogram_output_path=None, want_N95=True):
    """
    `results` are directly the json structure saved from the web interface.
    We assume that the caller of this function has parsed them for us.
    Here is an example of those `results`:
        {"params": {"batch_size": 8, "dropout": 0.15, "end_time": 1588003140.4521294,
                    "fotf": False, "lr": 0.0005, "num_conv": 64, "num_mlp": 128,
                    "phase": 1, "phase1_count": 100, "save_interval": 50,
                    "start_time": 1588002123.921673, "trials_per_model": 25, "weight_decay": 0},
        "user_email": "guillaume.alain.umontreal@gmail.com", "user_name": "Guillaume Alain",
        "start_time": 1588002123.921673, "end_time": 1588003140.4521294,
        "phase2_ratings": [ [0, "bad"], [0, "bad"], [0, "bad"], [0, "bad"], [0, "bad"], [0, "bad"],
                            [0, "bad"], [0, "bad"], [0, "bad"], [0, "bad"], [0, "bad"], [0, "good"],
                            [0, "good"], [0, "bad"], [0, "good"], [0, "bad"], [0, "bad"], [0, "bad"],
                            [0, "bad"], [0, "bad"], [0, "bad"], [0, "bad"], [0, "bad"], [0, "good"],
                            [0, "good"], [50, "good"], [50, "bad"], [50, "bad"], [50, "bad"],
                            [50, "bad"], [50, "bad"], [50, "bad"], [50, "bad"], [50, "bad"],
                            [50, "good"], [50, "bad"], [50, "bad"], [50, "good"], [50, "bad"],
                            [50, "good"], [50, "good"], [50, "good"], [50, "bad"], [50, "bad"],
                            [50, "bad"], [50, "bad"], [50, "bad"], [50, "bad"], [50, "bad"],
                            [50, "bad"], [100, "good"], [100, "good"], [100, "bad"], [100, "good"],
                            [100, "good"], [100, "good"], [100, "good"], [100, "bad"], [100, "bad"],
                            [100, "bad"], [100, "bad"], [100, "bad"], [100, "bad"], [100, "good"],
                            [100, "bad"], [100, "good"], [100, "good"], [100, "good"], [100, "good"],
                            [100, "good"], [100, "bad"], [100, "good"], [100, "bad"], [100, "bad"],
                            [100, "bad"]]}

    `histogram_output_path` : Path to where you want a histogram to be saved. Optional. Should terminate with '.png'.
    `want_N95` : Determines whether or not we want to compute the returned value
                 for `N_required_for_95pc_confidence`. This takes about 10 seconds
                 to extract when N=100 because it tries values starting from there.
                 We might want to avoid computing this.


    This function results a dict with the following key-values.

    "a_MLE" :   Best estimate for the `a` coefficient in theta(t) = a*t + b.
                    This is the value that represents the most closely what
                    we want to measure. That is, how much improvement happens
                    during phase 2 of this experiment.
    "b_MLE" :   Best estimate for the `b` coefficient in theta(t) = a*t + b.

    "N" :       This is how many measurements were taken.
                It's just a straightforward count of the "phase2_ratings".

    "N_required_for_95pc_confidence" :
                This is a quantity that is somewhat self-referential.
                Let's assume that the true values were indeed the ones
                that you got with (a_MLE, b_MLE).
                If you were to do that experiment properly, and you wanted
                to be 95% confident that your estimate for `a` was
                found in the interval [a_MLE-0.1, infinity],
                then how many evaluations would you have required?

                The point here is that you can compare with your actual value of N,
                and feel suspicious if `N_required_for_95pc_confidence` is much larger.

                If, however, `N_required_for_95pc_confidence` is indeed small,
                it does not guarantee that you're doing well.
                Your data currently leads to you believe that you don't need
                such a large N, but this could be a false sense of security
                since you didn't have that much data to begin with.
    """
    phase2_ratings = results["phase2_ratings"]

    T = np.array([t for (t, x, _) in phase2_ratings]).max()
    assert 0 == np.array([t for (t, x, _) in phase2_ratings]).min()

    A_t = np.array([t for (t, x, _) in phase2_ratings]).astype(np.float64) / T
    assert 0.0 == A_t.min()
    assert 1.0 == A_t.max()

    for (t, x, _) in phase2_ratings:
        assert x in ["good", "bad"]
    A_x = np.array([x=="good" for (t, x, _) in phase2_ratings]).astype(np.float64)


    (a_MLE, b_MLE, loss_at_MLE) = get_MLE_from_observations(A_t, A_x)

    results = {'a_MLE': a_MLE, 'b_MLE': b_MLE, 'N': len(A_x)}

    analysis_results = analyze_robustness_of_conclusion(
        a_MLE, b_MLE, len(A_x),
        interval = 0.1, histogram_output_path=histogram_output_path)

    for k in ['mass_on_right', 'mass_in_middle', 'interval']:
        results[k] = analysis_results[k]

    # This is the heavy part of the computation. Maybe you don't care about it.
    # Maybe I should add a flag to disable this.

    if want_N95:
        if 0.95 <= results['mass_on_right']:
            results['N_required_for_95pc_confidence'] = int(results['N'])
        else:
            # We don't want to be too wasteful here, because we're trying to
            # hit the first success in some exponential search. Kind of like
            # a bisection search, but we don't want to overshoot by too much.
            def f(N95):
                analysis_results = analyze_robustness_of_conclusion(a_MLE, b_MLE, N95, interval = 0.1)
                return 0.95 <= analysis_results['mass_on_right']

            (N95_low, N95_high) = search_by_doubling(f, results['N'], alpha=3)
            # print((N95_low, N95_high))
            (N95_low, N95_high) = search_by_doubling(f, N95_low, alpha=1.5)
            # print((N95_low, N95_high))
            # (N95_low, N95_high) = search_by_doubling(f, N95_low, alpha=1.2)
            # print((N95_low, N95_high))
            results['N_required_for_95pc_confidence'] = int(0.5*(N95_low+N95_high))
    else:
        results['N_required_for_95pc_confidence'] = None

    return results


def search_by_doubling(f, N0, alpha):
    N = N0
    while not f(N):
        N = int(N * alpha)
    return (int(N / alpha), N)


"""
Then these are the internal methods used for the analysis.
The "copper" in the porcelain-vs-copper analogy.
"""

def analyze_robustness_of_conclusion(
    a_MLE, b_MLE, nbr_obs,
    interval = 0.1, histogram_output_path=None):

    def f(a, b, nbr_obs):
        (sub_a, sub_b, sub_A_t, sub_A_x) = generate_one_user(mu_a=a, std_a=0.0, mu_b=b, std_b=0.0, nbr_obs=nbr_obs)
        (sub_a_MLE, sub_b_MLE, sub_loss_at_MLE) = get_MLE_from_observations(sub_A_t, sub_A_x)
        return sub_a_MLE

    nbr_reps = 1000
    A_data = np.array([f(a_MLE, b_MLE, nbr_obs) for _ in range(nbr_reps)])

    # Now we'd like to know how often these values in A_data ended up
    # comparable to a_MLE or better.
    # Let's say we want to know
    #     1) how often A_data ends up in [a_MLE-interval, a_MLE+interval]
    #     2) how often A_data ends up in [a_MLE-interval, infinity]

    interval = 0.1
    mass_in_middle = ((a_MLE-interval <= A_data) * (A_data <= a_MLE+interval)).mean()
    mass_on_right = (a_MLE-interval <= A_data).mean()

    results = { 'A_data': A_data,
                'a_MLE': a_MLE,
                'b_MLE': b_MLE,
                'nbr_obs': nbr_obs,
                'interval': interval,
                'mass_on_right': mass_on_right,
                'mass_in_middle': mass_in_middle}

    if histogram_output_path is not None:
        bins = np.linspace(-0.5, 0.5, 50)

        plt.hist(A_data, bins)
        plt.axvline(x=a_MLE-interval, linewidth=1.0, linestyle='--', c='#ffa500')
        plt.axvline(x=a_MLE+interval, linewidth=1.0, linestyle='--', c='#ffa500')
        plt.title("nbr_obs %d, mass %0.2f in [%0.2f, %0.2f], mass %0.2f in [%0.2f, inf]" %
                  (nbr_obs, mass_in_middle, a_MLE - interval, a_MLE + interval, mass_on_right, a_MLE - interval))
        pylab.draw()
        pylab.savefig(histogram_output_path, dpi=200)
        pylab.close()

        results['histogram_output_path'] = histogram_output_path

        # Save the results to a pickle file, because it's good when we are able
        # to reproduce plots.
        pickle_output_path = histogram_output_path.replace("png", "pkl")
        with open(pickle_output_path, "wb") as f:
            pickle.dump(results, f)

    return results



def generate_one_user(mu_a, std_a, mu_b, std_b, nbr_obs, want_random_t=False):
    """
    The parameters (a,b) will be sampled for that user.
    It will define theta(t) = clip(a*t+b, 0, 1).

    We'll draw them from a distribution with
        a ~ N(mu_a, std_a^2)
        b ~ N(mu_b, std_b^2)
    Later on, we'll run some kind of inference that will try to recover mu_a and mu_b.

    The point is that we should pick a common (mu_a, std_a, mu_b, std_b)
    for a collection of users who will each have their different values of (a, b).

    This method samples `nbr_obs` measurements x ~ Bernouilli(theta(t))
    and returns them as an array
        t of shape (nbr_obs,), containing values in interval [0, 1]
        x of shape (nbr_obs,), containing values 0 or 1
    where `t` is spread across the interval [0, 1].
    This `t` is uniformly at random if `want_random_t`,
    but otherwise it's uniformly in a deterministic way.
    """
    valid_values_found = False
    while not valid_values_found:
        a = mu_a + std_a * np.random.randn()
        b = mu_b + std_b * np.random.randn()
        theta_bounds = a * np.array([0,1]) + b  # max and min are at either ends
        if 1.0 < np.max(theta_bounds) or np.min(theta_bounds) < 0.0:
            continue
        else:
            valid_values_found = True

    if want_random_t:
        A_t = np.random.rand(nbr_obs)
    else:
        A_t = np.linspace(0, 1, nbr_obs)

    A_theta_t = a * A_t + b
    # this is a way to sample from our Bernouilli
    A_x = (np.random.rand(nbr_obs) <= A_theta_t).astype(np.float32)

    return (a, b, A_t, A_x)


def is_theta_in_bounds(a, b):
    """
    Returns True if theta(t) is always valid.
    Otherwise returns False.

    That is, whether
        theta(t) = a*t + b is always in the interval [0, 1]
        for all values of t in [0, 1].
    """
    theta_bounds = a * np.array([0,1]) + b  # max and min are at either ends
    does_it_spill = 1.0 < np.max(theta_bounds) or np.min(theta_bounds) < 0.0
    return not does_it_spill


def generate_one_user_normal(mu_a, std_a, mu_b, std_b, nbr_obs, want_random_t=False):
    """
    The parameters (a,b) will be sampled for that user.
    It will define theta(t) = clip(a*t+b, 0, 1).

    We'll draw them from a distribution with
        a ~ N(mu_a, std_a^2)
        b ~ N(mu_b, std_b^2)
    Later on, we'll run some kind of inference that will try to recover mu_a and mu_b.

    The point is that we should pick a common (mu_a, std_a, mu_b, std_b)
    for a collection of users who will each have their different values of (a, b).

    This method samples `nbr_obs` measurements x ~ Bernouilli(theta(t))
    and returns them as an array
        t of shape (nbr_obs,), containing values in interval [0, 1]
        x of shape (nbr_obs,), containing values 0 or 1
    where `t` is spread across the interval [0, 1].
    This `t` is uniformly at random if `want_random_t`,
    but otherwise it's uniformly in a deterministic way.
    """
    valid_values_found = False
    while not valid_values_found:
        a = mu_a + std_a * np.random.randn()
        b = mu_b + std_b * np.random.randn()
        if is_theta_in_bounds(a, b):
            valid_values_found = True

    if want_random_t:
        A_t = np.random.rand(nbr_obs)
    else:
        A_t = np.linspace(0, 1, nbr_obs)

    A_theta_t = a * A_t + b
    # this is a way to sample from our Bernouilli
    A_x = (np.random.rand(nbr_obs) <= A_theta_t).astype(np.float32)

    return (a, b, A_t, A_x)


def generate_one_user_uniform(nbr_obs, want_random_t=False):
    """
    Same as `generate_one_user_normal` but we'll get our values of (a, b)
    from
        a ~ Unif([-1, 1])
        b ~ Unif([0.05, 0.95])
    and toss out all the invalid values that would lead
    to theta(t) being out of bounds.
    """
    valid_values_found = False
    while not valid_values_found:
        a = -1 + 2*np.random.rand()
        b = np.random.rand()
        b = np.clip(b, 0.05, 0.95)
        if is_theta_in_bounds(a, b):
            valid_values_found = True

    if want_random_t:
        A_t = np.random.rand(nbr_obs)
    else:
        A_t = np.linspace(0, 1, nbr_obs)

    A_theta_t = a * A_t + b
    # this is a way to sample from our Bernouilli
    A_x = (np.random.rand(nbr_obs) <= A_theta_t).astype(np.float64)

    return (a, b, A_t, A_x)


def loglikelihood(a, b, A_t, A_x):
    """
    Helper function for other methods.
    This is the cross-entropy computed manually,
    based on the hypothesis that theta(t) = a*t+b.

    Note that we're not dividing by N.
    """
    # If the values would produce an illegal theta,
    # return negative infinity (which is log(0.0)).
    if not is_theta_in_bounds(a, b):
        return -np.inf

    eps = 1e-12
    A_theta_t = a * A_t + b
    term1 = A_x * np.log( A_theta_t + eps )
    term2 = (1-A_x) * np.log( 1 - A_theta_t + eps)
    return (term1 + term2).sum()


def get_MLE_from_observations(A_t, A_x):
    """
        Returns (a MLE, b MLE, loss value at MLE).
    """

    assert A_t.shape == A_x.shape
    assert np.all(np.logical_or(A_x == 0.0, A_x == 1.0))

    def get_loss(w):
        assert w.shape == (2,)
        (a, b) = w

        # The `loglikelihood` function checks for invalid values of (a, b).
        loss = -loglikelihood(a, b, A_t, A_x)
        assert 0.0 < loss
        return loss

    # We can start from any initial guess since this is a convex problem
    # and we'll end up at the global optima in all cases.
    w0 = np.array([0.5, 0.0])
    res = scipy.optimize.minimize(get_loss, w0)
    # The `res.x` component is what we want from scipy.optimize
    # but we'll return the loss as well.

    # Returns (a MLE, b MLE, loss value at MLE)
    return (res.x[0], res.x[1], res.fun)


def get_loglik_given_a_with_b_MLE(a, A_t, A_x):
    """
        You have a value of `a` and you want to know what is the
        loglikelihood of the data given the best value of `b`.

        That is, with that `a`, find the b_MLE and use it to
        compute the loglikelihood. This is a justifiable thing to
        do in order to compare various values of `a`.

        Returns a log likelihood value.
    """
    assert A_t.shape == A_x.shape
    assert np.all(np.logical_or(A_x == 0.0, A_x == 1.0))

    def get_loss(wb):
        # Uses the `a` from the parent function.
        # This is because we optimize only over `b`.
        assert wb.shape == (1,)
        b = wb[0]

        # The `loglikelihood` function checks for invalid values of (a, b).
        loss = -loglikelihood(a, b, A_t, A_x)
        assert 0.0 < loss
        return loss

    # We can start from any initial guess since this is a convex problem
    # and we'll end up at the global optima in all cases.
    # We do need to start from a valid value, though,

    # I got this idea from sketching out the boundaries on a paper.
    # This is not an intuitive formula based on the context,
    # but it requires thinking for a moment.
    initial_valid_b = np.clip(0.5 - a, 0, 1)

    wb0 = np.array([initial_valid_b])
    res = scipy.optimize.minimize(get_loss, wb0)
    # The `res.x` component is what we want from scipy.optimize
    # but we'll return the loss as well.

    # Flip the sign again because of the expected returned value
    # from the parent function.
    loglik = -res.fun

    # We're not returning the value of `b MLE` here because it's besides the point.
    return loglik


def get_loglik_given_a_marginalizing_b(A_t, A_x):
    """
        You have a value of `a` and you want to know what is the
        loglikelihood of the data given that you're marginalizing `b`.
    """
    pass



def test00():
    """
    Test that we can recover the original values of (a, b) to some precision, given enough observations.
    """

    (a, b, A_t, A_x) = generate_one_user_normal(mu_a=0.25, std_a=0.2, mu_b=0.5, std_b=0.2, nbr_obs=int(1e6))
    print("Parameters are (a, b) : (%0.2f, %0.2f)" % (a, b))

    (a_MLE, b_MLE, loss_at_MLE) = get_MLE_from_observations(A_t, A_x)
    print("Parameters MLE are (a, b) : (%0.2f, %0.2f) with loss %0.2f." % (a_MLE, b_MLE, loss_at_MLE))

    npt.assert_allclose(a, a_MLE, atol=0.01)
    npt.assert_allclose(b, b_MLE, atol=0.01)


def test01():
    """
    Test that we can recover the original values of (a, b) to some precision, given enough observations.
    """

    (a, b, A_t, A_x) = generate_one_user_uniform(nbr_obs=int(1e6))
    print("Parameters are (a, b) : (%0.2f, %0.2f)" % (a, b))

    (a_MLE, b_MLE, loss_at_MLE) = get_MLE_from_observations(A_t, A_x)
    print("Parameters MLE are (a, b) : (%0.2f, %0.2f) with loss %0.2f." % (a_MLE, b_MLE, loss_at_MLE))

    npt.assert_allclose(a, a_MLE, atol=0.01)
    npt.assert_allclose(b, b_MLE, atol=0.01)


def test02():
    """
    Test to make sure that the loglik at the MLE is better than the loglik at other values of `a`.

    And while we're at it, we can test whether `get_loglik_given_a_with_b_MLE`
    is able to find the b_MLE when given a_MLE. We do this by checking to see
    if it is able to achieve the same loss value as when the original (a_MLE, b_MLE)
    were used.

    Here we used a little nbr_obs because we aren't really checking to make sure
    that we recover the (true_a, true_b). This isn't important because it's tested
    elsewhere, and it requires a larger value of nbr_obs.
    """

    (true_a, true_b, A_t, A_x) = generate_one_user_uniform(nbr_obs=int(1e5))

    (a_MLE, b_MLE, loss_at_MLE) = get_MLE_from_observations(A_t, A_x)

    for a in np.linspace(0.01, 0.99, 20):
        loss_at_a = -get_loglik_given_a_with_b_MLE(a, A_t, A_x)
        npt.assert_array_less(loss_at_MLE, loss_at_a + 1e-6)

    loss_at_MLE_recomputed = -get_loglik_given_a_with_b_MLE(a_MLE, A_t, A_x)
    npt.assert_allclose(loss_at_MLE, loss_at_MLE_recomputed, rtol=0.1)


guillaume_00_results = {
    "params": {"batch_size": 8, "dropout": 0.15, "end_time": 1588003140.4521294,
                "fotf": False, "lr": 0.0005, "num_conv": 64, "num_mlp": 128,
                "phase": 1, "phase1_count": 100, "save_interval": 50,
                "start_time": 1588002123.921673, "trials_per_model": 25, "weight_decay": 0},
    "user_email": "guillaume.alain.umontreal@gmail.com", "user_name": "Guillaume Alain",
    "start_time": 1588002123.921673, "end_time": 1588003140.4521294,
    "phase2_ratings": [ [0, "bad"], [0, "bad"], [0, "bad"], [0, "bad"], [0, "bad"], [0, "bad"],
                        [0, "bad"], [0, "bad"], [0, "bad"], [0, "bad"], [0, "bad"], [0, "good"],
                        [0, "good"], [0, "bad"], [0, "good"], [0, "bad"], [0, "bad"], [0, "bad"],
                        [0, "bad"], [0, "bad"], [0, "bad"], [0, "bad"], [0, "bad"], [0, "good"],
                        [0, "good"], [50, "good"], [50, "bad"], [50, "bad"], [50, "bad"],
                        [50, "bad"], [50, "bad"], [50, "bad"], [50, "bad"], [50, "bad"],
                        [50, "good"], [50, "bad"], [50, "bad"], [50, "good"], [50, "bad"],
                        [50, "good"], [50, "good"], [50, "good"], [50, "bad"], [50, "bad"],
                        [50, "bad"], [50, "bad"], [50, "bad"], [50, "bad"], [50, "bad"],
                        [50, "bad"], [100, "good"], [100, "good"], [100, "bad"], [100, "good"],
                        [100, "good"], [100, "good"], [100, "good"], [100, "bad"], [100, "bad"],
                        [100, "bad"], [100, "bad"], [100, "bad"], [100, "bad"], [100, "good"],
                        [100, "bad"], [100, "good"], [100, "good"], [100, "good"], [100, "good"],
                        [100, "good"], [100, "bad"], [100, "good"], [100, "bad"], [100, "bad"],
                        [100, "bad"]]}

fred_00_results = {"params": {"batch_size": 8, "dropout": 0.15, "end_time": 1588088788.1216822, "fotf": False, "lr": 0.0005, "num_conv": 64, "num_mlp": 128, "phase": 2, "phase1_count": 100, "save_interval": 50, "start_time": 1588086183.7396204, "trials_per_model": 25, "weight_decay": 0}, "user_email": "frederic.osterrath@mila.quebec", "user_name": "Frederic Osterrath", "start_time": 1588086183.7396204, "end_time": 1588088788.1216822, "phase2_ratings": [[0, "bad"], [0, "bad"], [0, "bad"], [0, "bad"], [0, "good"], [0, "good"], [0, "bad"], [0, "bad"], [0, "bad"], [0, "good"], [0, "bad"], [0, "bad"], [0, "good"], [0, "bad"], [0, "good"], [0, "bad"], [0, "bad"], [0, "good"], [0, "good"], [0, "bad"], [0, "bad"], [0, "good"], [0, "bad"], [0, "good"], [0, "bad"], [50, "good"], [50, "good"], [50, "good"], [50, "good"], [50, "bad"], [50, "bad"], [50, "good"], [50, "good"], [50, "bad"], [50, "bad"], [50, "good"], [50, "good"], [50, "bad"], [50, "bad"], [50, "good"], [50, "bad"], [50, "bad"], [50, "good"], [50, "bad"], [50, "good"], [50, "good"], [50, "bad"], [50, "good"], [50, "bad"], [50, "bad"], [100, "bad"], [100, "good"], [100, "good"], [100, "bad"], [100, "good"], [100, "bad"], [100, "good"], [100, "good"], [100, "bad"], [100, "bad"], [100, "bad"], [100, "good"], [100, "good"], [100, "good"], [100, "good"], [100, "bad"], [100, "bad"], [100, "bad"], [100, "bad"], [100, "bad"], [100, "bad"], [100, "good"], [100, "bad"], [100, "bad"], [100, "bad"]]}
guillaume_01_results = {"params": {"batch_size": 8, "dropout": 0.15, "fotf": False, "lr": 0.0005, "num_conv": 64, "num_mlp": 128, "phase": 2, "phase1_count": 100, "save_interval": 50, "trials_per_model": 25, "weight_decay": 0}, "user_email": "guillaume.alain.umontreal@gmail.com", "user_name": "Guillaume Alain", "start_time": 1588109095.954791, "end_time": 1588110512.1466422, "phase1_ratings": [["bad", 1588109122.113052], ["good", 1588109132.0052826], ["good", 1588109136.7050369], ["good", 1588109143.6451876], ["good", 1588109150.7097025], ["good", 1588109158.2725399], ["bad", 1588109162.1340876], ["good", 1588109167.742795], ["bad", 1588109171.9531674], ["good", 1588109186.1136444], ["bad", 1588109195.9507627], ["bad", 1588109201.225646], ["bad", 1588109208.9833894], ["bad", 1588109245.0551574], ["good", 1588109252.7142], ["good", 1588109280.2260313], ["good", 1588109289.3821778], ["good", 1588109295.595939], ["bad", 1588109301.031955], ["bad", 1588109307.9271], ["bad", 1588109322.6466882], ["bad", 1588109332.0153859], ["bad", 1588109337.4777665], ["bad", 1588109344.3291447], ["bad", 1588109351.3862064], ["good", 1588109356.1348076], ["good", 1588109366.2291148], ["bad", 1588109370.1730254], ["bad", 1588109383.4833143], ["good", 1588109389.7241066], ["bad", 1588109393.495896], ["bad", 1588109398.0708976], ["good", 1588109404.4227352], ["bad", 1588109409.0550823], ["bad", 1588109414.3873973], ["bad", 1588109422.8417118], ["bad", 1588109428.3510673], ["bad", 1588109433.7262816], ["good", 1588109440.8531046], ["good", 1588109446.2837138], ["bad", 1588109453.2306025], ["good", 1588109459.5522845], ["good", 1588109468.0208352], ["good", 1588109474.0711193], ["bad", 1588109478.6804452], ["good", 1588109484.0917208], ["good", 1588109489.5077353], ["good", 1588109496.4007316], ["bad", 1588109500.3207738], ["bad", 1588109506.4004812], ["good", 1588109513.567616], ["bad", 1588109522.5949812], ["good", 1588109527.8308825], ["bad", 1588109533.331931], ["good", 1588109537.9162872], ["bad", 1588109543.3053637], ["bad", 1588109547.2820747], ["bad", 1588109555.204146], ["bad", 1588109563.7537405], ["good", 1588109579.269995], ["good", 1588109778.3214161], ["bad", 1588109784.395038], ["bad", 1588109789.777353], ["bad", 1588109794.3455164], ["bad", 1588109799.9671257], ["bad", 1588109805.2847562], ["good", 1588109810.7293153], ["bad", 1588109851.2823913], ["bad", 1588109889.681631], ["bad", 1588109894.3475444], ["good", 1588109899.9334733], ["bad", 1588109904.5076013], ["bad", 1588109912.3544965], ["bad", 1588109916.5132256], ["bad", 1588109921.8861322], ["bad", 1588109927.8966208], ["good", 1588109933.9399633], ["good", 1588109942.2956405], ["bad", 1588109946.897281], ["good", 1588109952.823161], ["good", 1588109958.2270484], ["bad", 1588109964.2988548], ["bad", 1588109968.1247394], ["good", 1588109973.5218222], ["good", 1588109978.9097865], ["bad", 1588109982.6717944], ["bad", 1588109988.0653374], ["good", 1588109994.220618], ["bad", 1588109999.5658765], ["good", 1588110006.0521307], ["bad", 1588110009.8848934], ["bad", 1588110015.9423082], ["bad", 1588110021.9550664], ["bad", 1588110027.9929264], ["bad", 1588110033.3699505], ["good", 1588110038.8122027], ["bad", 1588110045.8770702], ["bad", 1588110051.3894641], ["good", 1588110057.4874363], ["good", 1588110063.9049017]], "phase2_ratings": [[0, "bad", 1588110152.21168], [0, "bad", 1588110166.7871335], [0, "bad", 1588110185.060149], [0, "bad", 1588110189.24618], [0, "good", 1588110194.4280343], [0, "good", 1588110228.9742665], [0, "bad", 1588110231.9524875], [0, "bad", 1588110291.5505295], [0, "bad", 1588110305.6734455], [0, "bad", 1588110336.1154237], [0, "bad", 1588110355.2269118], [0, "bad", 1588110364.4183004], [0, "bad", 1588110374.0404084], [0, "good", 1588110377.785443], [0, "good", 1588110396.17151], [0, "bad", 1588110403.312605], [0, "good", 1588110407.7705767], [0, "bad", 1588110418.471284], [0, "bad", 1588110441.9199834], [0, "bad", 1588110446.2331843], [0, "bad", 1588110450.168099], [0, "bad", 1588110456.381091], [0, "good", 1588110487.2334483], [0, "bad", 1588110491.432273], [0, "bad", 1588110500.0056894], [50, "bad", 1588110143.9341867], [50, "bad", 1588110148.8948529], [50, "good", 1588110213.493873], [50, "good", 1588110222.7905512], [50, "good", 1588110236.8931859], [50, "bad", 1588110245.4984238], [50, "good", 1588110254.2864788], [50, "good", 1588110258.670286], [50, "bad", 1588110262.228402], [50, "good", 1588110274.7197478], [50, "bad", 1588110279.114836], [50, "good", 1588110283.6346948], [50, "good", 1588110297.699292], [50, "bad", 1588110302.1417115], [50, "bad", 1588110310.743187], [50, "bad", 1588110315.4089687], [50, "good", 1588110320.07587], [50, "good", 1588110345.9710028], [50, "good", 1588110350.7680473], [50, "bad", 1588110387.8128505], [50, "bad", 1588110400.8283582], [50, "bad", 1588110415.4173968], [50, "bad", 1588110426.4809704], [50, "good", 1588110464.5196424], [50, "bad", 1588110473.1717484], [100, "good", 1588110157.5156531], [100, "good", 1588110162.8526382], [100, "good", 1588110171.8442452], [100, "bad", 1588110176.2967248], [100, "bad", 1588110179.9635253], [100, "bad", 1588110201.6142201], [100, "bad", 1588110205.7662125], [100, "good", 1588110218.2473276], [100, "good", 1588110240.9746525], [100, "good", 1588110250.1804857], [100, "bad", 1588110265.8986902], [100, "good", 1588110270.298191], [100, "good", 1588110289.446272], [100, "good", 1588110331.237661], [100, "bad", 1588110341.6211467], [100, "good", 1588110359.2260337], [100, "bad", 1588110371.3291008], [100, "bad", 1588110384.74943], [100, "bad", 1588110434.5392802], [100, "bad", 1588110467.8918042], [100, "good", 1588110478.5601983], [100, "bad", 1588110480.6994083], [100, "bad", 1588110497.3962183], [100, "bad", 1588110506.7257092], [100, "good", 1588110512.146553]]}
remi_00_results = {"params": {"batch_size": 8, "dropout": 0.15, "fotf": False, "lr": 0.0005, "num_conv": 64, "num_mlp": 128, "phase": 2, "phase1_count": 100, "save_interval": 50, "trials_per_model": 25, "weight_decay": 0}, "user_email": "remiptudem@gmail.com", "user_name": "R\u00e9mi Pich\u00e9-Taillefer", "start_time": 1588118376.5797634, "end_time": 1588119913.0991209, "phase1_ratings": [["bad", 1588118410.6546223], ["good", 1588118418.740151], ["bad", 1588118423.2090743], ["bad", 1588118428.9872973], ["good", 1588118441.7262323], ["bad", 1588118476.828825], ["bad", 1588118486.9916914], ["bad", 1588118493.9120483], ["bad", 1588118504.1966608], ["bad", 1588118511.2867796], ["bad", 1588118526.2801347], ["bad", 1588118532.705912], ["bad", 1588118539.4695606], ["bad", 1588118544.893637], ["bad", 1588118562.57866], ["bad", 1588118569.0444772], ["bad", 1588118580.876323], ["bad", 1588118586.8275187], ["good", 1588118595.4736173], ["good", 1588118601.8712444], ["bad", 1588118608.2576625], ["bad", 1588118612.810098], ["good", 1588118618.0656757], ["bad", 1588118622.912705], ["bad", 1588118637.2889266], ["bad", 1588118642.08447], ["bad", 1588118657.638633], ["bad", 1588118664.2177007], ["good", 1588118675.809053], ["bad", 1588118680.241612], ["bad", 1588118687.0027273], ["bad", 1588118696.8358853], ["bad", 1588118701.986702], ["bad", 1588118715.097077], ["good", 1588118726.4738996], ["bad", 1588118737.53457], ["bad", 1588118751.6942236], ["bad", 1588118762.398608], ["bad", 1588118770.4536345], ["good", 1588118783.65424], ["bad", 1588118788.740541], ["bad", 1588118796.446965], ["bad", 1588118803.6678708], ["bad", 1588118811.3239598], ["bad", 1588118822.4640548], ["bad", 1588118832.2348404], ["bad", 1588118839.6326597], ["bad", 1588118846.4644265], ["good", 1588118856.8915348], ["bad", 1588118866.8388426], ["bad", 1588118877.577341], ["bad", 1588118886.20667], ["bad", 1588118893.380927], ["good", 1588118898.5770695], ["bad", 1588118908.3335192], ["good", 1588118917.253657], ["good", 1588118928.2779675], ["good", 1588118936.5687716], ["good", 1588118945.0454495], ["bad", 1588118959.6361954], ["good", 1588118972.8972785], ["bad", 1588118979.5917969], ["good", 1588118989.2739117], ["bad", 1588118995.271682], ["bad", 1588119009.5370142], ["good", 1588119014.0002098], ["bad", 1588119027.8594193], ["good", 1588119036.331048], ["good", 1588119045.2666397], ["bad", 1588119053.2381117], ["good", 1588119064.1445298], ["bad", 1588119068.7420757], ["bad", 1588119076.9185429], ["bad", 1588119080.789636], ["bad", 1588119089.1886768], ["bad", 1588119094.4365232], ["bad", 1588119102.9293337], ["bad", 1588119108.1415942], ["bad", 1588119112.7284367], ["good", 1588119128.2049098], ["bad", 1588119134.983743], ["good", 1588119143.8108664], ["bad", 1588119151.4290211], ["bad", 1588119160.5148928], ["bad", 1588119165.947573], ["bad", 1588119178.5631034], ["bad", 1588119188.7309022], ["bad", 1588119194.0547273], ["bad", 1588119201.0214093], ["good", 1588119209.8679209], ["bad", 1588119218.1896892], ["good", 1588119230.2528381], ["bad", 1588119240.9641397], ["bad", 1588119249.619554], ["good", 1588119255.7854297], ["good", 1588119262.650587], ["bad", 1588119270.3893025], ["bad", 1588119283.4143074], ["good", 1588119300.309247], ["bad", 1588119312.6621342]], "phase2_ratings": [[0, "good", 1588119375.4167097], [0, "bad", 1588119386.7269402], [0, "bad", 1588119423.9888728], [0, "bad", 1588119443.929305], [0, "bad", 1588119506.7275872], [0, "bad", 1588119529.4662383], [0, "bad", 1588119538.1362727], [0, "bad", 1588119542.648301], [0, "bad", 1588119561.9077857], [0, "good", 1588119584.9512672], [0, "good", 1588119590.5606046], [0, "bad", 1588119596.827764], [0, "good", 1588119637.2175384], [0, "bad", 1588119641.1167204], [0, "bad", 1588119656.1767144], [0, "bad", 1588119661.3069057], [0, "bad", 1588119728.0590425], [0, "good", 1588119768.276136], [0, "bad", 1588119773.3080847], [0, "bad", 1588119817.4646692], [0, "bad", 1588119847.2246552], [0, "bad", 1588119860.0151153], [0, "bad", 1588119881.4979117], [0, "bad", 1588119886.5610993], [0, "good", 1588119901.9540842], [50, "bad", 1588119408.9734662], [50, "bad", 1588119414.1307392], [50, "good", 1588119420.053756], [50, "bad", 1588119429.909775], [50, "bad", 1588119449.9596896], [50, "bad", 1588119468.3562984], [50, "bad", 1588119478.794116], [50, "bad", 1588119497.965586], [50, "bad", 1588119534.0206468], [50, "good", 1588119551.9680064], [50, "bad", 1588119649.6276665], [50, "good", 1588119679.8150756], [50, "bad", 1588119718.8635862], [50, "bad", 1588119723.7144654], [50, "good", 1588119736.1060462], [50, "bad", 1588119740.0714262], [50, "bad", 1588119763.5385866], [50, "good", 1588119779.4911587], [50, "bad", 1588119787.5203817], [50, "bad", 1588119794.3282938], [50, "bad", 1588119826.6998827], [50, "good", 1588119832.363638], [50, "bad", 1588119855.051219], [50, "bad", 1588119866.4576745], [50, "bad", 1588119908.0234773], [100, "bad", 1588119372.0347345], [100, "good", 1588119404.16078], [100, "bad", 1588119461.990019], [100, "good", 1588119492.8818533], [100, "good", 1588119501.377944], [100, "bad", 1588119514.7513292], [100, "bad", 1588119521.2149107], [100, "good", 1588119575.107279], [100, "good", 1588119602.8624237], [100, "good", 1588119616.1368191], [100, "good", 1588119620.7880433], [100, "bad", 1588119626.635015], [100, "good", 1588119675.5033245], [100, "good", 1588119685.8050177], [100, "bad", 1588119692.2198575], [100, "bad", 1588119706.0452857], [100, "good", 1588119713.711002], [100, "bad", 1588119750.2983043], [100, "bad", 1588119759.7378387], [100, "good", 1588119801.1415436], [100, "bad", 1588119809.0267866], [100, "good", 1588119840.9157832], [100, "good", 1588119876.0551558], [100, "bad", 1588119894.6798506], [100, "good", 1588119913.0990448]]}
guillaume_02_results = {"params": {"batch_size": 8, "dropout": 0.15, "fotf": False, "lr": 0.0005, "num_conv": 64, "num_mlp": 128, "phase": 2, "phase1_count": 100, "save_interval": 10, "trials_per_model": 8, "weight_decay": 0}, "user_email": "guillaume.alain.umontreal@gmail.com", "user_name": "Guillaume Alain", "start_time": 1588207067.83527, "end_time": 1588208230.9716704, "phase1_ratings": [["good", 1588207079.4272208], ["good", 1588207086.4500675], ["bad", 1588207102.3499453], ["good", 1588207110.2984028], ["bad", 1588207114.864989], ["good", 1588207120.8955305], ["bad", 1588207127.1370165], ["bad", 1588207134.9020371], ["bad", 1588207142.6617165], ["good", 1588207149.6292176], ["bad", 1588207159.4884439], ["bad", 1588207165.6394157], ["bad", 1588207171.5757017], ["bad", 1588207175.9480307], ["bad", 1588207181.1938505], ["good", 1588207187.1248276], ["bad", 1588207190.8521776], ["bad", 1588207195.9061506], ["good", 1588207201.829177], ["bad", 1588207206.3997245], ["bad", 1588207216.7245913], ["bad", 1588207222.742374], ["bad", 1588207228.7530744], ["bad", 1588207231.7890928], ["bad", 1588207239.147212], ["good", 1588207245.2410557], ["bad", 1588207250.643948], ["bad", 1588207255.814754], ["good", 1588207261.7208533], ["good", 1588207268.068634], ["bad", 1588207280.610813], ["bad", 1588207285.7731025], ["bad", 1588207291.913363], ["bad", 1588207300.2123933], ["bad", 1588207308.3047662], ["good", 1588207314.1757927], ["bad", 1588207320.2272208], ["bad", 1588207324.8398788], ["good", 1588207330.0560088], ["bad", 1588207335.3840363], ["bad", 1588207346.4620502], ["good", 1588207352.2061205], ["bad", 1588207357.9974644], ["good", 1588207363.7471027], ["good", 1588207371.1333988], ["good", 1588207377.6462588], ["good", 1588207384.2288525], ["good", 1588207390.2979786], ["good", 1588207397.137932], ["bad", 1588207403.2114067], ["bad", 1588207413.5895126], ["bad", 1588207419.7985427], ["bad", 1588207425.78887], ["bad", 1588207432.0131965], ["bad", 1588207438.7329538], ["bad", 1588207444.8412542], ["good", 1588207451.1012752], ["good", 1588207457.070122], ["bad", 1588207462.246126], ["bad", 1588207471.3136752], ["bad", 1588207480.9993482], ["bad", 1588207488.5236447], ["bad", 1588207496.733136], ["bad", 1588207504.0967832], ["bad", 1588207510.744636], ["bad", 1588207517.371408], ["good", 1588207524.130206], ["bad", 1588207528.7864156], ["bad", 1588207533.9105935], ["bad", 1588207539.8950143], ["good", 1588207551.1055436], ["bad", 1588207555.784435], ["bad", 1588207562.3651087], ["bad", 1588207567.5402827], ["bad", 1588207573.394649], ["bad", 1588207579.2191408], ["bad", 1588207585.061526], ["bad", 1588207592.5387385], ["good", 1588207597.5156085], ["bad", 1588207602.6498942], ["bad", 1588207613.8720891], ["bad", 1588207618.117777], ["good", 1588207624.1450694], ["good", 1588207629.4169214], ["bad", 1588207635.6707323], ["good", 1588207642.460817], ["bad", 1588207646.1554773], ["bad", 1588207650.6217258], ["good", 1588207656.5136337], ["good", 1588207663.2084618], ["bad", 1588207672.1775646], ["bad", 1588207677.3813868], ["bad", 1588207682.6627584], ["bad", 1588207688.5189207], ["good", 1588207694.9741058], ["good", 1588207701.6800413], ["bad", 1588207707.0252826], ["bad", 1588207712.9142458], ["bad", 1588207720.1547723], ["bad", 1588207728.909938]], "phase2_ratings": [[0, "bad", 1588207820.5320039], [0, "bad", 1588207827.3196323], [0, "bad", 1588207869.8578951], [0, "bad", 1588207888.7256012], [0, "bad", 1588207896.5442083], [0, "good", 1588207963.2274697], [0, "bad", 1588208058.529129], [0, "good", 1588208216.2590003], [10, "bad", 1588207856.4015765], [10, "bad", 1588207944.549533], [10, "good", 1588207990.7614224], [10, "bad", 1588208051.4735708], [10, "bad", 1588208065.5954032], [10, "good", 1588208090.7754815], [10, "good", 1588208135.997922], [10, "good", 1588208170.3263125], [20, "bad", 1588207845.0009918], [20, "bad", 1588207925.3551648], [20, "bad", 1588208005.968805], [20, "bad", 1588208099.346356], [20, "bad", 1588208152.7180727], [20, "bad", 1588208159.7822099], [20, "good", 1588208187.4020846], [20, "bad", 1588208209.9015627], [30, "bad", 1588207833.054916], [30, "bad", 1588207906.9259477], [30, "bad", 1588207918.8896403], [30, "good", 1588208043.0462272], [30, "good", 1588208107.5382824], [30, "good", 1588208201.3483908], [30, "good", 1588208206.2518525], [30, "bad", 1588208230.971597], [40, "bad", 1588207837.0068414], [40, "good", 1588207934.9676356], [40, "bad", 1588207967.1965985], [40, "bad", 1588207977.1028965], [40, "good", 1588208003.1145213], [40, "bad", 1588208010.4067044], [40, "bad", 1588208139.7783244], [40, "good", 1588208195.9812913], [50, "good", 1588207814.0576057], [50, "good", 1588207865.6400461], [50, "bad", 1588207930.3793173], [50, "bad", 1588207998.50647], [50, "bad", 1588208070.8015695], [50, "bad", 1588208085.1832714], [50, "good", 1588208120.081697], [50, "bad", 1588208178.6982334], [60, "bad", 1588207840.9121313], [60, "bad", 1588207851.7686272], [60, "good", 1588207948.2393174], [60, "bad", 1588208080.574035], [60, "bad", 1588208094.9084146], [60, "bad", 1588208110.8019102], [60, "bad", 1588208115.4247556], [60, "bad", 1588208220.3012526], [70, "bad", 1588207885.3294477], [70, "bad", 1588207940.6247573], [70, "good", 1588207959.0613952], [70, "bad", 1588207971.7070544], [70, "good", 1588208062.1509063], [70, "bad", 1588208102.7223225], [70, "bad", 1588208130.8898995], [70, "bad", 1588208182.7426882], [80, "bad", 1588207824.5755563], [80, "good", 1588207860.8590517], [80, "good", 1588207892.6912403], [80, "bad", 1588207981.3934782], [80, "good", 1588208076.165869], [80, "bad", 1588208124.0675662], [80, "bad", 1588208127.413147], [80, "good", 1588208174.8357186], [90, "good", 1588207875.2586696], [90, "good", 1588207901.9680657], [90, "bad", 1588207922.2575908], [90, "bad", 1588207986.2845657], [90, "bad", 1588207994.4396443], [90, "bad", 1588208143.0672529], [90, "bad", 1588208146.489809], [90, "good", 1588208191.9945474], [100, "good", 1588207817.7130399], [100, "bad", 1588207880.1693327], [100, "good", 1588207914.3756945], [100, "good", 1588207953.517419], [100, "bad", 1588208015.6128976], [100, "bad", 1588208047.1498153], [100, "good", 1588208055.3070648], [100, "bad", 1588208226.180352]]}

def run00():
    # results = guillaume_01_results
    results = guillaume_02_results
    results = analyze_phase_2(results, "debug_histogram.png")
    print(results)

if __name__ == "__main__":

    run00()

    if False:
        test00()
        test01()
        test02()
