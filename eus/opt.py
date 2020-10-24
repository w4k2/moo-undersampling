import autograd.numpy as anp
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.base import clone

from pymoo.algorithms.so_genetic_algorithm import GA
from pymoo.algorithms.nsga2 import NSGA2
from pymoo.util.termination.default import SingleObjectiveSpaceToleranceTermination
from pymoo.util.termination.default import MultiObjectiveSpaceToleranceTermination
from pymoo.model.crossover import Crossover
from pymoo.model.mutation import Mutation
from pymoo.model.problem import Problem
from pymoo.model.sampling import Sampling
from pymoo.optimize import minimize
from pymoo.configuration import Configuration

Configuration.show_compile_hint = False

DEF_C_PROB = 0.25
DEF_M_PROB = 0.25
DEF_TEST_SIZE = 0.2
DEF_RANDOM_STATE = 90210
DEF_P_SIZE = 50
DEF_SEED = DEF_RANDOM_STATE
DEF_MAX_EVALS = 10000


class USSampling(Sampling):
    def _do(self, problem, n_samples, **kwargs):
        X = np.full((n_samples, problem.n_var), False, dtype=np.bool)

        for k in range(n_samples):
            I = np.random.permutation(problem.n_var)[:problem.n_max]
            X[k, I] = True

        return X


class USCrossover(Crossover):
    def __init__(self, prob=DEF_C_PROB):
        super().__init__(2, 1, prob=prob)

    def _do(self, problem, X, **kwargs):
        n_parents, n_matings, n_var = X.shape

        _X = np.full((self.n_offsprings, n_matings, problem.n_var), False)

        for k in range(n_matings):
            p1, p2 = X[0, k], X[1, k]

            both_are_true = np.logical_and(p1, p2)
            _X[0, k, both_are_true] = True

            n_remaining = problem.n_max - np.sum(both_are_true)

            I = np.where(np.logical_xor(p1, p2))[0]

            S = I[np.random.permutation(len(I))][:n_remaining]
            _X[0, k, S] = True

        return _X


class USMutation(Mutation):
    def __init__(self, prob=DEF_M_PROB):
        super().__init__()
        self.prob = prob

    @staticmethod
    def mutation(x):
        is_false = np.where(x == False)[0]
        is_true = np.where(x == True)[0]
        x[np.random.choice(is_false)] = True
        x[np.random.choice(is_true)] = False
        return x

    def _do(self, problem, X, **kwargs):
        mut_p = np.random.random(len(X)) < self.prob
        if np.any(mut_p):
            X[mut_p] = np.array([self.mutation(x) for x in X[mut_p]])
        return X


class USProblem(Problem):
    def __init__(self, X, y, estimator, score_func, n_obj=1, test_size=DEF_TEST_SIZE, random_state=DEF_RANDOM_STATE):
        self.estimator = estimator
        self.score_func = score_func

        self.X_train, self.X_test, self.y_train, self.y_test = \
                train_test_split(X, y, test_size=test_size, random_state=random_state)

        binc = np.bincount(self.y_train)
        assert (binc[0] > binc[1])

        self.L = np.where(self.y_train == 0)[0]
        self.n_max = binc[0] - binc[1]

        super().__init__(n_var=binc[0], n_obj=n_obj, elementwise_evaluation=True)

    def validation(self, x):
        s_map = np.ones(len(self.y_train), dtype=bool)
        s_map[self.L[x]] = False

        clf = clone(self.estimator)
        clf.fit(self.X_train[s_map], self.y_train[s_map])
        y_pred = clf.predict(self.X_test)
        scores = self.score_func(self.y_test, y_pred)

        return clf, scores

    def _evaluate(self, x, out, *args, **kwargs):
        clf, score = self.validation(x)
        if type(score) is not list:
            score = [score]

        out["E"] = clf
        out["F"] = anp.column_stack(-1 * np.array(score))

def soo(X, y, estimator, score_func):
    n_obj = 1  # One fitness function
    problem = USProblem(X, y, estimator, score_func)

    algorithm = GA(
        pop_size=DEF_P_SIZE,
        sampling=USSampling(),
        crossover=USCrossover(),
        mutation=USMutation(),
    )

    termination = SingleObjectiveSpaceToleranceTermination(
        n_max_evals=DEF_MAX_EVALS
    )

    res = minimize(
        problem,
        algorithm,
        termination,
        seed=DEF_SEED,
        verbose=False,
        save_history=True
    )

    return res

def moo(X, y, estimator, score_func, n_obj):
    problem = USProblem(X, y, estimator, score_func, n_obj)

    algorithm = NSGA2(
        pop_size=DEF_P_SIZE,
        sampling=USSampling(),
        crossover=USCrossover(),
        mutation=USMutation(),
        # eliminate_duplicates=True
    )

    termination = MultiObjectiveSpaceToleranceTermination(
        n_max_evals=DEF_MAX_EVALS
    )

    res = minimize(
        problem,
        algorithm,
        termination,
        seed=DEF_SEED,
        verbose=False,
        save_history=True
    )

    return res
