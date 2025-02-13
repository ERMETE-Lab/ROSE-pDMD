import numpy as np

from scipy.interpolate import interp1d, CubicSpline
from pydmd import DMD, ParametricDMD, BOPDMD

from sklearn.utils.extmath import randomized_svd

import warnings
warnings.filterwarnings("ignore")

def get_mag_fenicsx(u_vec):
    """
    Function to get the magnitude of a vector field in FEniCSx format (data are assumed 2D).
    """
    u = u_vec[0::2]
    v = u_vec[1::2]
    return np.sqrt(u**2 + v**2)


def get_mag(u_vec):
    """
    Function to get the magnitude of a vector field in FEniCSx format (data are assumed 2D).
    """

    Nh = int(u_vec.shape[0]/2)

    u = u_vec[:Nh]
    v = u_vec[Nh:]
    return np.sqrt(u**2 + v**2)


def dmd_reconstruct(dmd_model, pod_modes, scaler = None, time=None):
    
    if time is not None:
        dmd_reconstructed = dmd_model.forecast(time).real.T
    else:   
        dmd_reconstructed = dmd_model.reconstructed_data.T.real

    if scaler is not None:
        dmd_reconstructed = scaler.inverse_transform(dmd_reconstructed)

    return np.dot(pod_modes, dmd_reconstructed.T)

def flatten_operator(A):
    _A = np.zeros((A.shape[0], A.shape[1]*A.shape[2]))
    for i in range(A.shape[0]):
        _A[i] = A[i].flatten()

    return _A

def unflatten_operator(_A, rank):
    A = np.zeros((_A.shape[0], rank, rank))
    for i in range(len(_A)):
        A[i] = _A[i].reshape(rank)

    return A

class ReducedOperatorInterpolation():
    
    def __init__(self, rank):
        self.rank = rank

    def fit(self, pod_coeff: np.ndarray, train_time: np.ndarray, test_time: np.ndarray,
            scaler = None, verbose = False):
        """
        Fit the POD coefficients with standard DMD algorithm. 
        The input data are assumed to be in the format (n_params, n_times, rank).
        """

        # Select the first rank POD coeff
        _pod_coeff = pod_coeff[:, :, :self.rank]

        if scaler is not None:
            self.scaler = scaler

            _flatten_coeff = _pod_coeff.reshape(-1, _pod_coeff.shape[-1])
            self.scaler.fit(_flatten_coeff)

            _pod_coeff = scaler.transform(_flatten_coeff).reshape(_pod_coeff.shape)
        else:
            self.scaler = None

        self.dmds = dict()
        self.train_pod_coeff = _pod_coeff

        for mu_i in range(_pod_coeff.shape[0]):
            
            if verbose:
                print(f"Training DMD for parameter {mu_i+1}/{_pod_coeff.shape[0]}", end="\r")
            self.dmds[mu_i] = DMD(svd_rank=-1)

            self.dmds[mu_i].fit(_pod_coeff[mu_i].T)

            self.dmds[mu_i].original_time["t0"]   = train_time[0]
            self.dmds[mu_i].original_time["tend"] = train_time[-1] - 1e-12
            self.dmds[mu_i].original_time["dt"]   = train_time[1]  - train_time[0]

            self.dmds[mu_i].dmd_time["t0"]   = test_time[0]
            self.dmds[mu_i].dmd_time["tend"] = test_time[-1] - 1e-12
            self.dmds[mu_i].dmd_time["dt"]   = test_time[1]  - test_time[0]

    def get_operators(self):
        """
        Get the reduced operators.
        """
        self.Aoperators = np.zeros((len(self.dmds), self.rank, self.rank))

        for pp in range(len(self.dmds)):
            _U, _, _ = self.dmds[pp].operator.compute_operator(self.train_pod_coeff[pp].T[:,:-1],self.train_pod_coeff[pp].T[:,1:])
            self.Aoperators[pp] = np.linalg.multi_dot([_U, self.dmds[pp].operator._Atilde, _U.T])

    def reduce_operators(self, rank_op_svd = None):
        """
        Reduce the operators to a lower rank.
        """
        self.get_operators()

        Aop_flattened = flatten_operator(self.Aoperators).T

        if rank_op_svd is None:
            self.rank_op_svd = Aop_flattened.shape[1]
        else:
            self.rank_op_svd = rank_op_svd

        self.operator_svd = randomized_svd(Aop_flattened, n_components=self.rank_op_svd, n_iter='auto')

    def get_op_interpolants(self, train_params, interp = 'linear'):

        self.interpolants_op = dict()

        for jj in range(self.rank_op_svd):
            if interp == 'linear' or interp == 'cubic':
                self.interpolants_op[jj] = interp1d(train_params, self.operator_svd[2][jj], kind=interp)
            else:
                raise ValueError('Interpolation method not recognized.')

    def predict_op(self, param):

        _vh = np.zeros((self.rank_op_svd, 1))
        for jj in range(self.rank_op_svd):
            _vh[jj] = self.interpolants_op[jj](param)

        return np.dot(self.operator_svd[0], np.dot(np.diag(self.operator_svd[1]), _vh))

    def advance(self, x0, n_steps, param):
        """
        \mathbf{x}_{k+1} = \mathbf{A} \mathbf{x}_k
        """

        operator = self.predict_op(param).reshape(self.rank, self.rank)

        if self.scaler is not None:
            x0 = self.scaler.transform(x0.reshape(1, -1)).T
        
        x = np.zeros((n_steps, x0.shape[0]))
        x[0] = x0.flatten()

        for i in range(1, n_steps):
            x[i] = np.dot(operator, x[i-1])

        if self.scaler is not None:
            x = self.scaler.inverse_transform(x)

        return x

class ReducedKoopmanOperatorInterpolation():
    
    def __init__(self, rank):
        self.rank = rank

    def fit(self, pod_coeff: np.ndarray, train_time: np.ndarray,
            scaler = None, verbose = False, tol = 0.2, opt_verbose = False,
            **kwargs):
        """
        Fit the POD coefficients with standard DMD algorithm. 
        The input data are assumed to be in the format (n_params, n_times, rank).
        """

        # Select the first rank POD coeff
        _pod_coeff = pod_coeff[:, :, :self.rank]

        if scaler is not None:
            self.scaler = scaler

            _flatten_coeff = _pod_coeff.reshape(-1, _pod_coeff.shape[-1])
            self.scaler.fit(_flatten_coeff)

            _pod_coeff = scaler.transform(_flatten_coeff).reshape(_pod_coeff.shape)
        else:
            self.scaler = None

        self.dmds = dict()
        self.train_pod_coeff = _pod_coeff

        for mu_i in range(_pod_coeff.shape[0]):
            
            if verbose:
                print(f"Training DMD for parameter {mu_i+1}/{_pod_coeff.shape[0]}", end="\r")
            self.dmds[mu_i] = BOPDMD(
                                        svd_rank=-1,
                                        num_trials=0,                                    # Number of bagging trials to perform - 0 means no bagging.
                                        varpro_opts_dict={"tol": tol, "verbose": opt_verbose}, # Set convergence tolerance and use verbose updates.
                                        compute_A = True,
                                        **kwargs
                                    )

            self.dmds[mu_i].fit(_pod_coeff[mu_i].T, train_time)

    def forecast(self, model: dict, time):
        modes = model['modes']
        amplitudes = model['amplitudes']
        eigs = model['eigs']

        x = np.linalg.multi_dot([
            modes, 
            np.diag(amplitudes),
            np.exp(np.outer(eigs, time)),
        ]).real

        if self.scaler is not None:
            x = self.scaler.inverse_transform(x.T).T

        return x

    def get_kop(self):

        self.train_kop =    {
                                'amplitudes': np.asarray([self.dmds[mu_i].amplitudes      for mu_i in range(self.train_pod_coeff.shape[0])], dtype=np.complex64),
                                'eigs':       np.asarray([self.dmds[mu_i].eigs            for mu_i in range(self.train_pod_coeff.shape[0])], dtype=np.complex64),
                                'modes':      np.asarray([self.dmds[mu_i].modes.flatten() for mu_i in range(self.train_pod_coeff.shape[0])], dtype=np.complex64)
                            }

    def get_kop_interpolants(self, train_params, interp = 'CubicSpline'):
        
        self.interpolants_kop = { key: dict() for key in self.train_kop.keys() }
        
        for key in self.train_kop.keys():
            for rr in range(self.train_kop[key].shape[1]):

                if interp == 'CubicSpline':
                    self.interpolants_kop[key][rr] = CubicSpline(train_params, self.train_kop[key][:, rr])   
                else:
                    assert (interp == 'linear') or (interp == 'cubic'), "Interpolation method not recognized."
                    self.interpolants_kop[key][rr] = interp1d(train_params, self.train_kop[key][:, rr], kind=interp)  

    def predict_kop(self, param):

        model = {
                    key: np.asarray([self.interpolants_kop[key][rr](param) for rr in range(self.train_kop[key].shape[1])]) 
                    for key in self.train_kop.keys() 
                }

        model['modes'] = model['modes'].reshape(self.rank, self.rank)

        return model