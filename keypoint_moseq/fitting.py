import os
import numpy as np
import tqdm
import jax
import warnings
from textwrap import fill
from datetime import datetime

from keypoint_moseq.viz import plot_progress
from keypoint_moseq.io import save_hdf5, _get_path
from jax_moseq.models.keypoint_slds import resample_model, init_model
from jax_moseq.utils import check_for_nans, unbatch, device_put_as_scalar


class StopResampling(Exception):
    pass


def _wrapped_resample(data, model, pbar=None, **resample_options):
    try: 
        model = resample_model(data, **model, **resample_options)
    except KeyboardInterrupt: 
        print('Early termination of fitting: user interruption')
        raise StopResampling()

    any_nans, nan_info, messages = check_for_nans(model)
    
    if any_nans:
        if pbar is not None: 
            pbar.close()
        warning_text = ['\nEarly termination of fitting: NaNs encountered']
        for msg in messages: 
            warning_text.append('  - {}'.format(msg))
        warning_text.append('\nFor additional information, see https://keypoint-moseq.readthedocs.io/en/latest/troubleshooting.html#nans-during-fitting')
        warnings.warn('\n'.join(warning_text))
        raise StopResampling()
    
    return model


def _set_parallel_flag(parallel_message_passing):
    if parallel_message_passing == 'force':
        parallel_message_passing = True
    elif parallel_message_passing is None:
        parallel_message_passing = jax.default_backend() != 'cpu'
    elif parallel_message_passing and jax.default_backend() == 'cpu':
        warnings.warn(fill(
            'Setting parallel_message_passing to True when JAX is CPU-bound can '
            'result in long jit times without speed increase for calculations. '
            '(To suppress this message, set parallel_message_passing="force")'))
    return parallel_message_passing


def fit_model(model,
              data,
              metadata,
              project_dir=None,
              name=None,
              num_iters=50,
              start_iter=0,
              verbose=False,
              ar_only=False,
              save_every_n_iters=10,
              generate_progress_plots=True,
              parallel_message_passing=None,
              **kwargs):
    """
    Fit a model to data.
    
    This method optionally:
        - saves checkpoints of the model and data at regular intervals
          (see :py:func:`jax_moseq.io.save_checkpoint`)
        - plots of the model's progress during fitting (see 
          :py:func:`jax_moseq.viz.plot_progress`)

    Parameters
    ----------
    model : dict
        Model dictionary containing states, parameters, hyperparameters, 
        noise prior, and random seed. 

    data: dict
        Data for model fitting (see :py:func:`keypoint_moseq.io.format_data`).

    metadata: tuple (keys, bounds)
        Recordings and start/end frames for the data (see
        :py:func:`keypoint_moseq.io.format_data`).

    project_dir : str, default=None
        Project directory; required if `save_every_n_iters>0`.

    name : str, default=None
        Name of the model. If None, the model is named using the current
        date and time.

    num_iters : int, default=50
        Number of Gibbs sampling iterations to run.

    start_iter : int, default=0
        Index of the starting iteration, which is non-zero when continuing
        a previous fit.

    verbose : bool, default=True
        If True, print the model's progress during fitting.

    ar_only : bool, default=False
        If True, fit an AR-HMM model using the latent trajectory
        defined by `model['states']['x']` (see 
        :py:func:`jax_moseq.models.arhmm.resample_model`).
        Otherwise fit a full keypoint-SLDS model
        (see :py:func:`jax_moseq.models.keypoint_slds.resample_model`)

    save_every_n_iters : int, default=10
        Save the current model every `save_every_n_iters`. If
        `save_every_n_iters=0`, nothing is saved.

    generate_progress_plots : bool, default=True
        If True, generate plots of the model's progress during fitting.
        Plots are saved to `{project_dir}/{name}/plots/`.

    parallel_message_passing : bool | string, default=None,
        Use parallel implementation of Kalman sampling, which can be faster
        but has a significantly longer jit time. If None, will be set 
        automatically based on the backend (True for GPU, False for CPU).
        A warning will be raised if `parallel_message_passing=True` and
        JAX is CPU-bound. Set to 'force' to skip this check.
        
    Returns
    -------
    model : dict
        Model dictionary containing states, parameters, hyperparameters,
        noise prior, and random seed.

    name : str
        Name of the model.
    """
    if generate_progress_plots and save_every_n_iters==0:
        warnings.warn(fill(
            'The `generate_progress_plots` option requires that '
            '`save_every_n_iters` be greater than 0. Progress plots will '
            'not be generated.'))
        generate_progress_plots = False

    if name is None: 
        name = str(datetime.now().strftime('%Y_%m_%d-%H_%M_%S'))
    
    if save_every_n_iters>0:
        savedir = os.path.join(project_dir,name)
        if not os.path.exists(savedir): os.makedirs(savedir)
        print(fill(f'Outputs will be saved to {savedir}'))

        checkpoint_path = os.path.join(savedir,'checkpoint.h5')
        if not os.path.exists(checkpoint_path):
            save_hdf5(checkpoint_path, {
                'model_snapshots' : {f'{start_iter}': model},
                'metadata' : metadata,
                'data' : data})

    parallel_message_passing = _set_parallel_flag(parallel_message_passing)
    model = device_put_as_scalar(model)

    with tqdm.trange(start_iter, num_iters+1) as pbar:
        for iteration in pbar:
            try: 
                model = _wrapped_resample(
                    data, model, pbar=pbar, ar_only=ar_only, verbose=verbose,
                    parallel_message_passing=parallel_message_passing)
            except StopResampling: break

            if save_every_n_iters>0 and iteration>start_iter:
                if (iteration%save_every_n_iters)==0 or iteration==num_iters:
                    save_hdf5(checkpoint_path, model, f'model_snapshots/{iteration}')
                    if generate_progress_plots:
                        plot_progress(model, data, checkpoint_path, iteration, project_dir, name, savefig=True)

    return model, name
    


def extract_results(model, metadata, project_dir=None, name=None, 
                    save_results=True, path=None): 
    """
    Extract model outputs and [optionally] save them to disk.

    Model outputs are saved to disk as a .h5 file, either at `path`
    if it is specified, or at `{project_dir}/{name}/results.h5` if it is not.
    If a .h5 file with the given path already exists, the outputs will be added
    to it. The results have the following structure::

        results.h5
        ├──recording_name1
        │  ├──syllable      # model state sequence (z), shape=(T,)
        │  ├──latent_state  # model latent state (x), shape=(T,latent_dim)
        │  ├──centroid      # model centroid (v), shape=(T,dim)
        │  └──heading       # model heading (h), shape=(T,)
        ⋮

    Parameters
    ----------
    model : dict
        Model dictionary containing states, parameters, hyperparameters,
        noise prior, and random seed.

    metadata: tuple (keys, bounds)
        Recordings and start/end frames for the data (see
        :py:func:`keypoint_moseq.io.format_data`).

    save_results : bool, default=True
        If True, the model outputs will be saved to disk.
        
    project_dir : str, default=None
        Path to the project directory. Required if `save_results=True`
        and `results_path=None`.

    name : str, default=None
        Name of the model. Required if `save_results=True`
        and `results_path=None`.

    path : str, default=None
        Optional path for saving model outputs.

    Returns
    -------
    results_dict : dict
        Dictionary of model outputs with the same structure as the
        results `.h5` file.
    """
    if save_results: 
        path = _get_path(path, project_dir, name, 'results.h5')

    states = jax.device_get(model['states'])
    keys, bounds = metadata

    # extract syllables; repeat first syllable an extra `nlags` times
    nlags = states['x'].shape[1] - states['z'].shape[1]
    syllables = unbatch(states['z'], keys, bounds+np.array([0,nlags]))
    syllables = {k: np.pad(z[nlags:], (nlags,0), mode='edge') for k,z in syllables.items()}
    
    # extract latent state, centroid, and heading
    latent_state = unbatch(states['x'], keys, bounds)
    centroid = unbatch(states['v'], keys, bounds)
    heading = unbatch(states['h'], keys, bounds)

    results_dict = {
        recording_name : {
            'syllable' : syllables[recording_name],
            'latent_state' : latent_state[recording_name],
            'centroid' : centroid[recording_name],
            'heading' : heading[recording_name]
        } for recording_name in syllables.keys()}
    
    if save_results: 
        save_hdf5(path, results_dict)
        print(fill(f'Saved results to {path}'))
        
    return results_dict


def apply_model(model, data, metadata, project_dir=None, name=None, 
                num_iters=20, ar_only=False, save_results=True, 
                verbose=False, results_path=None, 
                parallel_message_passing=None):
    """
    Apply a model to new data.

    Parameters
    ----------
    model : dict
        Model dictionary containing states, parameters, hyperparameters, 
        noise prior, and random seed. 

    data: dict
        Data for model fitting (see :py:func:`keypoint_moseq.io.format_data`).

    metadata: tuple (keys, bounds)
        Recordings and start/end frames for the data (see
        :py:func:`keypoint_moseq.io.format_data`).

    project_dir : str, default=None
        Path to the project directory. Required if `save_results=True`
        and `results_path=None`.

    name : str, default=None
        Name of the model. Required if `save_results=True`
        and `results_path=None`.

    num_iters : int, default=20
        Number of iterations to run the model. 

    ar_only : bool, default=False
        See :py:func:`keypoint_moseq.fitting.fit_model`.

    save_results : bool, default=True
        If True, the model outputs will be saved to disk (see 
        :py:func:`keypoint_moseq.fitting.extract_results` for
        the output format).

    verbose : bool, default=False
        Whether to print progress updates.

    results_path : str, default=None
        Optional path for saving model outputs.

    parallel_message_passing : bool | string, default=None,
        Use parallel implementation of Kalman sampling, which can be faster
        but has a significantly longer jit time. If None, will be set 
        automatically based on the backend (True for GPU, False for CPU).
        A warning will be raised if `parallel_message_passing=True` and
        JAX is CPU-bound. Set to 'force' to skip this check.
    
    Returns
    -------
    results_dict : dict
        Dictionary of model outputs (for results format, see
        :py:func:`keypoint_moseq.fitting.extract_results`).
    """
    parallel_message_passing = _set_parallel_flag(parallel_message_passing)

    if save_results:
        if results_path is None: 
            assert project_dir is not None and name is not None, fill(
                'The `save_results` option requires either a `results_path` '
                'or the `project_dir` and `name` arguments')
            results_path = os.path.join(project_dir,name,'results.h5')

    model = init_model(
        data=data, params=model['params'], 
        hypparams=model['hypparams'], verbose=verbose)

    with tqdm.trange(num_iters, desc='Applying model') as pbar:
        for iteration in pbar:
            try: 
                model = _wrapped_resample(
                    data, model, pbar=pbar, ar_only=ar_only, 
                    states_only=True, verbose=verbose,
                    parallel_message_passing = parallel_message_passing)
            except StopResampling: 
                break

    return extract_results(
        model, metadata, project_dir, name, save_results, results_path)


    
def update_hypparams(model_dict, **kwargs):
    """
    Edit the hyperparameters of a model.

    Hyperparameters are stored as a nested dictionary in the
    `hypparams` key of the model dictionary. This function
    allows the user to update the hyperparameters of a model
    by passing in keyword arguments with the same name as the
    hyperparameter. The hyperparameter will be updated if it
    is a scalar value.

    Parameters
    ----------
    model_dict : dict
        Model dictionary.

    kwargs : dict
        Keyword arguments mapping hyperparameter names to new values.

    Returns
    -------
    model_dict : dict
        Model dictionary with updated hyperparameters.
    """
    assert 'hypparams' in model_dict, fill(
        'The inputted model/checkpoint does not contain any hyperparams')
    
    not_updated = list(kwargs.keys())
    
    for hypparms_group in model_dict['hypparams']:
        for k,v in kwargs.items():
            if k in model_dict['hypparams'][hypparms_group]:
                old_value = model_dict['hypparams'][hypparms_group][k] 
                if not np.isscalar(old_value): print(fill(
                    f'{k} cannot be updated since it is not a scalar hyperparam'))
                else:
                    if not isinstance(v, type(old_value)): warnings.warn(fill(
                        f'{v} has type {type(v)} which differs from the current '
                        f'value of {k} which has type {type(old_value)}. {v} will '
                        f'will be cast to {type(old_value)}'))
                                     
                    model_dict['hypparams'][hypparms_group][k] = type(old_value)(v)
                    not_updated.remove(k)

    if len(not_updated)>0: warnings.warn(fill(
        f'The following hypparams were not found {not_updated}'))

    return model_dict