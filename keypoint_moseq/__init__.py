# use double-precision by default
from jax import config
config.update("jax_enable_x64", True)

# simple warning formatting
import warnings
warnings.formatwarning = lambda msg, *a: str(msg)

from .util import *

from .io import *

from .widgets import *

from .calibration import (
    noise_calibration,
)

from .fitting import (
    revert, 
    fit_model, 
    apply_model, 
    extract_results,
    resume_fitting, 
    update_hypparams,
)

from .viz import (
    plot_pcs, 
    plot_scree, 
    plot_progress, 
    plot_syllable_frequencies,
    plot_duration_distribution,
    generate_crowd_movies, 
    generate_grid_movies,
    generate_trajectory_plots,
)

from .analysis import (
    compute_moseq_df,
    compute_stats_df, 
    plot_fingerprint, 
    plot_syll_stats_with_sem,
    get_group_trans_mats,
    changepoint_analysis,
    generate_transition_matrices,
    visualize_transition_bigram,
    plot_transition_graph_group,
    plot_transition_graph_difference,
    get_behavioral_distance,
    plot_dendrogram,
    interactive_group_setting,
    label_syllables
)

from jax_moseq.models.keypoint_slds import (
    fit_pca, 
    init_model
)
from . import _version
__version__ = _version.get_versions()['version']
