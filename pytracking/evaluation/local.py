from pytracking.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_path = ''
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.lasot_path = '/home/wyn/myData/01-hyperspectral/challenge2023/datasets/validation/'
    settings.network_path = '/home/wyn/myStudy/2302-DAT/TransT-wyn/pytracking/networks/'    # Where tracking networks are stored.
    settings.nfs_path = ''
    settings.otb_path = ''
    settings.result_plot_path = '/home/wyn/myStudy/2302-DAT/TransT-wyn/pytracking/result_plots/'
    settings.results_path = '/home/wyn/myStudy/2302-DAT/TransT-wyn/pytracking/tracking_results/'    # Where to store tracking results
    settings.segmentation_path = '/home/wyn/myStudy/2302-DAT/TransT-wyn/pytracking/segmentation_results/'
    settings.tn_packed_results_path = ''
    settings.tpl_path = ''
    settings.trackingnet_path = ''
    settings.uav_path = ''
    settings.vot_path = ''
    settings.youtubevos_dir = ''

    return settings

