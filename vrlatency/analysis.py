import numpy as np
import pandas as pd
from io import StringIO
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm, colors


def read_params(path):
    """ Returns the parameters of experiment data file"""
    with open(path, "r") as f:
        header, body = f.read().split('\n\n')
    return dict([param.split(': ') for param in header.split('\n')])


def read_csv(path):
    """ Returns the data of experiment data file"""
    with open(path, "r") as f:
        header, body = f.read().split('\n\n')
    return pd.read_csv(StringIO(body))


def perc_range(x, perc):
    return perc * np.ptp(x) + np.min(x)


def get_display_latencies(df, thresh=.75):
    latencies = []
    sensorf, timef, trialf = df[['SensorBrightness', 'Time', 'Trial']].values.T
    threshf = perc_range(sensorf, thresh)
    trial_range = np.arange(df.Trial.min(), df.Trial.max()+1)
    for trial in trial_range:
        is_trial = trialf == trial
        sensor = sensorf[is_trial]
        time = timef[is_trial]
        off_idx = np.where(sensor < threshf)[0][0]

        try:
            detect_idx = np.where(sensor[off_idx:] > threshf)[0][0]
            latency = time[detect_idx + off_idx] - time[0]
            latencies.append(latency)
        except IndexError:
            latencies.append(np.nan)

    latencies = pd.Series(data=latencies, name='DisplayLatency', index=trial_range)
    latencies.index.name = 'Trial'
    return latencies


def get_tracking_latencies(df):
    """ Returns the latency values for each trial of a Tracking Experiment"""
    def detect_latency(df, thresh):
        diff = np.diff(df.RigidBody_Position > thresh)
        idx = np.where(diff != 0)[0][0]
        return df.Time.iloc[idx] - df.Time.iloc[0]

    latencies = df.groupby('Trial').apply(detect_latency, thresh=df.RigidBody_Position.mean())
    latencies.name = 'TrackingLatency'
    return latencies


def get_total_latencies(df):
    """ Returns the latency values for each trial of a Total Experiment"""

    data = df.copy()

    # Make columns with Sensor/LED values used for each trial
    sensors = {False: 'LeftSensorBrightness', True: 'RightSensorBrightness'}
    data['Sensor'] = data.apply(lambda row: row[sensors[row['LED_State']]], axis=1)

    thresh = data[['LeftSensorBrightness', 'RightSensorBrightness']].values.mean()

    # Apply trial-based time series analysis
    trials = data.groupby('Trial')
    latencies = trials.apply(lambda df: (df.Time.iloc[[np.where(df.Sensor > thresh)[0][0]]] - df.Time.iloc[0]).values[0])
    latencies.name = 'Total Trial Latency (us)'

    # Return dataframe of latencies (Trials x (Group, Latency)
    return latencies


def get_transition_samplenum(session):
    transition_samples = []
    for _, trial in session.groupby('Trial'):
        try:
            transition_sample = trial[trial.TrialTransitionTime == 0].Sample.values[0]
        except:
            transition_sample = np.nan
        transition_samples.append(transition_sample)
    return transition_samples


def read_display_df(filename):
    """Return dataframe object needed for the analysis"""
    session = filename.split('.')[0]
    df = read_csv(filename)
    df['Time'] /= 1000

    df['TrialTime'] = df.groupby('Trial').Time.apply(lambda x: x - x.min())
    df['Sample'] = df.groupby('Trial').cumcount()
    df['Session'] = session
    df['Session'] = pd.Categorical(df['Session'])
    df = df.reindex(['Session', 'Trial', 'Sample', 'Time', 'TrialTime', 'SensorBrightness'], axis=1)

    return df


def compute_sse(x1, x2, win=100):
    x1_mat = np.ndarray(buffer=x1, shape=(len(x1)-win, win), strides=(8, 8), dtype=x1.dtype)  # Rolling backwards
    return np.sum((x1_mat.T - x2[win//2:-win//2]) ** 2, axis=1)


def find_global_minimum(x):
    dx, ddx = np.diff(x), np.diff(x, 2)
    is_zerocrossing = (dx[1:] * dx[:-1]) < 0
    is_positive_slope = ddx > 0
    is_local_minimum = is_zerocrossing & is_positive_slope

    local_minimum_indices = np.where(is_local_minimum)[0] + 1
    global_minimum_indices = local_minimum_indices[np.argmin(x[local_minimum_indices])]
    global_minimum_index = int(global_minimum_indices)
    return global_minimum_index


def shift_by_sse(dd):

    sampling_rate = np.diff(dd.TrialTime.values[:2])[0]
    query = '(-5 < TrialTransitionTime) & (TrialTransitionTime < 5)'
    dd2 = dd.query(query)

    ref_trial = dd2[dd2.DisplayLatency == dd2.DisplayLatency.min()]  # Min latency used as reference
    ref_sensor = ref_trial['SensorBrightness'].values

    winsize = 30
    for trialnum, trial in dd2.groupby('Trial'):
        test_sensor = trial['SensorBrightness'].values
        residuals = compute_sse(test_sensor, ref_sensor, win=winsize)
        minimum = find_global_minimum(residuals)
        offset = minimum - winsize // 2
        dd.loc[dd.Trial == trialnum, 'TrialTransitionTime'] -= offset * sampling_rate

    return dd


def transform_display_dataframe(df, thresh=.75):

    latencies = get_display_latencies(df, thresh=thresh).to_frame().reset_index()
    dfl = pd.merge(df, latencies, on='Trial')
    dfl['TrialTransitionTime'] = dfl['TrialTime'] - dfl['DisplayLatency']

    return dfl


def get_display_df(filename):

    df = read_display_df(filename)
    dfl = transform_display_dataframe(df)
    return shift_by_sse(dfl.copy())



def plot_display_brightness_over_session(dd, ax=None):
    """"""
    ax = ax if ax else plt.gcf()

    my_cmap = cm.gray_r
    my_cmap.set_bad(color='w')

    nsamples_per_trial = dd.groupby('Trial')['DisplayLatency'].agg(len).min()
    H, xedges, yedges = np.histogram2d(dd.TrialTime, dd.SensorBrightness, bins=(nsamples_per_trial, 200))
    H = H.T
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    ax.imshow(H, interpolation='nearest', origin='low', cmap=my_cmap, aspect='auto',
               extent=extent, norm=colors.LogNorm())

    mean_latency = dd.groupby('Trial').DisplayLatency.mean().mean()
    for trialnum, trial in dd.groupby('Trial'):
        ax.plot(trial.TrialTransitionTime + mean_latency, trial.SensorBrightness, c='r', linewidth=1, alpha=.01)

    thresh = .75  # TODO: this must be an input or computed differently
    ax.hlines([perc_range(dd['SensorBrightness'], thresh)], *ax.get_xlim(), 'b', label='Threshold', linewidth=2,
               linestyle='dotted')

    return ax


def plot_display_brightness_distribution(sensor_brightness, ax=None):
    ax = ax if ax else plt.gcf()
    sns.distplot(sensor_brightness, hist_kws={'color': 'k'}, kde_kws={'alpha': 0}, vertical=True, ax=ax)
    return ax


def plot_display_latency_over_session(trials, latencies, ax=None):
    """Makes a line plot of latencies over the course of a session."""
    ax = ax if ax else plt.gcf()

    ax.plot(trials, latencies, c='k', linewidth=.5)
    ax.set(xlabel='Trial number', ylabel='Latency (ms)')

    return ax


def plot_display_latency_distribution(latencies, ax=None):

    ax = ax if ax else plt.gcf()
    sns.distplot(latencies[np.isnan(latencies) == False],
                 hist=True, color="k", kde_kws={"linewidth": 3, "alpha": 1}, vertical=True)

    return ax


def plot_display_figures(filename):
    """Returns a figure with all info concerning display experiment latencies."""

    dd = get_display_df(filename)
    session = dd.Session.values[0]


    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 5), gridspec_kw={'width_ratios': [3, 1]}, sharey=True)
    fig.tight_layout(w_pad=0)


    plot_display_brightness_over_session(dd, ax=ax1)
    plot_display_brightness_distribution(sensor_brightness=dd['SensorBrightness'], ax=ax2)
    ax1.set_ylim(*ax2.get_ylim())
    ax2.set(xticklabels='', yticklabels='')
    ax1.set(xlabel='Time (ms)', ylabel='Brightness')

    plot_display_latency_over_session(trials=dd['Trial'], latencies=dd['DisplayLatency'], ax=ax3)
    plot_display_latency_distribution(latencies=dd['DisplayLatency'], ax=ax4)
    ax3.set_ylim(*ax4.get_ylim())
    ax4.set(xticklabels='', yticklabels='')
    ax3.set(xlabel='Trial', ylabel='Latency (ms)')

    fig.suptitle(session)
    fig.subplots_adjust(top=.9)

    return fig
