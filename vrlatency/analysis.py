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


def get_display_dataframe(filename):
    """Return dataframe object needed for the analysis"""
    session = filename.split('.')[0]
    df = read_csv(filename)
    df['Session'] = session
    df['Time'] /= 1000
    trials = df.groupby(['Session', 'Trial'])
    df['TrialTime'] = trials.Time.apply(lambda x: x - x.min())
    df['Sample'] = trials.cumcount()
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


def display_brightness_figure(filename, ax1=None, ax2=None):

    want_to_draw = not bool(ax1 and ax2)
    my_cmap = cm.gray_r
    my_cmap.set_bad(color='w')

    df = get_display_dataframe(filename)
    session = df.Session.values[0]
    thresh = .75
    latencies = df.groupby('Session').apply(get_display_latencies, thresh=thresh).unstack()
    latencies.name = 'DisplayLatency'
    latencies = latencies.reset_index()

    dfl = pd.merge(df, latencies, on=['Session', 'Trial'])
    dfl['TrialTransitionTime'] = dfl['TrialTime'] - dfl['DisplayLatency']

    dd = shift_by_sse(dfl.copy())

    if want_to_draw:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), gridspec_kw={'width_ratios': [5, 2]}, sharey=True)

    # plot all the trials at their actual time
    nsamples_per_trial = dd.groupby('Trial')['DisplayLatency'].agg(len).min()
    H, xedges, yedges = np.histogram2d(dd.TrialTime, dd.SensorBrightness, bins=(nsamples_per_trial, 200))
    H = H.T
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    ax1.imshow(H, interpolation='nearest', origin='low', cmap=my_cmap, aspect='auto',
               extent=extent, norm=colors.LogNorm())

    # plot the shifted trials
    mean_latency = dd.groupby('Trial').DisplayLatency.mean().mean()
    for trialnum, trial in dd.groupby('Trial'):
        ax1.plot(trial.TrialTransitionTime + mean_latency, trial.SensorBrightness, c='r', linewidth=1, alpha=.01)

    # plot the threshold
    ax1.hlines([perc_range(dd['SensorBrightness'], thresh)], *ax1.get_xlim(), 'b',
               label='Threshold', linewidth=2, linestyle='dotted')

    # plot the histogram of the brightness values
    sns.distplot(dd['SensorBrightness'].values, ax=ax2, vertical=True, hist_kws={'color': 'k'}, kde_kws={'alpha': 0})
    ax1.set_ylim(*ax2.get_ylim())
    ax2.set_xticklabels(''), ax2.set_yticklabels('')
    ax1.set(xlabel='Time (ms)', ylabel='Brightness')
    plt.suptitle(session)
    plt.tight_layout(w_pad=0)

    if want_to_draw:
        plt.subplots_adjust(top=.9)
        plt.show()


def display_latency_figure(filename, ax1=None, ax2=None):

    want_to_draw = not bool(ax1 and ax2)

    df = get_display_dataframe(filename)
    session = df.Session.values[0]
    latencies = df.groupby('Session').apply(get_display_latencies, thresh=.75).unstack()
    latencies.name = 'DisplayLatency'
    latencies = latencies.reset_index()

    dl = latencies[latencies['Session'] == session]

    if want_to_draw:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), gridspec_kw={'width_ratios': [3, 1]}, sharey=True)

    ax1.plot(dl['Trial'], dl['DisplayLatency'], c='k', linewidth=.5)
    sns.distplot(dl['DisplayLatency'].dropna().values, hist=False, color="k", kde_kws={"linewidth": 3, "alpha": 1},
                 ax=ax2, vertical=True)
    ax1.set_ylim(*ax2.get_ylim())
    ax2.set_xticklabels(''), ax2.set_yticklabels('')
    ax1.set(xlabel='Trial number', ylabel='Latency (ms)')
    plt.tight_layout(w_pad=0)

    if want_to_draw:
        plt.suptitle(session)
        plt.subplots_adjust(top=.9)
        plt.show()


def display_figures(filename):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(7, 6), gridspec_kw={'width_ratios': [5, 2]})
    display_brightness_figure(filename, ax1=ax1, ax2=ax2)
    display_latency_figure(filename, ax1=ax3, ax2=ax4)
    plt.subplots_adjust(top=.9)
    plt.show()
