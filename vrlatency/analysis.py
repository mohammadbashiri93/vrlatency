import numpy as np
import pandas as pd
from io import StringIO
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt


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


def display_brightness_figure(filename, ax1=None, ax2=None):

    df = get_display_dataframe(filename)
    session = df.Session.values[0]
    latencies = df.groupby('Session').apply(get_display_latencies, thresh=.75).unstack()
    latencies.name = 'DisplayLatency'
    latencies = latencies.reset_index()

    dfl = pd.merge(df, latencies, on=['Session', 'Trial'])
    dfl['TrialTransitionTime'] = dfl['TrialTime'] - dfl['DisplayLatency']

    hh = np.array(get_transition_samplenum(dfl))
    hh = hh[~np.isnan(hh)]
    hh = hh.astype(int)

    if (not ax1) and (not ax2):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), gridspec_kw={'width_ratios': [2, 1]}, sharey=True)

    ax1.scatter(dfl.TrialTime, dfl.SensorBrightness, c='k', s=.2, alpha=.2)

    for test_trialnum in tqdm(range(dfl.Trial.min(), dfl.Trial.max(), 1)):

        test_trial = dfl[dfl.Trial == test_trialnum]
        fastest_transtion_sample = hh.mean().round().astype(int)
        try:
            test_transition_sample = test_trial[test_trial.TrialTransitionTime == 0].Sample.values[0]
            offset = test_transition_sample - fastest_transtion_sample
        except IndexError:
            offset = -1

        if offset >= 0:
            ax1.scatter(test_trial.TrialTime.values[offset:] +
                        (test_trial.TrialTime.values.min() - test_trial.TrialTime.values[offset]),
                        test_trial.SensorBrightness.values[offset:], c='r', s=.1, alpha=.5)

    thresh = 0.75
    ax1.hlines([perc_range(dfl['SensorBrightness'], thresh)], *ax1.get_xlim(), 'b', label='Threshold', linewidth=2,
               linestyle='dotted')

    sns.distplot(dfl['SensorBrightness'].values, ax=ax2, vertical=True, hist_kws={'color': 'k'}, kde_kws={'alpha': 0})
    ax2.set(xticklabels='')
    if ax1 and ax2:
        ax2.set_ylim(*ax1.get_ylim())

    ax1.set(xlabel='Trial Time (ms)', ylabel='Brightness')

    plt.suptitle(session, y=1.02)
    plt.tight_layout(w_pad=0)
    # plt.show()


def display_latency_figure(filename, ax1=None, ax2=None):

    df = get_display_dataframe(filename)
    session = df.Session.values[0]
    latencies = df.groupby('Session').apply(get_display_latencies, thresh=.75).unstack()
    latencies.name = 'DisplayLatency'
    latencies = latencies.reset_index()

    dl = latencies[latencies['Session'] == session]

    if (not ax1) and (not ax2):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), gridspec_kw={'width_ratios': [2, 1]}, sharey=True)

    ax1.plot(dl['Trial'], dl['DisplayLatency'], c='k', linewidth=.5)
    sns.distplot(dl['DisplayLatency'].dropna().values, hist=False, color="k", kde_kws={"linewidth": 3, "alpha": 1},
                 ax=ax2, vertical=True)
    ax1.set(xlabel='Trial number', ylabel='Latency (ms)')

    if ax1 and ax2:
        ax2.set_ylim(*ax1.get_ylim())

    plt.suptitle(session, y=1.02)
    plt.tight_layout(w_pad=0)
    # plt.show()


def display_figures(filename):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 5), gridspec_kw={'width_ratios': [3, 1]})
    display_brightness_figure(filename, ax1=ax1, ax2=ax2)
    display_latency_figure(filename, ax1=ax3, ax2=ax4)
    plt.show()
