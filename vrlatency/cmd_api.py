import click
import vrlatency as vrl
from vrlatency.analysis import perc_range, read_csv, transform_display_df, shift_by_sse, plot_display_figures
import matplotlib.pyplot as plt
import seaborn as sns
import time
import sys
import functools
import traceback
from os import path


def get_rigid_body(rigid_body):
    try:
        import natnetclient as natnet
        client = natnet.NatClient()
        try:
            led = client.rigid_bodies[rigid_body]
        except KeyError:
            raise KeyError("No Motive Rigid Body detected named '{}'.".format(rigid_body))
        if led.position is None:
            raise IOError("Motive is not sending rigid body positions")
    except ConnectionResetError:
        raise ConnectionResetError("Cannot detect Tracking Client.  Is your tracking system sending data?")

    return led


def add_options(options):
    def _add_options(func):
        for option in reversed(options):
            func = option(func)
        return func
    return _add_options


def simplify_exception_output(verbose=True, levels=3):
    def decorator(fun):
        @functools.wraps(fun)
        def wrapper(*args, **kwargs):
            try:
                return fun(*args, **kwargs)
            except Exception as exc:
                err_fmt_str = "{}\r\n".format(exc)
                if verbose:
                    exc_type, exc_obj, exc_tb = sys.exc_info()
                    tb = exc_tb
                    for _ in range(levels):
                        tb = tb.tb_next
                        if tb is None:
                            break
                        lineno, filename = tb.tb_lineno, path.basename(tb.tb_frame.f_code.co_filename)
                        err_fmt_str += '\t--(line {} in {})\r\n'.format(lineno, filename)
                click.ClickException(err_fmt_str).show()
                sys.exit()
        return wrapper
    return decorator


common_options = [
    click.option('--port', default='COM9', help="Port that Arduino board is connected to"),
    click.option('--baudrate', default=250000, help="Serial communication baudrate"),
    click.option('--trials', default=20, help="Number of trials for measurement"),
    click.option('--interval', default=.05, help="Time duration that a stimulus is shown in a trial, in seconds."),
    click.option('--jitter/--no-jitter', default=True, help="Whether to add a randomized delay to the onset of stimulus presentation."),
    # TODO: Add 'output' option here.
]


@click.group()
def cli():
    pass


@cli.command()
# @simplify_exception_output(verbose=True)
@add_options(common_options)
@click.option('--stimsize', default=10, help="Size of light stimulus projected onscreen.")
@click.option('--delay', default=.03, help="start delay length (secs) of trial to wait for stimulus to turn off")
@click.option('--screen', default=0, help="Monitor number to display stimulus on.")
@click.option('--allmodes/--singlemode', default=False, help="Whether to run experiment repeatedly, for all screen modes.")
@click.option('--output', default='.', type=click.Path(exists=True, file_okay=False, dir_okay=True, writable=True))
@click.option('--nsamples', default=200, type=int)
def display(port, baudrate, trials, stimsize, delay, screen, interval, jitter, allmodes, output, nsamples):

    arduino = vrl.Arduino.from_experiment_type(experiment_type='Display', port=port, baudrate=baudrate, nsamples=nsamples)

    stim = vrl.Stimulus(size=stimsize, color=(255, 255, 255))
    on_width = [interval, interval * 2] if jitter else interval

    monitor = vrl.screens[screen]
    original_mode = monitor.get_mode()
    modes = monitor.get_modes() if allmodes else [original_mode]

    for mode in modes:
        if not arduino.is_connected:
            arduino.connect()

        if allmodes:
            monitor.set_mode(mode)
            time.sleep(10)

        exp = vrl.DisplayExperiment(arduino=arduino,
                                    trials=trials, fullscreen=True, on_width=on_width,
                                    trial_delay=delay, screen_ind=screen, stim=stim)
        exp.run()
        exp.save(filename=path.join(output, exp.filename))

        df = read_csv(path=path.join(output, exp.filename))  # TODO: fix inconsistent arg name! path or filename!?
        session_name = exp.filename.split('.')[0]
        df_transformed = transform_display_df(df, session=session_name, thresh=.75)
        df_clustered = df_transformed[df_transformed.Cluster == 0].copy()
        df_shifted = shift_by_sse(df_clustered)
        plot_display_figures(df_shifted)
        plt.show()


@cli.command()
@simplify_exception_output(verbose=True)
@add_options(common_options)
@click.option('--rigid_body', default='LED', help="Name of rigid body from tracker that represents the arduino's LEDs.")
def tracking(port, baudrate, trials, interval, jitter, rigid_body):

    led = get_rigid_body(rigid_body)

    arduino = vrl.Arduino.from_experiment_type(experiment_type='Tracking', port=port, baudrate=baudrate)
    on_width = [interval, interval * 2] if jitter else interval
    exp = vrl.TrackingExperiment(arduino=arduino, trials=trials, fullscreen=True, on_width=on_width, rigid_body=led)
    exp.run()
    exp.save()


@cli.command()
@add_options(common_options)
@click.option('--stimdistance', default=.01, help="Percent of screen width to move stimulus in Total experiment.")
@click.option('--stimsize', default=10, help="Size of light stimulus projected onscreen.")
@click.option('--screen', default=0, help="Monitor number to display stimulus on.")
@click.option('--rigid_body', default='LED', help="Name of rigid body from tracker that represents the arduino's LEDs.")
@click.option('--allmodes/--singlemode', default=False, help="Whether to run experiment repeatedly, for all screen modes.")
def total(port, baudrate, trials, stimdistance, stimsize, screen, interval, jitter, rigid_body, allmodes):

    led = get_rigid_body(rigid_body)

    stim = vrl.Stimulus(size=stimsize)
    arduino = vrl.Arduino.from_experiment_type(experiment_type='Total', port=port, baudrate=baudrate)
    on_width = [interval, interval * 2] if jitter else interval
    exp = vrl.TotalExperiment(arduino=arduino, trials=trials, fullscreen=True, on_width=on_width, screen_ind=screen, stim=stim, rigid_body=led, stim_distance=stimdistance)
    exp.run()
    exp.save()


if __name__ == "__main__":
    cli()
