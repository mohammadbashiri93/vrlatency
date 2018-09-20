import serial
import serial.tools.list_ports
from struct import unpack, pack, error
from io import BytesIO
import time

INPUT_BUFFER_MAXSIZE = 1000000  # TODO: Find better estimate of max buffer size.


class Arduino(object):
    """ Handles attributes and methods related to stimulation/recording device

    Attributes:
        - port:
        - baudrate:
        - packet_fmt:
        - packet_size:
        - n_points:
        - channel:
    """
    options = {'Display': dict(packet_fmt='HH', packet_size=4, exp_char='D'),
               'Total': dict(packet_fmt='3H?', packet_size=7, exp_char='S'),
               'Tracking': dict(packet_fmt='?', packet_size=1, exp_char='T'),
               }

    def __init__(self, port, baudrate, packet_fmt, packet_size, exp_char, nsamples=200):
        """
        Interfaces and Connects to an arduino running the VRLatency programs.

        Arguments:
            - port (str): The port the arduino is conected on (ex: 'COM8')
            - baudrate (int): The baud rate for the arduino connection.
        """

        if nsamples * packet_size > INPUT_BUFFER_MAXSIZE:
            raise ValueError("too many samples are requested for the network's input buffer to handle by PySerial.  Lower nsamples")

        self.port = port
        self.baudrate = baudrate
        self.packet_fmt = packet_fmt
        self.packet_size = packet_size
        self.exp_char = exp_char
        self.nsamples = nsamples
        self.channel = serial.Serial(self.port, baudrate=self.baudrate, timeout=2.)

        self.channel.readline()
        self.channel.read_all()

    @classmethod
    def from_experiment_type(cls, experiment_type, *args, **kwargs):
        """Auto-gen Arduino params from experiment_type ('Tracking', 'Display', or 'Total')"""
        options = cls.options[experiment_type].copy()
        options.update(kwargs)
        return cls(*args, **options)

    @staticmethod
    def find_all():
        """ Displays the connected arduino devices to the machine"""
        ports = list(serial.tools.list_ports.comports())
        for p in ports:
            if ('Arduino' or 'arduino') in str(p):
                print(p)

    def connect(self):
        """Connect the device."""
        self.channel.open()

    def disconnect(self):
        """Disconnect the device"""
        self.channel.close()

    @property
    def is_connected(self):
        "Returns True if arduino is connected, and False otherwise."
        return self.channel.isOpen()

    def read(self):
        """Reads data packets sent over serial channel by arduino

        Returns:
            - list: data recorded by arduino
        """

        while self.channel.in_waiting < self.nsamples * self.packet_size:
            time.sleep(.001)

        packets = BytesIO(self.channel.read_all())
        dd = []
        while True:
            packet = packets.read(self.packet_size)
            if packet:
                d = unpack('<' + self.packet_fmt, packet)
                dd.append(d)
            else:
                break

        return dd

    def write(self, msg, nsamples=None):
        """ Write to arduino over serial channel

        Arguments:
            - msg (str): message to be sent to arduino
        """

        nsamples = nsamples if nsamples else self.nsamples
        packet = pack('<cH', bytes(msg, 'utf-8'), nsamples)
        self.channel.write(packet)

    def init_next_trial(self):
        """ Sends a message to aruino to signal start of a trial"""
        self.write(self.exp_char)


    # TODO: Add pinging to Arduino code (added to total, but not others!)
    def ping(self):
        """Checks connection.

        Returns:
            - bool: True if arduino is connected, False otherwise
        """
        self.channel.readline()
        self.channel.write(bytes('R', 'utf-8'))
        packet = self.channel.read(10)
        response = packet.decode(encoding='utf-8')
        return True if response == 'yes' else False
