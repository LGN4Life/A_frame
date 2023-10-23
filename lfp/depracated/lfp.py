from scipy import signal

import numpy as np
from scipy.interpolate import interp1d


class lfp:
    def __init__(self, Y, Fs):
        self.Y = Y
        self.Fs = Fs
        self.interval = 1 / Fs
        self.length = len(Y)
        self.X = self.interval * np.arange(self.length)
        self.filter = None
        # self.filter.a = None

    def subsample(self, target_frequency):
        if self.Fs < 20000:
            R = int(np.ceil(self.Fs / target_frequency))
        else:
            R = int(np.floor(self.Fs / target_frequency))

        # Remove excess samples from the end
        new_length = self.length - self.length % R
        self.Y = self.Y[:new_length].reshape(-1, R).mean(axis=1)

        self.interval *= R
        self.Fs = 1 / self.interval
        self.length = len(self.Y)
        self.X = self.interval * np.arange(self.length)

        new_x = np.arange(self.X[0], self.X[-1], 1 / target_frequency)
        interp_func = interp1d(self.X, self.Y, kind='linear', fill_value='extrapolate')
        self.Y = interp_func(new_x)
        self.X = new_x
        self.interval = 1 / target_frequency
        self.Fs = 1 / self.interval
        self.length = len(self.Y)
        return self


class filter:
    def __init__(self, params):
        self.filter_order = params['filter_order']
        self.stopband_attenuation = params['stopband_attenuation']
        self.cutoff_frequency = params['cutoff_frequency']
        self.nyquist_frequency = 0.5 * params['fs']
        self.Fs = params['fs']
        self.filter = np.nan

    def cheby2_lowpass(self, s):
        # Create the filter coefficients using a Chebyshev type II filter
        self.filter = signal.cheby2(self.filter_order, self.stopband_attenuation,
                                    self.cutoff_frequency / self.nyquist_frequency, 'low',
                                    self.Fs, output='sos')

        s = signal.sosfilt(self.filter, s)

        return self, s
