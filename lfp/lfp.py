from scipy import signal
import numpy as np
from scipy.interpolate import interp1d
# import matplotlib.pyplot as plt


class lfp:
    def __init__(self, Y, Fs, file_name, channel, X=None):
        self.Y = Y
        self.Fs = Fs
        self.interval = 1 / Fs
        self.length = len(Y)
        self.X = self.interval * np.arange(self.length)
        self.filter = None
        self.file_name = file_name
        self.channel = channel

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
        # when taking slices from OpenEphys files (to line up with SMR files) self.X[0] may
        # not be 0
        self.X =  self.interval * np.arange(self.length) + self.X[0]

        new_x = np.arange(self.X[0], self.X[-1], 1 / target_frequency)
        interp_func = interp1d(self.X, self.Y, kind='linear', fill_value='extrapolate')
        self.Y = interp_func(new_x)
        self.X = new_x
        self.interval = 1 / target_frequency
        self.Fs = 1 / self.interval
        self.length = len(self.Y)
        return self

    def remove_60Hz(self):
    

        #the total length of the recording has to be a mutiple of the window size
        #make this the case
        window_size = 0.6
        recording_duration = self.length * self.interval
        recording_segments =np.floor(recording_duration/window_size)

        segment_length =np.floor(self.Y.shape[0]/recording_segments)
        end_segment = recording_segments*segment_length
        signal = self.Y[0:end_segment.astype(int)]
        signal_x = self.X[0:end_segment.astype(int)]

        
        # fig, axs =  plt.subplots(1,1)
        # axs.plot(signal_x, signal)
        signal = np.reshape(signal,(segment_length.astype(int),recording_segments.astype(int))).T
        
        time = np.arange(segment_length)*1/self.Fs


        f = np.arange(4,100)
        # pre_ps = LFP.custom_fft(signal',f,obj.Fs);
        # pre_signal = signal;
        
        for f_index in range(1,4):
            for r_index in range(1,4):
                CosF = np.cos(time*2*np.pi*60*f_index).T
                SinF = np.sin(time*2*np.pi*60*f_index).T
                LineNoise_Measured = (np.matmul(signal, CosF) + 1j*np.matmul(signal, SinF)) /(signal.shape[1]/2)
                LineNoise_Measured = np.tile(LineNoise_Measured,(signal.shape[1],1)).T 
                LineNoise =time*2*np.pi*60*f_index
                
                LineNoise =np.tile(LineNoise,(signal.shape[0],1))
                
                LineNoise = np.imag(LineNoise_Measured) * np.sin(LineNoise) + \
                    np.real(LineNoise_Measured) * np.cos(LineNoise)
                signal= signal-(LineNoise)
    
            


       

        self.Y[0:end_segment.astype(int)] = np.reshape(signal.T,(1,end_segment.astype(int)))
        self.Y[end_segment.astype(int)+1:] = 0
        # axs.plot(self.X, self.Y)
        # plt.show()
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
