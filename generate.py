import argparse
from math import fabs, ceil
import numpy as np
from scipy import fftpack as fft
from scipy.io import wavfile as wav


def generate_cov_matrix(size):
    '''
    Generate a random, positive semi-definite matrix with shape (size, size).
    :param size:
    :return:
    '''
    M = np.random.rand(size, size)
    M = (M + M.T) / 2
    return M * M.T


def next_sample(prev_sample, cov):
    '''
    Produces a new frequency-domain sample conditioned on the previous sample.
    :param variance:
    :param prev_sample: a complex-valued numpy array with shape (2, channels, fft_size)
    :return: a complex-valued numpy array with shape (2, channels, fft_size)
    '''
    _, channels, size = prev_sample.shape
    zero_mean = np.zeros(size)
    step = np.random.multivariate_normal(zero_mean, cov, (2, channels))
    return prev_sample + step


def to_temporal_domain(sample):
    '''
    Takes a frequency-domain sample and maps it to the real-only temporal domain (via ifft).
    :param sample:
    :return:
    '''
    complex_sample = sample[0] + 1.0j * sample[1]
    temporal_sample = fft.ifft(complex_sample).real

    return temporal_sample


def main(args):

    size = 2 ** args.fft_order
    if args.random_cov:
        cov = generate_cov_matrix(size)
    else:
        diagonal = np.full(size, args.variance)
        cov = np.diag(diagonal)
        if args.variance <= 0:
            cov = cov.T * cov


    sample = np.zeros((2, args.channels, 2 ** args.fft_order))
    uninterpolated_samples = np.zeros((args.channels, 0))

    for _ in range(args.steps):
        sample = next_sample(sample, cov)
        audio = to_temporal_domain(sample)
        # normalize samples
        coeff = max(fabs(audio.max()), fabs(audio.min()))
        audio = audio / coeff
        uninterpolated_samples = np.concatenate((uninterpolated_samples, audio), axis=1)

    audio_time = args.steps * args.step_time / 1000.0
    xp = np.linspace(0.0, audio_time, uninterpolated_samples.shape[1])

    num_samples = ceil(args.sample_rate * audio_time)
    x = np.linspace(0.0, audio_time, num_samples)

    audio = []
    for channel in uninterpolated_samples:
        interpolated_channel = np.interp(x, xp, channel)
        audio.append(interpolated_channel)

    audio = np.array(audio)
    wav.write(args.outfile, args.sample_rate, audio.T)


if __name__ == '__main__':
    # Options:
    #    - output file name (default 'out.wav') [args.outfile]
    #    - sample rate in Hz (default 44100) [args.sample_rate]
    #    - channels (default 2) [args.channels]
    #    - number of fft chunks (default 20) [args.steps]
    #    - milliseconds per chunk (default 250) [args.step_length]
    #    - fft order (exponent for a power of 2) (default 12) [args.fft_order]
    #    - variance for multivariate normal dist (default 1) [args.variance]
    #    - use a random covariance matrix (default False) [args.random_cov] todo
    #    - center the gaussian distribution around a specific frequency (default False) [args.freq] todo

    parser = argparse.ArgumentParser(description='Generates audio by doing a multivariate Gaussian random walk '
                                                 'through Fourier space.')
    parser.add_argument('-f', '--outfile', metavar='FILE', type=str, default='out.wav',
                        help='The file that the generated audio is written to.')
    parser.add_argument('--sample-rate', metavar='RATE', type=int, default=44100,
                        help='The sample rate (in Hz) of the generated audio. (Default: 44.1 kHz)')
    parser.add_argument('-c', '--channels', metavar='NUM_CHANNELS', default=2, type=int,
                        help='The number of channels in the generated audio. (Default: 2)')
    parser.add_argument('-s', '--steps', metavar='STEPS', default=20, type=int,
                        help='The number of steps to take in the Gaussian random walk.')
    parser.add_argument('--step-time', metavar='STEP_LENGTH', default=250, type=int,
                        help='The amount of audio time (in ms) that each random walk step corresponds to'
                             ' (default 250ms).')
    parser.add_argument('--fft-order', metavar='ORDER', default=12, type=int,
                        help='Frequency domain samples have size 2**ORDER. (Default: 12)')
    parser.add_argument('--variance', metavar='VARIANCE', type=float, default=1.0,
                        help='The variance to use when sampling from the frequency domain.')
    parser.add_argument('--random-cov', action='store_true', default=False,
                        help='Use a random covariance matrix for sampling steps.')
    args = parser.parse_args()
    main(args)