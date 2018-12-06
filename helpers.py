import numpy as np
import csv
import scipy.interpolate as interp


def sensors_map(filename, width, height, pad):
    x, y, _ = get_sensors(filename)
    n_sens = len(x)
    if (width < np.sqrt(n_sens) or height < np.sqrt(n_sens)):
        print("Too small image size!")
        return None
    x = np.round((width - 2 * pad - 1) * (normalise(np.array(x), updown=True) + 1) / 2) + pad
    y = np.round((height - 2 * pad - 1) * (normalise(np.array(y), updown=True) + 1) / 2) + pad
    return np.vstack((x, y)).transpose()

def get_sensors(filename):
    x = []
    y = []
    z = []
    with open(filename) as f:
        reader = csv.reader(f)
        for str in reader:
            sensor = [float(x) for x in str[0].split("\t")]
            x.append(sensor[0])
            y.append(sensor[1])
            z.append(sensor[2])
    return x, y, z

def normalise(eeg, each_row=False, updown=False):
    if (eeg.ndim == 1):
        if (updown):
            maxval = np.max(eeg)
            if (maxval > 0):
                eeg[eeg > 0] = eeg[eeg > 0] / maxval
            minval = np.min(eeg)
            if (minval < 0):
                eeg[eeg < 0] = eeg[eeg < 0] / np.abs(minval)
        else:
            eeg = eeg / np.max(np.abs(eeg))
    else:
        if (each_row):
            for j in range(eeg.shape[0]):
                eeg[j, :] = normalise(eeg[j, :], updown=updown)
        else:
            if (updown):
                maxval = np.max(eeg)
                if (maxval > 0):
                    eeg[eeg > 0] /= maxval
                minval = np.min(eeg)
                if (minval < 0):
                    eeg[eeg < 0] /= np.abs(minval)
            else:
                eeg /= np.max(np.abs(eeg))
    return eeg

def eeg2rgb(eeg, fs, map, width, height, method='cubic', param=1):
    # eeg = butter_bandpass_filter(eeg, 4, 30, fs)
    freqs, spectrum = fourier(eeg, fs)
    teta_start = np.argmax(freqs >= 4)
    teta_end = np.argmax(freqs > 7) - 1
    alpha_start = np.argmax(freqs >= 8)
    alpha_end = np.argmax(freqs > 13) - 1
    beta_start = np.argmax(freqs >= 14)
    beta_end = np.argmax(freqs > 30) - 1
    pix_x = []
    pix_y = []
    samples_red = []
    samples_green = []
    samples_blue = []
    for j in range(eeg.shape[0]):
        pix_x.append(int(map[j][0]))
        pix_y.append(int(map[j][1]))
        samples_red.append(sum(spectrum[j, teta_start:teta_end+1]))
        samples_green.append(sum(spectrum[j, alpha_start:alpha_end+1]))
        samples_blue.append(sum(spectrum[j, beta_start:beta_end+1]))
    if method == 'cubic' or method == 'linear' or method == 'nearest':
        X, Y = np.meshgrid(np.arange(0, width, 1), np.arange(0, height, 1))
        img_red = interp.griddata((pix_y, pix_x), samples_red, (Y, X), method=method)
        img_green = interp.griddata((pix_y, pix_x), samples_green, (Y, X), method=method)
        img_blue = interp.griddata((pix_y, pix_x), samples_blue, (Y, X), method=method)
        img_red[np.isnan(img_red)] = 0
        img_green[np.isnan(img_green)] = 0
        img_blue[np.isnan(img_blue)] = 0
    elif method == 'gauss':
        sigma = param
        ii, jj = np.ogrid[-sigma:sigma+1, -sigma:sigma+1]
        gauss = np.exp(-((ii)**2 + (jj)**2) / (2 * sigma**2))
        img_red = np.zeros((height, width))
        img_green = np.zeros((height, width))
        img_blue = np.zeros((height, width))
        for j in range(len(pix_x)):
            img_red[pix_y[j]-sigma:pix_y[j]+sigma+1, pix_x[j]-sigma:pix_x[j]+sigma+1] = samples_red[j] * gauss
            img_blue[pix_y[j]-sigma:pix_y[j]+sigma+1, pix_x[j]-sigma:pix_x[j]+sigma+1] = samples_blue[j] * gauss
            img_green[pix_y[j]-sigma:pix_y[j]+sigma+1, pix_x[j]-sigma:pix_x[j]+sigma+1] = samples_green[j] * gauss
    elif method == 'no':
        img_red = np.zeros((height, width))
        img_green = np.zeros((height, width))
        img_blue = np.zeros((height, width))
        for j in range(len(pix_x)):
            img_red[pix_y[j], pix_x[j]] = samples_red[j]
            img_green[pix_y[j], pix_x[j]] = samples_green[j]
            img_blue[pix_y[j], pix_x[j]] = samples_blue[j]
    else:
        print("No such method!!")
        quit()
    return [img_red / np.max(img_red), img_green / np.max(img_green), img_blue / np.max(img_blue)]

def fourier(f, fs, half=True):
    if (f.ndim == 1):
        y = np.fft.fft(f)
        y = np.power(np.abs(y), 2)
        n = len(y)
        if (half):
            y = y[:int(n/2)]
    else:
        y = np.fft.fft(f, axis=1)
        y = np.power(np.abs(y), 2)
        n = y.shape[1]
        if (half):
            y = y[:, :int(n/2)]
    x = np.fft.fftfreq(n) * fs
    if (half):
        x = x[:int(n/2)]
    return x, y