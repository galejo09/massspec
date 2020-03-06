import numpy as np
import os
from os.path import join, getsize, isfile, isdir
from pathlib import Path
from dateutil.parser import parse
from scipy import signal
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse


class AnalyzeAcq:
    def __init__(self, directory):
        self.directory = directory

    def load_data(self, date, listfiles=True, subset="all"):
        """
        Stores file names within the experiment directory.

        :param date: date of the experiment in any format
        :type: string
        :param listfiles: if True, every file name in arg subset and its index is printed
        :type: bool
        :param subset: list of file names to be stored (e.g. [1, 2] for files 01.signal.div, 02.signal.div); all files stored by default
        :type: list

        :return: full paths of files in date folder
        :type: list
        """
        d_format = parse(date)
        d = f"/{d_format.year}-{d_format:%m}-{d_format:%d}"
        full_path = Path(self.directory + d)

        if isdir(full_path) is False:
            raise Exception(f'Folder {d} does not exist')

        data = []

        i = 0

        if subset is "all":
            for root, dirs, files in os.walk(full_path):
                for f in files:
                    if 'div' in f:
                        data.append(str(Path(os.path.join(root, f))))
        if isinstance(subset, list):
            subset = [str(file_no).zfill(2) for file_no in subset]
            for root, dirs, files in os.walk(full_path):
                for file_no in subset:
                    files = [f for f in files if 'div' in f]
                    for f in files:
                        if f[:2] == file_no:
                            data.append(str(Path(os.path.join(root, f))))
                            if listfiles is True:
                                print(f"{i} : {f}")
                                i += 1

        return data

    def read_header(self, file, display=False):
        """
        Reads the header of a binary acquisiton file.

        :param display: if True, the values in the header are printed
        :type: bool

        :return: values in the header
        :type: dict
        """
        if os.path.exists(file) is False:
            raise Exception(f'The file {file} does not exist.')
        elif file.endswith('.div') is False:
            raise IOError(f'The file must have a .div extension')
        else:
            f = np.fromfile(file, count=-1)

        params = ['uuid', 'nVersion', 't', 'nFramesCount', 'uLen', 'u1Pos', 'dHighV',
                  'dLowV', 'nFrameSize', 'nSize270', '1AcqFile', 'dThreshold1', 'dSamplingPeriod']

        dtypes = [np.uint8, np.int, np.uint32, np.int, np.uint32, np.uint64,
                  np.float64, np.float64, np.int32, np.int32, np.int32, np.float64, np.float64]

        counts = [16, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

        offsets = [0, 16, 20, 28, 32, 36, 44, 52, 60, 64, 68, 72, 80]

        data = {}

        for p, d, c, o in zip(params, dtypes, counts, offsets):
            data[p] = np.fromfile(file, dtype=d, count=c, offset=o)

        if data['nVersion'] >= 2:
            data['nCommentLen'] = np.fromfile(
                file, dtype=np.int32, count=1, offset=88)  # comment length
            data['szComment'] = np.fromfile(
                file, dtype=np.uint8, count=data['nCommentLen'][0], offset=92)  # comment string

        data['u1Size'] = os.path.getsize(
            file)  # size of the file in bytes

        if display == True:
            for key, value in data.items():
                print(f'{key} : {value}')

        return data

    def time(self, header):
        """
        Returns time axis in us.

        :param header: header of acquisition file
        :type: dict

        :return: time array in us
        :type: numpy.ndarray
        """
        t = (np.arange(0, header['uLen'][0] - 16)
             * header['dSamplingPeriod'][0]) / 1e3

        return t

    def read_single_frame(self, file, header, frame):
        """
        Returns spectrum of frame from file.

        :param header: header of acquisiton file
        :type: dict
        :param frame: frame number; if frame is outside the bounds of (1, nFramesCount), an exception is raised
        :type: int

        :return: spectrum in uV
        :type: numpy.ndarray
        """
        # check if frame number is less than the number of frames in file
        if frame not in range(1, header['nFramesCount'][0] + 1):
            raise ValueError("Choose a frame number between 1 and {}".format(
                header['nFramesCount'][0]))
        else:
            spectrum = np.fromfile(file, dtype=np.int8, count=header['uLen'][0], offset=np.int(
                header['u1Pos'][0] + (frame - 1) * header['uLen'][0]))
            spectrum = spectrum[0:-16]

        # convert bytes to volts; the equivalent MATLAB script is byte2volts
        int8max = np.float64(127)  # originally double(int8(2^7)) in MATLAB
        spectrum = -(((header['dHighV'] - header['dLowV']) /
                      (2.0 * int8max)) * (spectrum + int8max) + header['dLowV'])

        return spectrum

    def read_frames(self, file, header, subset="all", average=False):
        """
        Returns a list of spectra from the acquisition file.

        :param header: header of acquisiton file
        :type: dict
        :param subset: range of frames as a tuple; default is 'all'
        :type: tuple
        :param average: if True, mean of spectra is returned
        :type: bool

        :return: 2D array of spectra or mean spectrum
        :type: numpy.ndarray
        """
        n_frames = header['nFramesCount'][0]

        spectra = []

        if subset is "all":
            for frame in range(1, n_frames + 1):
                spectrum = self.read_single_frame(file, header, frame)
                spectra.append(spectrum)
        elif isinstance(subset, tuple):
            start = subset[0]
            end = subset[1]
            if start > n_frames:
                raise IOError(
                    f"Frame {start} is greater than the total number of frames ({n_frames})")
            elif end > n_frames:
                raise IOError(
                    f"Frame {end} is greater than the total numner of frames ({n_frames})")
            for frame in range(start, end + 1):
                spectrum = self.read_single_frame(file, frame)
                spectra.append(spectrum)
        else:
            raise TypeError(
                "Subset argument must be string 'all' or tuple (start, end)")

        if average is True:
            return np.mean(spectra, axis=0)

        return np.asarray(spectra)

    def m2z(self, t, voltages, params):
        """
        Converts flight times to mass-to-charge ratios calibrated according to the proton peak.
                Also returns relative voltages.

        :param t: flight times in us
        :type: numpy.ndarray
        :param params: constants for the mass-to-charge formula;
            if a list, the order must be as follows: [V_0, V_1, s_0, s_1, d];
            if a dict, the keys must be strings: "V_0", "V_1", "s_0", "s_1", "d";
            voltages must be provided in V and distances in m.
        :type: list

        :return: mass-to-charge ratios, relative voltages
        :type: numpy.ndarray, numpy.ndarray
        """
        if len(params) != 5:
            raise IOError("Arg params must have length 5")

        if isinstance(params, list):
            V_0 = params[0]
            V_1 = params[1]
            s_0 = params[2]
            s_1 = params[3]
            d = params[4]
        elif isinstance(params, dict):
            V_0 = params["V_0"]
            V_1 = params["V_1"]
            s_0 = params["s_0"]
            s_1 = params["s_1"]
            d = params["d"]
        else:
            TypeError("Arg params must be a list or dictionary")

        Da = 6.022e26  # 1kg in Da
        e = 1.602e-19  # elementary charge

        t = t / 1e6  # convert from us to s

        E_0 = (V_0 - V_1) / s_0
        E_1 = V_1 / s_1

        mz = 2 * Da * e * np.square((t * (E_0 * s_0 + E_1 * s_1)**(1 / 2)) / d)

        peaks_i = signal.find_peaks(voltages, height=1)[
            0]  # find voltage peaks

        peaks = []

        for index in peaks_i:
            peaks.append(voltages[index])

        # identify the first peak as the proton
        proton, proton_i = peaks[0], peaks_i[0]

        # let the calibration factor be alpha
        alpha = np.absolute(1 - mz[proton_i])

        if mz[proton_i] > 1:
            mz = mz - alpha
        elif mz[proton_i] < 1:
            mz = mz + alpha

        return mz, voltages / np.max(voltages)

    def delta_t(self, h1, h2):
        """
        Calculates the time difference between two spectra in hours.

        :param h1: header of the first spectra
        :type: dict
        :param h2: header of the second spectra
        :type: dict

        :return: time difference in hours
        :type: numpy.float64
        """
        return np.absolute(h2["t"][0] - h1["t"][0]) / 3600

    def find_peaks(self, mz, voltages):
        """
        Enables manual peak annotation and returns the coordinates.

        Right-click to mark a peak with a red circle.

        :param mz: mass-to-charge ratios (x-axis)
        :type: numpy.ndarray
        :param voltages: voltages (y-axis)
        :type: numpy.ndarray

        :return: peak coordinates as tuples (x, y)
        :type: list
        """
        coords = []

        fig, ax = plt.subplots(figsize=(15, 15))
        ax.plot(mz, voltages, zorder=-1)

        def on_click(event):
            if event.button == 3:  # right-click
                ix, iy = event.xdata, event.ydata
                dot = Ellipse((ix, iy), width=1.4, height=0.0007, color='r')
                ax.add_patch(dot)
                ax.figure.canvas.draw()
                coords.append((ix, iy))

        fig.canvas.mpl_connect('button_press_event', on_click)
        fig.show()

        return coords

    def identify_peaks(self, protein, substrate, peaks,
                       protein_index=0, unknown=False):
        """
        Returns fragment labels for successfully identified peaks. Peaks that cannot be identified are returned with
        a label of the following format "unknown: {mass of adduct}".

        :param protein: one of ["bradykinin_H", "bradykinin_2H"]
        :type: string
        :param substrate: one of ["ITO", "Si", "chalcogenide", "borosilicate"]
        :type: string
        :param peaks: peak coordinates (mz, voltage) as tuples
        :type: list
        :param protein_index: index of the specified protein within list(peaks); first peak is the default
        :type: int
        :param unknown: if True, the index of the peak and the mass of the corresponding adduct are printed for all unidentified peaks
        :type: bool

        :return: names of fragments of the specified protein as strings
        :type: list
        """
        proteins = {
            "bradykinin_H": "[M+H]$^+$",
            "bradykinin_2H": "[M+2H]$^{2+}$"
        }

        substrates = ["ITO", "Si", "chalcogenide", "borosilicate"]

        if protein not in proteins.keys():
            raise IOError(
                f"Arg protein must be one of the following: 'bradykinin_H', 'bradykinin_2H'")

        if substrate not in substrates:
            raise IOError(
                f"Arg substrate must be one of the following: 'ITO', 'Si', 'chalcogenide', 'borosilicate'")

        if protein_index not in range(len(peaks) - 1):
            raise IOError(
                f"Arg protein_index must be an integer in the range (0, {len(peaks)-1})")

        labels, unknowns = [], []

        if protein is "bradykinin_H":
            expected_mass = 1061.23
            proton = np.absolute(expected_mass - peaks[protein_index][0])
            if 0 <= proton <= 1:
                if substrate is "ITO":
                    masses = {
                        23: "[M+Na]$^+$",
                        39: "[M+K]$^+$",
                        64: "[M+Zn]$^+$"
                    }
                if substrate is "Si":
                    masses = {
                        17: "[M-NH$_3$+H]$^+$",
                        23: "[M+Na]$^+$",
                        39: "[M+K]$^+$",
                        63: "[M+Cu]$^+$"
                    }
            else:
                raise IOError(
                    f"Peak of index {protein_index} could not be identified as {protein}")

        mz, voltages = list(zip(*peaks))

        for i, mass in enumerate(mz):
            if i is protein_index:
                continue
            else:
                adduct = np.absolute(np.round(mass - mz[protein_index]))
                lower_lim, upper_lim = adduct - 1, adduct + 1
                if adduct in masses.keys():
                    labels.append(masses[adduct])
                elif lower_lim in masses.keys():
                    labels.append(masses[lower_lim])
                elif upper_lim in masses.keys():
                    labels.append(masses[upper_lim])
                else:
                    unknowns.append((i, adduct))
                    labels.append(f"unknown: {adduct}")

        labels.insert(protein_index, proteins[protein])

        if unknown is True:
            for tup in unknowns:
                peak_index, adduct_mass = tup[0], tup[1]
                print(
                    f"Peak {peak_index} was unable to be identified. Mass = {adduct_mass}")

        return labels

    def label_peaks(self, header, mz, voltages, peaks,
                    labels, plotprops, savefig=None):
        """
        Generates an annotated spectrum.

        :param header: header of acquisiton file
        :type: dict
        :param mz: mass-to-charge ratios (x-axis)
        :type: numpy.ndarray
        :param voltages: voltages (y-axis)
        :type: numpy.ndarray
        :param peaks: peak coordinates (mz, voltage) as tuples
        :type: list
        :param labels: str(label) for each peak in list(peaks); note that peaks and labels are matched by index
        :type: list
        :param plotprops: properties of the plot;
        {"title" : str,
        "xlim" : (left, right),
        "ylim" : (bottom, top),
        "labelspacing" : float}
        :param savefig: spectrum will be saved with this file name; if None, the spectrum will not be saved
        :type: str

        :return: annotated spectrum
        :type: matplotlib.figure.Figure
        """
        plt.figure(figsize=(15, 15))
        plt.plot(mz, voltages,
                 label=f"Average of {header['nFramesCount'][0]} shots")
        plt.xlim(plotprops["xlim"][0], plotprops["xlim"][1])
        plt.ylim(plotprops["ylim"][0], plotprops["ylim"][1])
        plt.xlabel("$m/z$")
        plt.ylabel("Relative voltage")
        plt.title(plotprops["title"])
        plt.legend(loc="upper right")

        # labels will be spaced by this amount in the y-direction
        z = plotprops["labelspacing"]

        for i, coord in enumerate(peaks):
            plt.annotate(labels[i],
                         xy=(coord[0], coord[1]),
                         ha='center',
                         xytext=(coord[0], coord[1] + z),
                         size=10,
                         arrowprops=dict(arrowstyle="->")
                         )
            z += 0.01

        if savefig is None:
            plt.show()
        elif isinstance(savefig, str):
            plt.savefig(savefig)
        else:
            raise IOError("Arg savefig must be a string ('filename.png')")
