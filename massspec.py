"""
massspec is a Python library for the analysis of TOF mass spectra.

It can be used to:
- convert flight times to m/z
- mark, identify, and label ion peaks
- plot multiple spectra in one figure
- calculate shot-to-shot reproducibility

Note that this module was initially developed for cryo femtosecond mass 
spectrometry in the Atomically Resolved Dynamics Department at the Max Planck 
Institute for the Structure and Dynamics of Matter.
"""
import numpy as np
import os
import pandas as pd
import pywt
import sigfig
import seaborn as sns
import scipy.signal
from os.path import isdir
from pathlib import Path
from dateutil.parser import parse
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse


class AnalyzeAcq:
    def __init__(self, directory):
        self.directory = directory

    def load_data(self, date, subset="all", listfiles=True):
        """
        Stores file names within the experiment directory.

        :param date: date of the experiment in any format
        :type: str
        :param listfiles: if True, every file name in arg subset
            and its index is printed
        :type: bool
        :param subset: list of file names to be stored
            (e.g. [1, 2] for files 01.signal.div, 02.signal.div);
            all files stored by default
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

        if subset == "all":
            for root, dirs, files in os.walk(full_path):
                for f in files:
                    if 'div' in f:
                        data.append(str(Path(os.path.join(root, f))))
                        if listfiles is True:
                            print(f"{i} : {f}")
                            i += 1
        elif isinstance(subset, list):
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
        Reads the header of a binary acquisition file.

        :param file: acquisition file
        :type: str
        :param display: if True, the values in the header are printed
        :type: bool

        :return: values in the header
        :type: dict
        """
        if os.path.exists(file) is False:
            raise Exception(f'The file {file} does not exist.')
        elif file.endswith('.div') is False:
            raise IOError(f'The file must have a .div extension')

        params = [
            'uuid',
            'nVersion',
            't',
            'nFramesCount',
            'uLen',
            'u1Pos',
            'dHighV',
            'dLowV',
            'nFrameSize',
            'nSize270',
            '1AcqFile',
            'dThreshold1',
            'dSamplingPeriod']

        dtypes = [
            np.uint8,
            np.int,
            np.uint32,
            np.int,
            np.uint32,
            np.uint64,
            np.float64,
            np.float64,
            np.int32,
            np.int32,
            np.int32,
            np.float64,
            np.float64]

        counts = [16, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

        offsets = [0, 16, 20, 28, 32, 36, 44, 52, 60, 64, 68, 72, 80]

        data = {}

        for p, d, c, o in zip(params, dtypes, counts, offsets):
            data[p] = np.fromfile(file, dtype=d, count=c, offset=o)

        if data['nVersion'] >= 2:
            data['nCommentLen'] = np.fromfile(
                file, dtype=np.int32, count=1, offset=88)  # comment length
            data['szComment'] = np.fromfile(
                file,
                dtype=np.uint8,
                count=data['nCommentLen'][0],
                offset=92)  # comment string

        data['u1Size'] = os.path.getsize(
            file)  # size of the file in bytes

        if display:
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

        :param file: acquisition file
        :type: str
        :param header: header of acquisiton file
        :type: dict
        :param frame: frame number; if frame is outside the bounds of
            (1, nFramesCount), an exception is raised
        :type: int

        :return: spectrum in mV
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

    def read_frames(self, file, header, subset="all",
                    laser="PHAROS", average=True):
        """
        Returns a list of spectra from the acquisition file.

        :param file: acquisition file
        :type: str
        :param header: header of acquisiton file
        :type: dict
        :param subset: range of frames as a tuple; default is 'all'
        :type: tuple
        :param laser: laser used in the experiment;
            either 'PIRL' (3 micron) or 'PHAROS' (1 micron)
        :type: str
        :param average: if True, mean of spectra is returned;
            if laser = "PIRL", shots with only noise are excluded from the mean
        :type: bool

        :return: 2D array of spectra or mean spectrum
        :type: numpy.ndarray
        """
        n_frames = header['nFramesCount'][0]

        spectra = []

        if subset == "all":
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
            if laser == "PHAROS":
                return np.mean(spectra, axis=0)
            elif laser == "PIRL":
                # exclude spectra with just noise from the calculation of the
                # average
                noise_indices = []
                total_shots = len(spectra)
                for i, shot in enumerate(spectra):
                    if np.max(shot) < 10:
                        noise_indices.append(i)
                spectra = [spectra[i]
                           for i in range(total_shots) if i not in noise_indices]
                header["nFramesCount"] = np.array([len(spectra)])
                print(
                    f"{len(noise_indices)}/{total_shots} spectra are strictly noise.")
                return np.mean(spectra, axis=0)
            else:
                raise IOError(
                    "Arg laser must be one of: 'PHAROS', 'PIRL'")

        return np.asarray(spectra)

    def m2z(self, t, voltages, params):
        """
        Converts flight times to mass-to-charge ratios calibrated according
        to the proton peak.

        Also returns relative voltages.

        :param t: flight times in us
        :type: numpy.ndarray
        :param voltages: voltages in mV
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

        peaks_i = scipy.signal.find_peaks(voltages, height=1)[
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

    def pub_mode(self, fontsize=14, titlesize=24):
        """
        Sets rc params to produce figures that are formatted for publication.
        
        :param fontsize: font size of axes labels 
        :type: int
        :param titlesize: font size of title
        :type: int
        """
        font = {
            "family": "sans-serif",
            "weight": "normal",
            "size": fontsize
        }
        plt.rc("font", **font)
        plt.rc("figure", titlesize=titlesize)
        plt.rc("legend", fontsize='medium')

    def find_peaks(self, mz, voltages, offset=0, display=True):
        """
        Enables manual peak annotation and returns the coordinates.

        Right-click to mark a peak with a red circle.

        :param mz: mass-to-charge ratios (x-axis)
        :type: numpy.ndarray (or list to plot multiple spectra)
        :param voltages: voltages (y-axis)
        :type: numpy.ndarray (or list to plot multiple spectra)
        :param offset: space between each plot if plotting more than one
            spectra; must have same units as voltages
        :type: float
        :param display: if True, peak coordinates are printed
        :type: bool

        :return: peak coordinates as tuples (x, y)
        :type: list
        """
        coords = []

        fig, ax = plt.subplots(figsize=(15, 15))
        if isinstance(mz, list) and isinstance(voltages, list):
            z = 0
            for x, y in zip(mz, voltages):
                ax.plot(
                    x,
                    y + z,
                    zorder=-1)
                z += offset
        else:
            ax.plot(mz, voltages, zorder=-1)

        def on_click(event):
            if event.button == 3:  # right-click
                ix, iy = event.xdata, event.ydata
                dot = Ellipse((ix, iy), width=1.4, height=0.0007, color='r')
                ax.add_patch(dot)
                ax.figure.canvas.draw()
                coords.append((ix, iy))
                if display is True:
                    print(f"{coords.index((ix, iy))} : ({ix}, {iy})")

        fig.canvas.mpl_connect('button_press_event', on_click)

        return coords

    def identify_peaks(
            self,
            file,
            peaks,
            protein='',
            substrate='',
            protein_index=None,
            unknown=False,
            csv=True,
            png=True):
        """
        Returns fragment labels for successfully identified peaks. Peaks that
        cannot be identified are returned with a label of the following format:
        "unknown adduct: {mass of adduct}" for M+H and "{mass}" for M+2H.

        Leave protein and substrate as empty strings for general identification
        of unknown peaks, e.g. label without reference to a protein and without
        mass-to-charge correction for substrate.

        :param file: acquisition file
        :type: str
        :param peaks: peak coordinates (mz, voltage) as tuples
        :type: list
        :param protein: one of
            ["", "bradykinin_H", "bradykinin_2H"]
        :type: str
        :param substrate: one of
            ["", "ITO", "Si", "chalcogenide"]
        :type: str
        :param protein_index: index of the specified protein within
            list(peaks); if protein peak is not present, set to None
        :type: int
        :param unknown: if True, the index of the peak and the mass of the
            corresponding adduct are printed for all unidentified peaks
        :type: bool
        :param csv: if True, a csv file will be created with columns
            0 : Index
            1 : Ion
            2 : Theoretical m/z
            3 : m/z
            4 : Relative Intensity
            The file name will have the following format
            "{date}_{file #}_{protein}_{substrate}.csv"
        :type: bool
        :param png: if True, a table will be created with columns
            0 : Index
            1 : Ion
            2 : Theoretical m/z
            3 : m/z
            4 : Relative Intensity
            The file name will have the following format
            "{date}_{file #}_{protein}_{substrate}.png"
        :type: bool

        :return: fragment labels of the format str({name} ({mass}) \n theory:
            {theoretical mass}) (known peaks) and/or mass-to-charge ratios
            (unknown peaks)
        :type: list
        """
        proteins = {
            "": None,
            "bradykinin_H": "[M+H]$^+$",
            "bradykinin_2H": "[M+2H]$^{2+}$"
        }

        substrates = ["", "ITO", "Si", "chalcogenide", "borosilicate"]

        if protein not in proteins.keys():
            raise IOError(
                f"Arg protein must be one of the following: '', 'bradykinin_H', 'bradykinin_2H'")

        if substrate not in substrates:
            raise IOError(
                f"Arg substrate must be one of the following: '', 'ITO', 'Si', 'chalcogenide', 'borosilicate'")

        labels, unknowns, theory_mzs = [], [], []

        if protein == "bradykinin_H":
            M_H = peaks[protein_index][0]
            M_H_expected_mass = 1061.23
            delta = np.absolute(M_H - M_H_expected_mass)
            percent_error = delta / M_H_expected_mass * 100
            if 0 <= percent_error <= 1:
                if substrate == "ITO":
                    masses = {
                        1016.21: "-COOH",
                        1044.20: "-NH$_3$",
                        1084.22: "[M+Na]$^+$",
                        1100.33: "[M+K]$^+$",
                        1126.61: "[M+Zn]$^+$"
                    }
                elif substrate == "Si":
                    masses = {
                        1044.20: "-NH$_3$",
                        1084.22: "[M+Na]$^+$",
                        1100.33: "[M+K]$^+$"
                    }
                elif substrate == "chalcogenide":
                    masses = {
                        1016.21: "-COOH",
                        1084.22: "[M+Na]$^+$",
                        1100.33: "[M+K]$^+$"
                    }
        elif protein == "bradykinin_2H":
            if protein_index is not None:
                M_2H = peaks[protein_index][0]
                M_2H_expected_mass = 531.12
                delta = np.absolute(M_2H - M_2H_expected_mass)
                percent_error = delta / M_2H_expected_mass * 100
                if 0 <= percent_error <= 1:
                    pass
                else:
                    raise IOError(
                        f"Peak of index {protein_index} could not be identified as {protein}")
            masses = {
                201.20: "x$_1$",
                226.30: "a$_2$",
                230.25: "v$_2$",
                254.31: "b$_2$",
                268: "c$_2$",
                307: "z$_2$",
                323.42: "a$_3$",
                348.38: "x$_2$",
                365: "c$_3$",
                376.43: "w$_3$",
                380.47: "a$_4$",
                404: "z$_3$",
                417.48: "y$_3$",
                422: "c$_4$",
                446.51: "x$_3$",
                474.54: "v$_4$",
                490.56: "z$_4$",
                527.65: "a$_5$",
                553: "b$_5$",
                569: "c$_5$",
                598.72: "d$_6$",
                614.72: "a$_6$",
                638: "z$_5$",
                642.73: "b$_6$",
                711.84: "a$_7$",
                736.80: "x$_6$",
                753: "c$_7$",
                764.86: "w$_7$",
                805.91: "y$_7$",
                859.02: "a$_8$",
                885: "b$_8$",
                900: "c$_8$",
                903.02: "y$_8$",
                932: "x$_8$",
                961.08: "-R$_{arg}$",
                970.10: "-Benzyl"
            }

        mz, voltages = list(zip(*peaks))

        if protein == "" and substrate == "":
            for i, mass in enumerate(mz):
                unknowns.append((i, mass))
                labels.append(f"{mass:.2f}")
        elif protein == "bradykinin_H":
            for i, mass in enumerate(mz):
                if i is protein_index:
                    continue
                else:
                    adduct = mass - mz[protein_index]
                    identified = False
                    for theory_mass in masses.keys():
                        upper_lim = (0.01 * theory_mass) + theory_mass
                        lower_lim = -(0.01 * theory_mass) + theory_mass
                        if mass > lower_lim and mass < upper_lim:
                            if masses[theory_mass] not in labels:
                                theory_mzs.append(theory_mass)
                                labels.append(
                                    masses[theory_mass])
                                identified = True
                                break
                            else:
                                # If more than one ion is within 1% of the
                                # theoretical mass, determine which m/z is
                                # closer to the theoretical mass
                                index = labels.index(masses[theory_mass])
                                val1 = mz[index]
                                val2 = mass
                                delta1 = np.absolute(val1 - theory_mass)
                                delta2 = np.absolute(val2 - theory_mass)
                                if delta2 == min(delta1, delta2):
                                    # Re-label the other peak as unknown
                                    unknowns.append((index, val1))
                                    theory_mzs[index] = "-"
                                    labels[index] = f"{val1-mz[protein_index]:.2f}"
                                    theory_mzs.append(theory_mass)
                                    labels.append(
                                        masses[theory_mass])
                                    identified = True
                                    break
                    if identified is False:
                        unknowns.append((i, mass))
                        theory_mzs.append("-")
                        labels.append(f"{adduct:.2f}")
            theory_mzs.insert(
                protein_index,
                M_H_expected_mass)
            labels.insert(
                protein_index,
                proteins[protein])
        elif protein == "bradykinin_2H":
            for i, mass in enumerate(mz):
                if i is protein_index:
                    continue
                else:
                    identified = False
                    for theory_mass in masses.keys():
                        upper_lim = (0.01 * theory_mass) + theory_mass
                        lower_lim = -(0.01 * theory_mass) + theory_mass
                        if mass > lower_lim and mass < upper_lim:
                            if masses[theory_mass] not in labels:
                                theory_mzs.append(theory_mass)
                                labels.append(
                                    masses[theory_mass])
                                identified = True
                                break
                            else:
                                # If more than one ion is within 1% of the
                                # theoretical mass, determine which m/z is
                                # closer to the theoretical mass
                                index = labels.index(masses[theory_mass])
                                val1 = mz[index]
                                val2 = mass
                                delta1 = np.absolute(val1 - theory_mass)
                                delta2 = np.absolute(val2 - theory_mass)
                                if delta2 == min(delta1, delta2):
                                    # Re-label the other peak as unknown
                                    unknowns.append((index, val1))
                                    theory_mzs[index] = "-"
                                    labels[index] = f"{val1:.2f}"
                                    theory_mzs.append(theory_mass)
                                    labels.append(
                                        masses[theory_mass])
                                    identified = True
                                    break
                    if identified is False:
                        unknowns.append((i, mass))
                        theory_mzs.append("-")
                        labels.append(f"{mass:.2f}")
            if protein_index is not None:
                theory_mzs.insert(
                    protein_index,
                    M_2H_expected_mass)
                labels.insert(
                    protein_index,
                    proteins[protein])

        if unknown is True:
            for tup in unknowns:
                peak_index, mass = tup[0], tup[1]
                print(
                    f"Peak {peak_index} was unable to be identified. Mass = {mass}")

        mz = [round(mass, 1) for mass in mz]
        voltages = [sigfig.round(np.float(v), sigfigs=2) for v in voltages]
        df = pd.DataFrame({"Ion": labels,
                           "Theoretical m/z": theory_mzs,
                           "m/z": mz,
                           "Relative intensity": voltages})
        display(df)

        date = file.split("\\")[-2]
        fname = file.split("\\")[-1][:2]

        if csv is True:
            df.to_csv(f"{date}_{fname}_{protein}_{substrate}.csv")

        if png is True:
            # Fill the 'Ion' and 'Theoretical m/z' columns with dummy numbers
            # since sns.heatmap does not accept strings
            df = pd.DataFrame({"Ion": np.random.rand(len(mz)),
                               "Theoretical m/z": np.random.rand(len(mz)),
                               "m/z": mz,
                               "Relative intensity": voltages})
            # values to use in the heatmap
            annotations = np.column_stack([labels, theory_mzs, mz, voltages])

            # Use a blank heatmap as a table
            fig, ax = plt.subplots(figsize=(8, 15))
            ax.xaxis.tick_top()
            ax.tick_params(length=0)
            sns.heatmap(df,
                        cmap=['white'],
                        annot=annotations,
                        fmt='',
                        linewidths=0.1,
                        linecolor='k',
                        cbar=False,
                        yticklabels=False,
                        ax=ax)
            plt.savefig(
                f"{date}_{fname}_{protein}_{substrate}.png",
                bbox_inches='tight')

        return labels

    def label_peaks(self, header, mz, voltages, peaks,
                    peak_labels, plotprops, legend_labels=[], savefig=None):
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
        :param peak_labels: str(label) for each peak in list(peaks); note that
            peaks and labels are matched by index
        :type: list
        :param legend_labels: str(label) for each plot if plotting more than
            one spectra
        :type: list
        :param plotprops: properties of the plot;
        {"title" : str,
        "figsize" : tup,
        "xlim" : (left, right),
        "ylim" : (bottom, top),
        "offset" : float
        "labelspacing" : float,
        "labelsize" : float}
        :type: dict
        :param savefig: spectrum will be saved with this file name; if None,
            the spectrum will only be shown
        :type: str

        :return: annotated spectrum
        :type: matplotlib.figure.Figure
        """
        fig = plt.figure(figsize=plotprops["figsize"])
        if isinstance(mz, list) and isinstance(voltages, list):
            z = 0
            for i, x, y in zip(range(len(mz)), mz, voltages):
                plt.plot(
                    x,
                    y + z,
                    label=legend_labels[i])
                z += plotprops["offset"]
            handles, labels = plt.gca().get_legend_handles_labels()
            plt.legend(
                reversed(handles),
                reversed(labels),
                loc='upper left',
                bbox_to_anchor=(
                    1,
                    1))
        else:
            plt.plot(mz, voltages,
                     label=f"Average of {header['nFramesCount'][0]} shots")
            plt.legend(loc='upper left', fontsize='medium')
        plt.xlim(plotprops["xlim"][0], plotprops["xlim"][1])
        plt.ylim(plotprops["ylim"][0], plotprops["ylim"][1])
        plt.xlabel("$m/z$")
        plt.ylabel("Relative intensity")
        plt.title(plotprops["title"])

        labelspacing = 0.005

        for i, coord in enumerate(peaks):
            plt.annotate(peak_labels[i],
                         xy=(coord[0], coord[1]),
                         ha='center',
                         xytext=(coord[0], coord[1] + labelspacing),
                         size=plotprops["labelsize"],
                         arrowprops=dict(arrowstyle="->")
                         )
            labelspacing += plotprops["labelspacing"]

        if savefig is None:
            plt.show()
        elif isinstance(savefig, str):
            plt.savefig(savefig, bbox_inches='tight')
        else:
            raise IOError("Arg savefig must be a string (i.e. 'filename.png')")

        return fig

    def get_pulse_energies(self, date, wavelength, nd=''):
        """
        Retrieves pulse energy measurements for specified date and wavelength.

        :param date: date of the experiment in any format
        :type: str
        :param wavelength: wavelength in nm; one of '1026nm', '513nm', '342nm'
        :type: str
        :param nd: ND filter; one of '0.1ND', '0.3ND', '1.0+0.5ND', 
            '1.0+0.5+0.4ND', '1.0+0.5+0.1ND', 'None'; 
            if '', the function assumes that only 1 set of ND filters were used 
            for the specified date
        :type: str
        
        :return: dictionary of format
            {'PHAROS GUI power in mW' : 'Measured pulse energy in uJ'}
        :type: dict
        """
        d_format = parse(date)
        date = f"{d_format.year}-{d_format:%m}-{d_format:%d}"

        df_powers = pd.read_csv(
            self.directory +
            "\\power_measurements.csv",
            keep_default_na=False)

        measurements = df_powers[df_powers.date == date]
                                          
        wavelengths = list(set(measurements['wavelength'].tolist()))
                     
        if wavelength not in wavelengths:        
            raise IOError(
                f"Wavelength {wavelength} was not used on {date}. Choose one of {wavelengths}.")
                     
        w = measurements[measurements.wavelength == wavelength]
                     
        if measurements['wavelength'].tolist().count(wavelength) != 1 and nd == '':
            filters = w['filter'].tolist()
            raise IOError(
                f"Choose one set of filters from the following: {filters}")
                            
        if nd != '':
            powers = w[w['filter'] == nd].iloc[:, 4:]
        else:
            powers = w.iloc[:, 4:]                 
                                       
        cols = powers.columns.tolist()
                                          
        powervals = powers.values.tolist()[0]
                     
        dict_powers = dict(zip(cols, powervals))

        for power, pulseE in dict_powers.items():
            if dict_powers[power] != "":
                dict_powers[power] = str(pulseE) + r' $\mu$J'

        dict_powers = {k:v for k,v in dict_powers.items() if v != ''}            

        return dict_powers

    def plot_spectra(
            self, headers, mz, voltages, labels, plotprops, savefig=None):
        """
        Plots multiple spectra in one figure.

        :param headers: the headers of the acquisiton files
        :type: list
        :param mz: mass-to-charge ratios of all spectra (x-axis)
        :type: list
        :param voltages: voltages of all spectra (y-axis)
        :type: list
        :param labels: labels corresponding to each spectra; note that mz[0],
            voltages[0], and labels[0] must refer to the same spectrum
        :type: list
        :param plotprops: properties of the plot; "xlim", "ylim" optional
        {"title" : str,
        "figsize" : tup,
        "offset" : float,
        "xlim" : tup,
        "ylim" : tup}
        :type: dict
        :param savefig: spectrum will be saved with this file name; if None,
            the spectrum will only be shown
        :type: str

        :return: several spectra in one figure
        :type: matplotlib.figure.Figure
        """
        fig = plt.figure(figsize=plotprops["figsize"])
        z = 0
        for i, x, y in zip(range(len(mz)), mz, voltages):
            avg = headers[i]['nFramesCount'][0]
            plt.plot(
                x,
                y + z,
                label=f"{labels[i]}, avg. of {avg} shots")
            z += plotprops["offset"]
        if "xlim" and "ylim" in plotprops:
            x1, x2 = plotprops["xlim"]
            y1, y2 = plotprops["ylim"]
            plt.xlim((x1, x2))
            plt.ylim((y1, y2))
        plt.yticks([])
        plt.xlabel("$m/z$")
        plt.title(plotprops["title"])
        handles, labels = plt.gca().get_legend_handles_labels()
        plt.legend(
            reversed(handles),
            reversed(labels),
            loc='upper left',
            bbox_to_anchor=(
                1,
                1))

        if savefig is None:
            plt.show()
        elif isinstance(savefig, str):
            plt.savefig(savefig, bbox_inches='tight')
        else:
            raise IOError(
                "Arg savefig must be a string (i.e. 'filename.png')")

        return fig

    def shot2shot(self, file, header, t, voltages, flight_time=False):
        """
        Measures shot-to-shot reproducibility by calculating the percentage of
        shots that produce a molecular ion (bradykinin) peak.

        The signal is denoised with Undecimated Discrete Wavelet Transform
        (UDWT) and smoothed with the Savitsky-Golay filter before peak-finding.

        :param file: acquisition file
        :type: str
        :param header: header of acquisiton file
        :type: dict
        :param t: flight times in us
        :type: numpy.ndarray
        :param voltages: voltages in mV for multiple shots
        :type: numpy.ndarray
        :param flight_time: if True, average flight time is also returned
        :type: bool

        :return: percentage of shots that produce a molecular ion peak
        :type: float
        """
        # Check if file is empty
        if header['nFramesCount'] == 0:
            print(f"{file} does not contain voltage data.")
            return

        # Estimate the molecular ion flight time (est_flight_t) from the
        # average spectrum
        voltages_avg = self.read_frames(file, header)

        voltages_filtered = scipy.signal.savgol_filter(voltages_avg,
                                                       window_length=101,
                                                       polyorder=2)

        peaks_avg = scipy.signal.find_peaks(voltages_filtered,
                                            height=3.5 * np.std(voltages_avg[16000:21000]),
                                            distance=200)[0]

        widths = scipy.signal.peak_widths(voltages_filtered, peaks_avg)[0]

        t_peaks_avg, v_peaks_avg = [], []

        for i in peaks_avg:
            t_peaks_avg.append(t[i])
            v_peaks_avg.append(voltages_filtered[i])

        est_flight_t = 0

        # Use width of the molecular ion peak to set lower limit of flight time
        for i, time in enumerate(t_peaks_avg):
            if time > 21.5 and time < 23:
                est_flight_t = time
                # Convert units of peak width from samples to us; approx. 1000
                # points/us
                peak_width = widths[i] / 1000
                lower_flight_t = time - peak_width / 2
                break

        # If the average spectrum does not have significant molecular ion peak,
        # take lowest possible estimate of flight time
        if est_flight_t is 0:
            lower_flight_t = 21.6
            upper_lim = 21000
            lower_lim = 16000
        else:
            # Set range of noise measurement
            upper_lim = int(est_flight_t) * 1000
            lower_lim = upper_lim - 5000

        t_peaks, v_peaks = [], []

        for shot in voltages:
            # Perform Undecimated Discrete Wavelet Transform to denoise signal
            coeffs = pywt.swt(data=shot,
                              wavelet='db8')

            coeffst = []

            for ca, cd in coeffs:
                cat = pywt.threshold(data=ca,
                                     value=np.std(ca[lower_lim:upper_lim]) * 2,
                                     mode='hard')
                cdt = pywt.threshold(data=cd,
                                     value=np.std(cd[lower_lim:upper_lim]) * 2,
                                     mode='hard')
                coeffst.append((cat, cdt))

            shot_udwt = pywt.iswt(coeffs=coeffst,
                                  wavelet='db8')

            # Apply Savitzky–Golay filter to smooth denoised signal
            shot_filtered = scipy.signal.savgol_filter(shot_udwt,
                                                       window_length=101,
                                                       polyorder=2)

            # Noise is measured as the standard deviation of the raw signal 5
            # us before the expected molecular ion peak
            noise = np.std(shot[lower_lim:upper_lim])

            # 7:2 signal-to-noise ratio
            signal = noise * 3.5

            # Filtering the signal lowers intensity values; adjust to match
            # original intensities in the range of the molecular ion peak
            low, high = 21600, 23000
            if np.amax(shot[low:high]) >= signal:
                shot_filtered = np.amax(
                    shot[low:high]) / np.amax(shot_filtered[low:high]) * shot_filtered

            peaks = scipy.signal.find_peaks(shot_filtered,
                                            height=signal,
                                            distance=200)[0]

            t_shot_peaks, v_shot_peaks = [], []

            for i in peaks:
                t_shot_peaks.append(t[i])
                v_shot_peaks.append(shot_filtered[i])

            t_peaks.append(t_shot_peaks)
            v_peaks.append(v_shot_peaks)

        flight_times = []

        shots = np.zeros(len(voltages))

        # the first peak greater than or equal to the lower limit of flight
        # time is considered the molecular ion peak
        for shot, times in enumerate(t_peaks):
            for time in times:
                if time >= lower_flight_t:
                    flight_times.append(time)
                    shots[shot] = 1
                    break

        mean_flight_t = np.mean(flight_times)

        count = np.sum(shots)

        count_percent = np.round((count / len(voltages)) * 100)

        print(
            f"number of spectra with [M+H] peak : {count},\nnumber of shots : {len(voltages)},\n% of shots with [M+H] peak : {count_percent}")

        if flight_time is True:
            return count_percent, mean_flight_t

        return count_percent
