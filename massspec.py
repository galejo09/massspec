import numpy as np
import os
from os.path import join, getsize, isfile, isdir
from pathlib import Path

class AnalyzeAcq:
    def __init__(self, directory):
        self.directory = directory


    def load_data(self, date, listfiles=True, files="all"):
        d_format = parse(date)
        d = f"/{d_format.year}-{d_format:%m}-{d_format:%d}"
        full_path = Path(self.directory + d)
        print(full_path)

        if isdir(full_path) is False:
            raise Exception(f'Folder {d} does not exist')

        if listfiles is True:
            for f in [file for file in os.listdir(full_path)]:
                print(f)

        data = []
        
        if files is "all":
            for root, dirs, files in os.walk(full_path):
                for f in files:
                    data.append(str(Path(os.path.join(root, f))))
        if type(files) is list:
            for root, dirs, files in os.walk(full_path):
                for f in files:
                    if f[:2] in files:
                        data.append(str(Path(os.path.join(root, f))))

        return data

                    
    def read_header(self, display=False):
        """
        Reads the header of a binary acquisiton file.

        :param display: if True, the values in the header are printed
        :type: bool

        :return: values in the header
        :type: dict
        """
        if os.path.exists(self.file) is False:
            raise Exception(f'The file {self.file} does not exist.')
        else:
            f = np.fromfile(self.file, count=-1)

        params = ['uuid', 'nVersion', 't', 'nFramesCount', 'uLen', 'u1Pos', 'dHighV',
                  'dLowV', 'nFrameSize', 'nSize270', '1AcqFile', 'dThreshold1', 'dSamplingPeriod']

        dtypes = [np.uint8, np.int, np.uint32, np.int, np.uint32, np.uint64,
                  np.float64, np.float64, np.int32, np.int32, np.int32, np.float64, np.float64]

        counts = [16, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

        offsets = [0, 16, 20, 28, 32, 36, 44, 52, 60, 64, 68, 72, 80]

        data = {}

        for p, d, c, o in zip(params, dtypes, counts, offsets):
            data[p] = np.fromfile(self.file, dtype=d, count=c, offset=o)

        if data['nVersion'] >= 2:
            data['nCommentLen'] = np.fromfile(
                self.file, dtype=np.int32, count=1, offset=88)  # comment length
            data['szComment'] = np.fromfile(
                self.file, dtype=np.uint8, count=data['nCommentLen'][0], offset=92)  # comment string

        data['u1Size'] = os.path.getsize(
            self.file)  # size of the file in bytes

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
        t = (np.arange(0, header['uLen'][0]-16) * header['dSamplingPeriod'][0]) / 1e3

        return t


    def read_single_frame(self, header, frame):
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
        if frame not in range(1, header['nFramesCount'][0]+1):
            raise ValueError("Choose a frame number between 1 and {}".format(header['nFramesCount'][0]))
        else:
            spectrum = np.fromfile(self.file, dtype=np.int8, count=header['uLen'][0], offset=np.int(header['u1Pos'][0] + (frame-1)*header['uLen'][0]))
            spectrum = spectrum[0:-16]

        # convert bytes to volts; the equivalent MATLAB script is byte2volts
        int8max = np.float64(127) # originally double(int8(2^7)) in MATLAB
        spectrum = -( ((header['dHighV'] - header['dLowV']) / (2.0*int8max)) * (spectrum+int8max) + header['dLowV'])

        return spectrum


    def read_frames(self, header, subset="all"):
        """
        Returns a list of spectra from the acquisition file.

        :param header: header of acquisiton file
        :type: dict
        :param subset: range of frames as a tuple; default is "all" 
        :type: tuple

        :return: 2D array of spectra
        :type: numpy.ndarray
        """
        n_frames = header['nFramesCount'][0]

        spectra = []

        if subset is "all":
            for frame in range(1, n_frames+1):
                spectrum = self.read_single_frame(header, frame)
                spectra.append(spectrum)
        elif type(subset) is tuple:  
            start = subset[0]
            end = subset[1]
            if start > n_frames:
                raise IOError(f"Frame {start} is greater than the total number of frames ({n_frames})")
            elif end > n_frames:
                raise IOError(f"Frame {end} is greater than the total numner of frames ({n_frames})")
            for frame in range(start, end+1):
                spectrum = self.read_single_frame(header, frame)
                spectra.append(spectrum)
        else:
            raise TypeError("Subset argument must be string 'all' or tuple (start, end)")

        return np.asarray(spectra)


    def avg_spectra(self, spectra):
        return np.mean(spectra, axis=0)


    def mass2charge(self, t, params):
        """
        Converts flight times to mass-to-charge ratios.

        :param t: flight times in us
        :type: numpy.ndarray
        :param params: constants for the mass-to-charge formula; 
            if a list, the order must be as follows: [E_0, s_0, E_1, s_1, d]; 
            if a dict, the keys must be strings: "E_0", "s_0", "E_1", "s_1", "d";
            E-fields must be provided in V/m and distances in meters.
        :type: list

        :return: mass-to-charge ratios
        :type: numpy.ndarray 
        """
        if type(params) is list:
            E_0 = params[0]
            s_0 = params[1]
            E_1 = params[2]
            s_1 = params[3]
            d = params[4]
        elif type(params) is dict:
            E_0 = params["E_0"]
            s_0 = params["s_0"]
            E_1 = params["E_1"]
            s_1 = params["s_1"]
            d = params["d"]
        else:
            TypeError("Arg params must be a list or dictionary")
            
        return 2*( (t*(E_0*s_0 + E_1*s_1)**(1/2))/d )**2
