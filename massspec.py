import numpy as np
import os


class AnalyzeAcq:
    def __init__(self, file):
        self.file = file

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
        """
        t = (np.arange(0, header['uLen'][0]-16) * header['dSamplingPeriod'][0]) / 1e3
        return t