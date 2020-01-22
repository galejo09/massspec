import numpy as np 

class AcqReadHeader:
	def __init__(self, file):
		self.file = np.fromfile(file, count=-1)

	def read(file, display=False): 		

		params = ['uuid', 'nVersion', 't', 'nFramesCount', 'uLen', 'u1Pos', 'dHighV', 'dLowV', 'nFrameSize', 'nSize270', '1AcqFile', 'dThreshold1', 'dSamplingPeriod']

		dtypes = [np.uint8, np.int, np.uint32, np.int, np.uint32, np.uint64, np.float64, np.float64, np.int32, np.int32, np.int32, np.float64, np.float64]

		counts = [16, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

		offsets = [0, 16, 16, 20, 28, 32, 36, 44, 52, 60, 64, 68, 72, 80]

		data = {}

		for p, d, c, o in zip(params, dtypes, counts, offsets):
			data[p] = np.fromfile(file, dtype=d, count=c, offset=o)

		if data['nVersion'] >= 2:
			data['nCommentLen'] = np.fromfile(file, dtype=np.int32, count=1, offset=88) # comment length
			data['szComment'] = np.fromfile(file, dtype=np.uint8, count=data['nCommentLen'], offset=92) # comment string

		if display == True:
			for key, value in data.items():
				print(f'{key} : {value}')

		return data