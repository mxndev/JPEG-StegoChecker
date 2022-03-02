import numpy as np
from keras.backend import epsilon
from tensorflow import keras
from datetime import timedelta
import math
from timeit import default_timer as timer
import sys, getopt


class DCTRExctractor:
    'Extract DCTR values'

    def __init__(self, Q=50, q=10, T=7):
        'Initialization'
        self.Q = Q
        self.q = q
        self.T = T
        self.dim = 1600 * (self.T + 1)
        self.start = timer()
        self.generateBase()

    def generateBase(self):
        self.bases = np.zeros((8, 8), dtype=object)

        for mode_r in range(8):
            for mode_c in range(8):

                basic = np.zeros((8, 8))
                for m in range(8):
                    for n in range(8):

                        if mode_r == 0:
                            wr = 1 / math.sqrt(2.)
                        else:
                            wr = 1
                        if mode_c == 0:
                            wc = 1 / math.sqrt(2.)
                        else:
                            wc = 1

                        val = ((wr * wc) / 4) * math.cos(math.pi * mode_r * (2 * m + 1) / 16) * math.cos(
                            math.pi * mode_c * (2 * n + 1) / 16)
                        basic[m][n] = val
                self.bases[mode_r][mode_c] = basic

    def JPEG2spatial(self, coeffs, quant_table):
        print('calculate spatial')
        self.start = timer()
        spatialImage = np.zeros((len(coeffs), len(coeffs[0])))

        for mode_r in range(8):
            for mode_c in range(8):

                Q = quant_table[mode_r][mode_c]
                basis = self.bases[mode_r][mode_c]

                for r in range(len(coeffs) - 8):
                    for c in range(len(coeffs) - 8):

                        DCTval = coeffs[mode_r + r][mode_c + c] * Q / self.q

                        self.start = timer()
                        for m in range(8):
                            for n in range(8):
                                spatialImage[r + m][c + n] = spatialImage[r + m][c + n] + (DCTval * basis[m][n])
        end = timedelta(seconds=timer() - self.start)
        print('calculated spatial time:' + str(end))
        return spatialImage

    def Corr8x8float(self, marray, kernel):
        print('start calculate cor8x8')
        sseKernelArray = np.zeros((16, 4))
        self.start = timer()
        for i in range(8):
            sseKernelArray[(i * 2)][0] = kernel[i][0]
            sseKernelArray[(i * 2)][1] = kernel[i][1]
            sseKernelArray[(i * 2)][2] = kernel[i][2]
            sseKernelArray[(i * 2)][3] = kernel[i][3]

            sseKernelArray[(i * 2) + 1][0] = kernel[i][4]
            sseKernelArray[(i * 2) + 1][1] = kernel[i][5]
            sseKernelArray[(i * 2) + 1][2] = kernel[i][6]
            sseKernelArray[(i * 2) + 1][3] = kernel[i][7]

        residual = np.zeros((len(marray) - len(kernel) + 1, len(marray[0]) - len(kernel[0]) + 1))
        for ir in range(len(marray) - len(kernel) + 1):
            for ic in range(len(marray[0]) - len(kernel[0]) + 1):
                convVal = 0
                for r in range(8):
                    temp = np.matmul(marray[ir + r][ic:ic + 4], sseKernelArray[(r * 2)])
                    convVal = convVal + temp

                    temp = np.matmul(marray[ir + r][ic + 4:ic + 8], sseKernelArray[(r * 2) + 1])
                    convVal = convVal + temp

                residual[ir][ic] = convVal
        end = timedelta(seconds=timer() - self.start)
        print('calculated cor8x8 time:' + str(end))

        return residual

    def GetFeatures(self, spatialImage):
        self.start = timer()
        Fint = np.zeros((self.dim))
        for i in range(self.dim):
            Fint[i] = 0

        for mode_r in range(8):
            for mode_c in range(8):
                DCTR = self.Corr8x8float(spatialImage, self.bases[mode_r][mode_c])
                print('start calculate features')

                'absolute value, rounding and thresholding of DCT residuals'
                for x in range(len(DCTR)):
                    for y in range(len(DCTR[x])):
                        DCTR[x][y] = int(round(DCTR[x][y]))
                        DCTR[x][y] = min(DCTR[x][y], self.T)

                'compute and merge histograms'
                Findex_mode = ((mode_r * 8) + mode_c) * 25 * (self.T + 1)
                for shift_r in range(8):
                    for shift_c in range(8):
                        shift_r_min = min(shift_r, 8 - shift_r)
                        shift_c_min = min(shift_c, 8 - shift_c)

                        Findex_shift = Findex_mode + (shift_r_min * 5 + shift_c_min) * (self.T + 1)

                        for r in range(shift_r, len(DCTR), 8):
                            for c in range(shift_c, len(DCTR[0]), 8):
                                Fint[Findex_shift + int(DCTR[r][c])] += 1
                end = timedelta(seconds=timer() - self.start)
                print('features' + str(mode_r) + str(mode_c) + 'time:' + str(end))
        F = np.zeros((self.dim))
        for i in range(0, self.dim, self.T + 1):
            sum = 0
            for j in range(self.T + 1):
                sum = sum + Fint[i + j]
            for j in range(self.T + 1):
                F[i + j] = Fint[i + j] / sum

        return F

    def generateDCTR(self, coeff_array, quant_array):
        spatialImage = self.JPEG2spatial(coeff_array, quant_array)
        single_fea = self.GetFeatures(spatialImage)
        return single_fea


def f1_m(precision, recall):
    return 2 * ((precision * recall) / (precision + recall + epsilon()))


def predictLoadedImage(input_dctr):
    model = keras.models.load_model('DCTR_model.h5')
    return model.predict(input_dctr)


args = sys.argv[1:]

try:
    opts, args = getopt.getopt(args, "hf:", ["file="])
except getopt.GetoptError:
    print('JPEG-StegoChecker.py -f <inputfile>')
    sys.exit(2)
for opt, arg in opts:
    if opt == '-h':
        print('JPEG-StegoChecker.py -f <inputfile>')
        sys.exit()
    elif opt in ("-f", "--file"):
        jpeginfo = jpegtools.read_jpeg(arg)
        dctr = DCTRExctractor().generateDCTR(jpeginfo['coef_arrays'][0], jpeginfo['quant_tables'][0])
        stegoPossibility = predictLoadedImage(dctr)
        print("# ==================== File 1.jpg ==================== #")
        print("Path: ", arg)
        print("Image size: ", arg)
        print("Photo is clean probability: ", stegoPossibility*100, "%")
        print("Photo has hidden data probability: ", (1-stegoPossibility*100), "%")
        print("Check result: ", stegoPossibility > 0.5 and "hidden data detected" or " image clean")
        print("# ==================================================== #")