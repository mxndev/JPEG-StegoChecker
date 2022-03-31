import numpy as np
from tensorflow import keras
import sys, getopt, os
import jpeg_toolbox
from scipy.fftpack import dct, idct
from scipy.signal import convolve2d


def conv2(x, y, mode='same'):
    return np.rot90(convolve2d(np.rot90(x, 2), np.rot90(y, 2), mode=mode), 2)


def dct2(block):
    return dct(dct(block.T, norm='ortho').T, norm='ortho')


def idct2(block):
    return idct(idct(block.T, norm='ortho').T, norm='ortho')


def multiplyMatrices(x, y):
    matrix = np.zeros((len(x), len(y)))

    for i in range(len(x)):
        for j in range(len(y)):
            matrix[i][j] += x[i] * y[j]
    return matrix


def blockproc(A,m,n,fun):

    results_rows = []
    for y in range(0,A.shape[0],m):
        results_cols = []
        for x in range(0,A.shape[1],n):
            results_cols.append(fun(A[y:y+m,x:x+n]))
        results_rows.append(results_cols)

    patch_rows = results_rows[0][0].shape[0]
    patch_cols = results_rows[0][0].shape[1]
    final_array_cols = results_rows[0][0].shape[1] * len(results_rows[0])
    final_array_rows = results_rows[0][0].shape[0] * len(results_rows)

    final_array = np.zeros([final_array_rows,final_array_cols],dtype=results_rows[0][0].dtype)
    for y in range(len(results_rows)):
        for x in range(len(results_rows[y])):
            data = results_rows[y][x]
            final_array[y*patch_rows:y*patch_rows+data.shape[0],x*patch_cols:x*patch_cols+data.shape[1]] = data

    return final_array


def DCTR(jpeg_info=None, qf=75.0):
    ## Set parameters
    # number of histogram bins
    T = 4
    # compute quantization step based on quality factor
    if qf < 50:
        q = min(8 * (50 / qf), 100)
    else:
        q = max(8 * (2 - (qf / 50)), 0.2)

    ## Prepare for computing DCT bases
    k = range(0, 8)
    l = range(0, 8)
    k, l = np.meshgrid(k, l)
    A = 0.5 * np.cos((np.multiply((2.0 * k + 1), l) * np.pi) / 16)
    A[0, :] = A[0,:] / np.sqrt(2)
    A = np.transpose(A)
    ## Compute DCTR locations to be merged
    mergedCoordinates = np.empty(25, dtype=object)
    for i in np.arange(0, 5).reshape(-1):
        for j in np.arange(0, 5).reshape(-1):
            coordinates = np.array([[i, j], [i, 10 - j], [10 - i, j], [10 - i, 10 - j]])
            coordinates = coordinates[np.apply_along_axis(all, 1, coordinates < 9),:]
            mergedCoordinates[(i - 1) * 5 + j] = np.unique(coordinates, axis=0)

    ## Decompress to spatial domain
    fun = lambda x=None: np.multiply(x.data, jpeg_info["quant_tables"][0][0])
    I_spatial = blockproc(jpeginfo["coef_arrays"][0], 8, 8, fun)
    fun = lambda x=None: idct2(x)
    I_spatial = blockproc(I_spatial, 8, 8, fun) + 128
    ## Compute features
    modeFeaDim = np.asarray(mergedCoordinates).size * (T + 1)
    F = np.zeros(64 * modeFeaDim)
    for mode_r in np.arange(0, 8):
        for mode_c in np.arange(0, 8):
            modeIndex = mode_r * 8 + mode_c
            # Get DCT base for current mode
            DCTbase = multiplyMatrices(A[:, mode_r], np.transpose(A[:, mode_c]))
            # Obtain DCT residual R by convolution between image in spatial domain and the current DCT base
            R = conv2(I_spatial - 128, DCTbase, 'valid')
            # Quantization, rounding, absolute value, thresholding
            R = np.abs(np.round(R / q))
            R[R > T] = T
            # Core of the feature extraction
            for merged_index in np.arange(0, len(mergedCoordinates)):
                f_merged = np.zeros(T)
                firstArray = mergedCoordinates[merged_index]
                for coord_index in np.arange(0, len(firstArray)):
                    r_shift = firstArray[coord_index][0]
                    c_shift = firstArray[coord_index][1]
                    R_sub = R[r_shift:-1:8, c_shift:-1:8]
                    hist = np.histogram(R_sub, np.arange(0, T + 1))[0]
                    f_merged = np.add(f_merged, hist)
                F_index_from = (modeIndex * modeFeaDim) + (merged_index * (T + 1)) + 1
                F_index_to = (modeIndex * modeFeaDim) + (merged_index * (T + 1)) + T + 1
                F[F_index_from:F_index_to] = f_merged / sum(f_merged)

    return F


def predictLoadedImage(input_dctr):
    model = keras.models.load_model('dctr_model')
    return model.predict(np.array((input_dctr,)))


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
        # generate coeff tables
        print("Loading and processing file " + arg + "...")
        jpeginfo = jpeg_toolbox.load(arg)
        dctr = DCTR(jpeginfo)
        stegoPossibility = predictLoadedImage(dctr)
        print("# ==================== File " + arg + " ==================== #")
        print("Image resolution: " + str(jpeginfo["image_width"]) + "x" + str(jpeginfo["image_height"]))
        print("Image size: " + str(os.path.getsize(arg)) + " bytes")
        print("Photo is clean probability: " + str(((1-stegoPossibility[0][0])*100)) + "%")
        print("Photo has hidden data probability: " + str(stegoPossibility[0][0]*100) + "%")
        print("Check result: " + (stegoPossibility[0][0] > 0.5 and "hidden data detected" or "image clean"))
        print("# " + ("=" * (len(arg)+47)) + " #")
        sys.exit()
    else:
        print('JPEG-StegoChecker.py -f <inputfile>')
        sys.exit()
