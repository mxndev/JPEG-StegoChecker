# JPEG-StegoChecker

Python based tool, that has ability to check, if specific JPEG file has a chance to embedded steganography or not

## Minimal requirements
* Python 3.9.0 or higher
* Numpy 1.22.1 or higher
* Keras 2.4.3 or higher
* Tensorflow 2.5.0 or higher
* SciPy 1.7.0 or higher

Also there is need to install JPEG Toolbox for Python library available here:
```
https://github.com/daniellerch/python-jpeg-toolbox
```

In project folder there is prepared version of this library ready to install by command:
```
cd python-jpeg-toolbox
pip install .
```
or downloading it directly from repository:
```
pip3 install git+https://github.com/daniellerch/python-jpeg-toolbox --user
```

## Usage
To run python script, you need to type
```
python stegochecker.py -f “stego_data/1.jpg”
```

### Example output:
```
# ==================== File 1.jpg ==================== #
- Image resolution: 240x320
- Image size: 587 bytes
- Photo is clean probability: 34.4545676%
- Photo has hidden data probability: 65.5454324%
- Check result: hidden data detected
# ==================================================== #
```