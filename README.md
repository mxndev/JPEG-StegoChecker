# JPEG-StegoChecker

Python based tool, that has ability to check, if specific JPEG file has a chance to embedded steganography or not

## Minimal requirements
* Python 3.9.0 or higher
* Numpy 1.22.1 or higher
* Keras 2.4.3 or higher
* Tensorflow 2.5.0 or higher

## Usage
To run python script, you need to type
```sh
python stegochecker.py -f “stego_data/1.jpg”
```

### Example output:
```# ==================== File 1.jpg ==================== #
- Path: stego_data/1.jpg
- Image size: 587 B
- Photo is clean probability: 34%
- Photo has hidden data probability: 66%
- Check result: hidden data detected
# ================================================== #
```