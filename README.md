# dcmtrans

## Install
### Via pip

Not Implemeted yet
```sh
$ pip install git+http://remote/git/repo/dcmtrans.git
```

### Via git-submodule
```sh
$ git submodule add -b <dcmtrans branch/tag name> -- http://remote/git/repo/dcmtrans.git path/to/put/dcmtrans
$ git status
...
modified: .gitmodule
new file: path/to/put/dcmtrans
...
$ git commit
```

## Examples

```python
import pydicom
import numpy as np
from dcmtrans import dcmtrans

def read_dicom_image(filename):
    # read dicom file by pydicom
    dcmobj = pydicom.dcmread(filename, force=True)
    
    # read PixelData: return None when all methods are fail
    image_array = dcmtrans.read_pixel(filename, return_on_fail=None)
    
    # deal with fail case
    if image_array is None:
        return np.zeros([1, int(dcmobj.get('Rows', 1)), int(dcmobj.get('Columns', 1))])
        
    # apply modality, value of interest, photometric interpretation transform to row image (np.array) 
    imgs, excepts, info = dcmtrans.dcmtrans(dcmobj, image_array, window=['default'], depth=256)
    
    # deal with fail case
    image_np = imgs[0]
    if image_np is None:
        return np.zeros([1, int(dcmobj.get('Rows', 1)), int(dcmobj.get('Columns', 1))])
        
    # reshape
    image_np = np.array(image_np, dtype=np.uint8)
    dim = len(image_np.shape)
    if dim < 3:
        for _ in range(3 - dim):
            image_np = np.expand_dims(image_np, 0)
    return image_np
```
