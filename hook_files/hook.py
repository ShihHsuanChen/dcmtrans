from PyInstaller.utils.hooks import copy_metadata


hiddenimports = [
    'dcmtrans',
    'dcmtrans.reconstruction',
    'dcmtrans.trans_method',
]

datas = copy_metadata('pylibjpeg')
datas += copy_metadata('pylibjpeg_libjpeg')
