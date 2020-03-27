import os
import shutil
import zipfile

import numpy as np
import requests


def process_mnist_rot():
    basepath = os.path.dirname(__file__)
    url = "http://www.iro.umontreal.ca/~lisa/icml2007data/mnist_rotation_new.zip"
    outfile = "mnist-rot.zip"
    outpath = f"{basepath}/{outfile}"

    if not os.path.exists(outpath):
        print("Downloading %s..." % outfile)
        r = requests.get(url, stream=True)
        with open(outpath, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
    else:
        print("%s already downloaded..." % outfile)

    print("Unpacking...")
    zip_ref = zipfile.ZipFile(outpath, 'r')
    zip_ref.extractall(f"{basepath}/mnist-rot/")
    zip_ref.close()

    train = np.loadtxt(f'{basepath}/mnist-rot/mnist_all_rotation_normalized_float_train_valid.amat')
    train_img = train[:, :28 * 28]
    train_lbl = train[:, -1, None].astype('int')

    test = np.loadtxt(f'{basepath}/mnist-rot/mnist_all_rotation_normalized_float_test.amat')
    test_img = test[:, :28 * 28]
    test_lbl = test[:, -1, None].astype('int')

    np.savez(f'{basepath}/mnist-rot.npz', X=train_img, Y=train_lbl, Xt=test_img, Yt=test_lbl)

    shutil.rmtree(f'{basepath}/mnist-rot')
    os.remove(outpath)
