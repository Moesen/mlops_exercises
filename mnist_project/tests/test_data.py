from tests import _SRC_PATH, _PATH_DATA
import sys
sys.path.append(_SRC_PATH)

from models.dataset import CorruptMNISTDataset
import os

data_path = os.path.join(_PATH_DATA, "processed/corrupt_mnist.pt")
train_data = CorruptMNISTDataset(data_path, train=True)
test_data = CorruptMNISTDataset(data_path, train=False)

def test_dataset_len():
    n_train_size = 40000
    n_test_size = 5000

    assert n_train_size == len(train_data)
    assert n_test_size == len(test_data)
    

def test_data_shape():
    assert list(next(iter(train_data.x)).shape) == [28, 28]

def test_labels():
    assert list(train_data.Y.unique()) == [0,1,2,3,4,5,6,7,8,9]
    assert list(test_data.Y.unique()) == [0,1,2,3,4,5,6,7,8,9]