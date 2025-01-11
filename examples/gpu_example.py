import os
import sys
import h5py
import time
import numpy as np
from urllib.request import build_opener, install_opener, urlretrieve

from lorann import compute_recall
from lorann_gpu.jax import Lorann

opener = build_opener()
opener.addheaders = [("User-agent", "Mozilla/5.0")]
install_opener(opener)


def download(source_url, destination_path):
    if not os.path.exists(destination_path):
        print(f"downloading {source_url} -> {destination_path}...")
        urlretrieve(source_url, destination_path)


k = 100  # number of nearest neighbors to search for

if len(sys.argv) == 1:
    print("Usage: python %s dataset" % sys.argv[0])
    sys.exit(1)

if sys.argv[1] == "fashion-mnist":
    dataset = "fashion-mnist-784-euclidean"

    ### LoRANN index hyperparameters
    n_clusters = 256  # number of clusters, a good starting point is around sqrt(n)
    global_dim = 128  # globally reduced dimension (s), None to turn off
    euclidean = True  # whether to use Euclidean distance

    ### LoRANN query hyperparameters
    clusters_to_search = 10
    points_to_rerank = 300  # number of points for exact re-ranking
elif sys.argv[1] == "sift":
    dataset = "sift-128-euclidean"

    ### LoRANN index hyperparameters
    n_clusters = 1024  # number of clusters, a good starting point is around sqrt(n)
    global_dim = None  # globally reduced dimension (s), None to turn off
    euclidean = True  # whether to use Euclidean distance

    ### LoRANN query hyperparameters
    clusters_to_search = 32
    points_to_rerank = 800  # number of points for exact re-ranking
else:
    raise RuntimeError("Invalid data set. Possible options: fashion-mnist, sift")

download("http://ann-benchmarks.com/%s.hdf5" % dataset, "%s.hdf5" % dataset)

# read the data set
f = h5py.File("%s.hdf5" % dataset, "r")
train = f["train"][:]
test = f["test"][:]

# if using angular distance, make sure all vectors have unit norm
if not euclidean:
    train[np.linalg.norm(train, axis=1) == 0] = 1.0 / np.sqrt(train.shape[1])
    train /= np.linalg.norm(train, axis=1)[:, np.newaxis]

# build the LoRANN index
print("Building the index...")
index = Lorann.build(  # for other GPU backends, use Lorann(...) instead of Lorann.build(...)
    data=train,
    n_clusters=n_clusters,
    global_dim=global_dim,
    euclidean=euclidean,
)

# we do a query before-hand for Jax to compile the search function
_ = index.search(test, k, clusters_to_search, points_to_rerank)

print("Querying the index...")
start_time = time.time()
results = index.search(test, k, clusters_to_search, points_to_rerank)
end_time = time.time()

recall = compute_recall(results, f["neighbors"][:])
print("Recall:", recall)
print("Average query time (ms):", (end_time - start_time) / len(results) * 1e3)
