import os
import sys
import lorann
import h5py
import time
import numpy as np
from urllib.request import build_opener, install_opener, urlretrieve

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
    quantization_bits = 8  # number of bits used to quantize the parameter matrices
    euclidean = True  # whether to use Euclidean distance

    ### LoRANN query hyperparameters
    clusters_to_search = 10
    points_to_rerank = 300  # number of points for exact re-ranking
elif sys.argv[1] == "sift":
    dataset = "sift-128-euclidean"

    ### LoRANN index hyperparameters
    n_clusters = 1024  # number of clusters, a good starting point is around sqrt(n)
    global_dim = None  # globally reduced dimension (s), None to turn off
    quantization_bits = 4  # number of bits used to quantize the parameter matrices
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

# if using cosine distance, make sure all vectors have unit norm
if not euclidean:
    train[np.linalg.norm(train, axis=1) == 0] = 1.0 / np.sqrt(train.shape[1])
    train /= np.linalg.norm(train, axis=1)[:, np.newaxis]

# initialize the LoRANN index
index = lorann.LorannIndex(
    data=train,
    n_clusters=n_clusters,
    global_dim=global_dim,
    quantization_bits=quantization_bits,
    euclidean=euclidean,
)

if os.path.isfile("%s.lorann" % sys.argv[1]):
    # load index from disk
    index = lorann.LorannIndex.load("%s.lorann" % sys.argv[1])
else:
    print("Building the index...")
    index.build()

    # serialize index to disk
    index.save("%s.lorann" % sys.argv[1])

print("Querying the index...")
start_time = time.time()
results, distances = index.search(
    test, k, clusters_to_search, points_to_rerank, return_distances=True
)
# results, distances = index.exact_search(test, k, return_distances=True)
end_time = time.time()

recall = lorann.compute_recall(results, f["neighbors"][:])
print("Recall:", recall)
print("Average query time (ms):", (end_time - start_time) / len(results) * 1e3)

# print(distances)
