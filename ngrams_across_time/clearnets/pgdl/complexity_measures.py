import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.utils import check_X_y, _safe_indexing

from ngrams_across_time.clearnets.pgdl.intermediate_outputs import intermediateOutputs


def get_generalization_score(model, dataset):
    return complexityDB(
        model, dataset, pool=True, computeOver=400, batchSize=40
    ) * (
        1 - complexityMixup(model, dataset)
    )


def complexityDB(
    model,
    dataset,
    pool=True,
    use_pca=False,
    layer="initial",
    computeOver=400,
    batchSize=40,
):
    """
    Function to calculate feature clustering based measures. Based on the sklearn implementation of DB Index.

    Parameters
    ----------
    model : keras.Model()
            The Keras model for which the complexity measure is to be computed
    dataset : data.Dataset
            Dataset object from PGDL data loader
    pool : bool, optional
            Whether to use max-pooling for dimensionality reduction, default True
    use_pca : bool, optional
            Whether to use PCA for dimensionality reduction, default False
    layer : str or int, optional
            Which layer to compute DB on. Either 'initial', for the first conv/pooling layer in the
            model, 'pre-penultimate' for the 3rd-from-last conv/pool layer, or an int indicating the
            layer. Defaults to 'initial'.

    Returns
    -------
    float
            complexity measure
    """

    def db(X, labels):
        X, labels = check_X_y(X, labels)
        le = LabelEncoder()
        labels = le.fit_transform(labels)
        n_samples, _ = X.shape
        n_labels = len(le.classes_)

        assert 1 < n_labels < n_samples, (
            "Number of labels is %d. Valid values are 2 to n_samples - 1 (inclusive)"
            % n_labels
        )

        intra_dists = np.zeros(n_labels)
        centroids = np.zeros((n_labels, len(X[0])), dtype=float)
        for k in range(n_labels):
            cluster_k = _safe_indexing(X, labels == k)
            centroid = cluster_k.mean(axis=0)
            centroids[k] = centroid
            intra_dists[k] = np.average(
                pairwise_distances(cluster_k, [centroid], metric="euclidean")
            )

        centroid_distances = pairwise_distances(centroids, metric="euclidean")

        if np.allclose(intra_dists, 0) or np.allclose(centroid_distances, 0):
            return 0.0

        centroid_distances[centroid_distances == 0] = np.inf
        combined_intra_dists = intra_dists[:, None] + intra_dists
        scores = np.max(combined_intra_dists / centroid_distances, axis=1)
        return np.mean(scores)

    db_score = {}
    it = iter(dataset.repeat(-1).batch(batchSize))
    batch = next(it)
    extractor = intermediateOutputs(model, batch=batch)
    if pool == True:
        max_pool = tf.keras.layers.MaxPooling2D(
            pool_size=(4, 4), strides=None, padding="valid", data_format=None
        )
    else:
        max_pool = tf.keras.layers.Lambda(lambda x: x + 0)
    layers = []

    layer_dict = {"initial": [0, 1, 2], "pre-penultimate": [-3, -4, -5]}

    if isinstance(layer, str):
        for l in layer_dict[layer]:
            c = list(model.get_layer(index=l).get_config().keys())
            if "strides" in c:
                layers.append(l)
            if len(layers) == 1:
                break
    else:
        for l in [layer]:
            c = list(model.get_layer(index=l).get_config().keys())
            if "strides" in c:
                layers.append(l)
            if len(layers) == 1:
                break

    for l in layers:
        it = iter(dataset.repeat(-1).shuffle(5000, seed=1).batch(batchSize))
        for i in range(computeOver // batchSize):
            batch1 = next(it)
            batch2 = next(it)
            batch3 = next(it)
            feature = np.concatenate(
                (
                    max_pool(extractor(batch1[0].numpy())[l])
                    .numpy()
                    .reshape(batch1[0].shape[0], -1),
                    max_pool(extractor(batch2[0].numpy())[l])
                    .numpy()
                    .reshape(batch2[0].shape[0], -1),
                    max_pool(extractor(batch3[0].numpy())[l])
                    .numpy()
                    .reshape(batch3[0].shape[0], -1),
                ),
                axis=0,
            )
            target = np.concatenate((batch1[1], batch2[1], batch3[1]), axis=0)
            if use_pca == True:
                pca = PCA(n_components=25)
                feature = pca.fit_transform(feature)
            try:
                db_score[l] += db(feature, target) / (computeOver // batchSize)
            except Exception as e:
                db_score[l] = db(feature, target) / (computeOver // batchSize)

    score = np.mean(list(db_score.values()))

    return score


def complexityMixup(model, dataset, computeOver=500, batchSize=50):
    """
    Function to calculate label-wise Mixup based measure

    Parameters
    ----------
    model : tf.keras.Model()
            The Keras model for which the complexity measure is to be computed
    dataset : tf.data.Dataset
            Dataset object from PGDL data loader
    computeOver : int
            The number of samples over which to compute the complexity measure
    batchSize:
            The batch size

    Returns
    -------
    float
            complexity measure
    """
    it = iter(dataset.repeat(-1).shuffle(5000, seed=1).batch(batchSize))
    batch = next(it)
    n_classes = 1 + np.max(batch[1].numpy())
    batchSize = n_classes * 10
    computeOver = batchSize * 10
    it = iter(dataset.repeat(-1).batch(batchSize))
    N = computeOver // batchSize
    batches = [next(it) for i in range(N)]
    vr = []

    def intrapolateImages(img, alpha=0.5):
        temp = np.stack([img] * img.shape[0])
        tempT = np.transpose(temp, axes=(1, 0, 2, 3, 4))
        ret = alpha * temp + (1 - alpha) * tempT
        mask = np.triu_indices(img.shape[0], 1)
        return ret[mask]

    def veracityRatio(model, batches, label, version_loss=None, label_smoothing=0.1):
        ret = []
        lossObject = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        for b in batches:
            img = b[0][b[1] == label]
            lbl = b[1][b[1] == label]
            int_img = intrapolateImages(img)
            int_lbl = np.stack([label] * int_img.shape[0])
            int_logits = model(int_img)
            if version_loss == "log":
                logLikelihood = lossObject(int_lbl, int_logits)
                ret.append(logLikelihood)
            elif version_loss == "cosine":
                int_preds = tf.nn.softmax(int_logits, axis=1)
                target = (
                    tf.one_hot(int_lbl, int_preds.shape[-1]) * (1 - label_smoothing)
                    + label_smoothing / 2
                )
                ret.append(
                    (tf.keras.losses.CosineSimilarity()(target, int_preds) + 1) / 2
                )
            elif version_loss == "mse":
                int_preds = tf.nn.softmax(int_logits, axis=1)
                target = tf.one_hot(
                    int_lbl, int_preds.shape[-1]
                )  # * (1 - label_smoothing) + label_smoothing/2
                ret.append(tf.keras.losses.MeanSquaredError()(target, int_preds))
            else:
                int_preds = tf.argmax(int_logits, axis=1)
                ret.append(np.sum(int_preds == label) / np.size(int_preds))
        return np.mean(ret)

    for l in range(n_classes):
        try:
            vr.append(veracityRatio(model, batches, l))
        except:
            pass

    return np.mean(vr)
