"""
ATAF dataset/model registry + generic loaders.

Mirrors /home/siddarth/ADBA/run_batch.py's DATASET_CONFIGS / MODEL_PATHS, plus
generic loaders that work for any entry in either table. Shared by both the
sequential and batched BruSLeAttack demo notebooks so this isn't duplicated.
"""
import zipfile
import csv
import io
import torch
from PIL import Image
import torchvision.transforms as T

DATASET_CONFIGS = {
    'IMAGENET_3599': {
        'size': 224,
        'mean': [0.485, 0.456, 0.406],
        'std':  [0.229, 0.224, 0.225],
        'n_class': 1000,
        'zip_path': "/home/HDD/ATAF/Datasets/ImageNetDataset/ATAF-Framework-Ready/ImageNet-3599-Targeted.zip",
        'filename_col': 'filename', 'label_col': 'original_label', 'target_col': 'target_label', 'has_header': True,
    },
    'CIFAR': {
        'size': 32,
        'mean': [0.4914, 0.4822, 0.4465],
        'std':  [0.2023, 0.1994, 0.2010],
        'n_class': 10,
        'zip_path': "/home/HDD/ATAF/Datasets/CIFAR10-Dataset/CIFAR-10-60k-targeted.zip",
        'filename_col': 'image_name', 'label_col': 'class_label', 'target_col': 'target_label', 'has_header': True,
    },
    'GTSRB': {
        'size': 32,
        'mean': [0.3403, 0.3121, 0.3214],
        'std':  [0.2724, 0.2608, 0.2669],
        'n_class': 43,
        'zip_path': "/home/HDD/ATAF/Datasets/GTSRB/GTSRB_test_ataf.zip",
        # this zip's labels.csv has no header row and no target_label column -> untargeted attacks only
        'filename_col': None, 'label_col': None, 'target_col': None, 'has_header': False,
    },
    'EMNIST': {
        'size': 28,
        'mean': [0.17510417543456253] * 3,   # computed from the downloaded balanced-train split
        'std':  [0.3332371468327344] * 3,
        'n_class': 47,
        'zip_path': "/home/siddarth/BruSLiAttack/Datasets/EMNIST/EMNIST-balanced.zip",
        # target_label: randomly assigned per image (uniform over the other 46
        # classes, seed=10), same convention as CIFAR/ImageNet's target_label columns
        'filename_col': 'image_name', 'label_col': 'class_label', 'target_col': 'target_label', 'has_header': True,
    },
}

MODEL_ARCH_DATASET = {
    'resnet50': 'IMAGENET_3599', 'DeiT_T': 'IMAGENET_3599', 'DeiT_S': 'IMAGENET_3599', 'DeiT_B': 'IMAGENET_3599',
    'resnet18_cifar10': 'CIFAR', 'resnet50_cifar10': 'CIFAR', 'deit_cifar10': 'CIFAR',
    'resnet50_gtsrb32': 'GTSRB',
    'resnet18_emnist': 'EMNIST',
}

MODEL_PATHS = {
    'resnet50':         "/home/HDD/ATAF/Model-files/working_model_files/cnn/torch/resnet50_testing_model.pth",
    'resnet18_cifar10': '/home/HDD/ATAF/Model-files/working_model_files/cnn/torch/cifar10/resnet18_cifar10.pth',
    'resnet50_cifar10': '/home/HDD/ATAF/Model-files/working_model_files/cnn/torch/cifar10/resnet50_cifar10.pth',
    'deit_cifar10':     '/home/HDD/ATAF/Model-files/working_model_files/cnn/torch/cifar10/deit_cifar10_epoch_0070_ataf_ready.pth',
    'resnet50_gtsrb32': '/home/HDD/ATAF/Model-files/GTSRB/gtsrb_resnet50_32.pth',
    'DeiT_T':           '/home/HDD/ATAF/Model-files/Patchfool/pytorch_models/deit_variations/deit_tiny_patch16_224_ataf_ready.pth',
    'DeiT_S':           '/home/HDD/ATAF/Model-files/Patchfool/pytorch_models/deit_variations/deit_small_patch16_224_ataf_ready.pth',
    'DeiT_B':           '/home/HDD/ATAF/Model-files/Patchfool/pytorch_models/deit_variations/deit_base_patch16_224_ataf_ready.pth',
    'resnet18_emnist':  '/home/siddarth/BruSLiAttack/Models/resnet18_emnist_balanced.pth',
}


class GenericPretrainedModel:
    """Same role as utils.PretrainedModel, but takes mean/std directly instead
    of matching against a hardcoded dataset-name string."""
    def __init__(self, model, mean, std):
        self.model = model
        self.mu = torch.tensor(mean).float().view(1, 3, 1, 1).cuda()
        self.sigma = torch.tensor(std).float().view(1, 3, 1, 1).cuda()

    def predict(self, x):
        return self.model((x - self.mu) / self.sigma)

    def predict_label(self, x):
        return torch.max(self.predict(x), 1)[1]

    def __call__(self, x):
        return self.predict(x)


def load_model_generic(model_arch, device='cuda'):
    # checkpoints are full pickled nn.Modules, not state_dicts (see run_batch.py: load_model)
    net = torch.load(MODEL_PATHS[model_arch], map_location=device, weights_only=False)
    net = net.to(device)
    net.eval()
    return net


_LABELS_CACHE = {}
_FILELIST_CACHE = {}
_ZIPFILE_CACHE = {}

def _get_zipfile(cfg):
    """Persistent, reused zipfile.ZipFile per zip_path. Constructing a
    ZipFile isn't free -- it parses the archive's entire central directory
    (every entry's name/offset/size) up front, an O(n) cost paid on every
    single construction. load_indexed_image_and_labels used to reopen the
    zip fresh per image, so loading N images re-parsed the whole directory
    N times. Negligible for ImageNet's 3599-entry zip, but for CIFAR's
    60001 entries this was the dominant cost at any real batch size (300
    images took 90+s; 10000 would have been ~45-50 min). Never explicitly
    closed -- these are read-only handles reused for the life of the
    process, which is fine for a script/notebook, not a long-running server."""
    if cfg['zip_path'] not in _ZIPFILE_CACHE:
        _ZIPFILE_CACHE[cfg['zip_path']] = zipfile.ZipFile(cfg['zip_path'])
    return _ZIPFILE_CACHE[cfg['zip_path']]


def _get_sorted_image_files(cfg):
    """Sorted images/*.{jpg,jpeg,png} filenames for a dataset's zip, cached
    per zip_path so the full namelist isn't re-listed+sorted on every call."""
    if cfg['zip_path'] in _FILELIST_CACHE:
        return _FILELIST_CACHE[cfg['zip_path']]
    z = _get_zipfile(cfg)
    image_files = sorted(
        f for f in z.namelist()
        if f.startswith('images/') and f.lower().endswith(('.jpeg', '.jpg', '.png'))
    )
    _FILELIST_CACHE[cfg['zip_path']] = image_files
    return image_files


def _read_labels_csv(cfg):
    if cfg['zip_path'] in _LABELS_CACHE:
        return _LABELS_CACHE[cfg['zip_path']]
    lookup = {}
    z = _get_zipfile(cfg)
    with z.open('labels.csv') as f:
        text = io.TextIOWrapper(f, encoding='utf-8')
        if cfg['has_header']:
            for row in csv.DictReader(text):
                fn = row[cfg['filename_col']]
                ol = int(row[cfg['label_col']])
                tl = int(row[cfg['target_col']]) if cfg['target_col'] else None
                lookup[fn] = (ol, tl)
        else:
            for line in text:
                fn, ol = line.strip().split(',')
                lookup[fn] = (int(ol), None)
    _LABELS_CACHE[cfg['zip_path']] = lookup
    return lookup


def load_indexed_image_and_labels(dataset_key, index):
    """Loads the `index`-th image (sorted filename order) from the dataset's
    zip, plus its (original_label, target_label) from labels.csv.
    target_label is None where the dataset doesn't provide one (e.g. GTSRB)."""
    cfg = DATASET_CONFIGS[dataset_key]
    fname = _get_sorted_image_files(cfg)[index]
    z = _get_zipfile(cfg)
    with z.open(fname) as f:
        img = Image.open(f).convert('RGB').resize((cfg['size'], cfg['size']))
        x0 = T.ToTensor()(img).unsqueeze(0)  # [1,3,H,W], range [0,1]

    short_name = fname.replace('images/', '')
    olabel, tlabel = _read_labels_csv(cfg)[short_name]
    return x0, olabel, tlabel, fname


def load_batch_images_and_labels(dataset_key, start, n):
    imgs, olabels, tlabels, fnames = [], [], [], []
    for i in range(start, start + n):
        x0, ol, tl, fn = load_indexed_image_and_labels(dataset_key, i)
        imgs.append(x0); olabels.append(ol); tlabels.append(tl); fnames.append(fn)
    imgs = torch.cat(imgs, dim=0)
    olabels = torch.tensor(olabels)
    tlabels = torch.tensor(tlabels) if all(t is not None for t in tlabels) else None
    return imgs, olabels, tlabels, fnames


# ============================================================================
# TensorFlow registry + loaders (additions -- everything above is untouched).
#
# `tensorflow` is imported lazily (only inside the functions/methods below,
# never at module level), so importing ataf_registry.py for the existing
# torch pipeline still works even in an environment without TensorFlow
# installed, and doesn't pay any import-time cost for it either.
#
# DATASET_CONFIGS / MODEL_ARCH_DATASET / labels.csv reading are reused as-is
# from above -- none of that is framework-specific. Only image decoding
# (NHWC numpy instead of NCHW torch) and model loading/wrapping differ.
# ============================================================================


_tf = None

def _get_tf():
    global _tf
    if _tf is None:
        import tensorflow as tf
        _tf = tf
    return _tf


# Same keys as MODEL_PATHS, pointing at TensorFlow checkpoints instead of
# PyTorch ones. Fill these in once the TF model files are available --
# load_model_generic_tf() raises a clear error for any key still None.
MODEL_PATHS_TF = {
    'resnet50':        "/home/HDD/ATAF/Model-files/working_model_files/cnn/tensorflow/resnet50_tf_complete.h5",
    'resnet18_cifar10': None,
    # verified: generic (x-mean)/std normalization gives 77% clean accuracy on
    # real CIFAR10 data (checked against raw/[-1,1]/caffe alternatives too --
    # this is clearly the right one, not a guess)
    'resnet50_cifar10': "/home/HDD/ATAF/Model-files/working_model_files/cnn/tensorflow/cifar10_tf/resnet50_cifar10_best_ataf.h5",
    'deit_cifar10':     None,
    'resnet50_gtsrb32': None,
    'DeiT_T':           None,
    'DeiT_S':           None,
    'DeiT_B':           None,
    # NOT YET TRAINED: this path is where train_emnist_resnet18_tf.py will save
    # its checkpoint once run -- tf.keras.models.load_model() will fail with a
    # clear error here until training actually completes.
    'resnet18_emnist':  '/home/siddarth/BruSLiAttack/Models/resnet18_emnist_balanced_tf.h5',
}


# Declarative preprocessing config per model_arch, same style as
# DATASET_CONFIGS -- covers tf.keras.applications' three standard
# preprocessing conventions via 'mode', so adding a new TF model is just a
# dict entry, not a new bespoke function:
#   'caffe' -- RGB->BGR, scale [0,1]->[0,255], subtract a fixed per-channel
#              mean (no std division). ResNet/VGG-family Keras models.
#   'tf'    -- scale to [-1, 1], no channel flip, no mean/std. Inception/
#              Xception/MobileNet-family Keras models.
#   'torch' -- scale [0,1], normalize by ImageNet mean/std (RGB) -- the same
#              convention the torch pipeline's GenericPretrainedModel uses.
# 'mean'/'std' are optional per-arch overrides; each mode falls back to its
# standard ImageNet default if omitted. An arch with no entry here just uses
# GenericPretrainedModelTF's default (x - mean)/std normalization directly.
PREPROCESS_CONFIG_TF = {
    'resnet50': {'mode': 'caffe'},
}


def _tf_preprocess(x, mode, mean=None, std=None):
    tf = _get_tf()
    if mode == 'caffe':
        m = tf.constant(mean or [103.939, 116.779, 123.68], dtype=tf.float32)
        return x[..., ::-1] * 255.0 - m
    if mode == 'tf':
        return x * 2.0 - 1.0
    if mode == 'torch':
        m = tf.constant(mean or [0.485, 0.456, 0.406], dtype=tf.float32)
        s = tf.constant(std or [0.229, 0.224, 0.225], dtype=tf.float32)
        return (x - m) / s
    raise ValueError(f"Unknown preprocess mode {mode!r} -- expected 'caffe', 'tf', or 'torch'.")


def get_preprocess_fn_tf(model_arch):
    """Builds the preprocessing callable for `model_arch` from
    PREPROCESS_CONFIG_TF, or None if the arch has no entry there (in which
    case GenericPretrainedModelTF falls back to its own mean/std
    normalization). Pass the result as GenericPretrainedModelTF(...,
    preprocess_fn=get_preprocess_fn_tf(model_arch))."""
    cfg = PREPROCESS_CONFIG_TF.get(model_arch)
    if cfg is None:
        return None
    return lambda x: _tf_preprocess(x, cfg['mode'], cfg.get('mean'), cfg.get('std'))


def load_model_generic_tf(model_arch):
    """Loads a TensorFlow model for `model_arch` from MODEL_PATHS_TF, and
    strips a trailing softmax if present so the returned model outputs
    logits -- feval_score expects logits (it calls cross-entropy with
    from_logits=True, which would double-apply softmax on an
    already-softmaxed output).

    ASSUMPTION (update this if it doesn't match your actual files): paths
    are loadable via tf.keras.models.load_model -- i.e. a SavedModel
    directory or a single .h5/.keras file. If the checkpoints are a
    different format (raw weights needing separate architecture code,
    ONNX, etc.) this function needs to change accordingly.
    """
    tf = _get_tf()
    path = MODEL_PATHS_TF.get(model_arch)
    if path is None:
        raise ValueError(f"No TensorFlow model path set for '{model_arch}' yet -- "
                          f"fill in MODEL_PATHS_TF['{model_arch}'] in ataf_registry.py.")

    model = tf.keras.models.load_model(path)
    last = model.layers[-1]
    if getattr(last, 'activation', None) is tf.keras.activations.softmax:
        weights = model.get_weights()
        last.activation = tf.keras.activations.linear
        model = tf.keras.models.model_from_json(model.to_json())
        model.set_weights(weights)
    return model


class GenericPretrainedModelTF:
    """TensorFlow/Keras equivalent of GenericPretrainedModel: same role
    (normalize before the wrapped model), but for NHWC (channels-last)
    tensors -- TF/Keras' native layout, vs. the torch version's NCHW.

    By default applies generic (x - mean)/std normalization like the torch
    version. Pass `preprocess_fn` (e.g. from PREPROCESS_FN_TF[model_arch])
    to use a model-specific preprocessing function instead -- required for
    checkpoints like tf.keras.applications.ResNet50 that expect their own
    specific channel order / scale / mean convention.
    """

    def __init__(self, model, mean, std, preprocess_fn=None):
        tf = _get_tf()
        self.model = model
        self.preprocess_fn = preprocess_fn
        self.mu = tf.constant(mean, dtype=tf.float32)[None, None, None, :]
        self.sigma = tf.constant(std, dtype=tf.float32)[None, None, None, :]

    def predict(self, x):
        tf = _get_tf()
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        xp = self.preprocess_fn(x) if self.preprocess_fn is not None else (x - self.mu) / self.sigma
        return self.model(xp, training=False)

    def predict_label(self, x):
        tf = _get_tf()
        return tf.argmax(self.predict(x), axis=1)

    def __call__(self, x):
        return self.predict(x)


def load_indexed_image_and_labels_tf(dataset_key, index):
    """TensorFlow-side equivalent of load_indexed_image_and_labels: same
    zip/labels.csv reading as the torch version, but returns a numpy array
    in NHWC layout ([1,H,W,3], range [0,1]) instead of an NCHW torch tensor."""
    import numpy as np
    cfg = DATASET_CONFIGS[dataset_key]
    fname = _get_sorted_image_files(cfg)[index]
    z = _get_zipfile(cfg)
    with z.open(fname) as f:
        img = Image.open(f).convert('RGB').resize((cfg['size'], cfg['size']))
        x0 = (np.asarray(img, dtype=np.float32) / 255.0)[None, ...]  # [1,H,W,3]

    short_name = fname.replace('images/', '')
    olabel, tlabel = _read_labels_csv(cfg)[short_name]
    return x0, olabel, tlabel, fname


def load_batch_images_and_labels_tf(dataset_key, start, n):
    import numpy as np
    imgs, olabels, tlabels, fnames = [], [], [], []
    for i in range(start, start + n):
        x0, ol, tl, fn = load_indexed_image_and_labels_tf(dataset_key, i)
        imgs.append(x0); olabels.append(ol); tlabels.append(tl); fnames.append(fn)
    imgs = np.concatenate(imgs, axis=0)
    olabels = np.array(olabels, dtype=np.int64)
    tlabels = np.array(tlabels, dtype=np.int64) if all(t is not None for t in tlabels) else None
    return imgs, olabels, tlabels, fnames
