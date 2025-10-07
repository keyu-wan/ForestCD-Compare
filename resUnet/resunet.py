# resunet.py
# A tunable ResUNet for forest change detection (Sentinel-2 multi-band, binary change)
import tensorflow as tf
from tensorflow.keras import layers as L
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import HeNormal

# ------------------------
# Blocks
# ------------------------
def conv_bn_act(x, filters, k=3, s=1, act="relu", bn=True, name=None):
    x = L.Conv2D(filters, k, strides=s, padding="same", kernel_initializer=HeNormal(),
                 use_bias=not bn, name=None if name is None else name + "_conv")(x)
    if bn:
        x = L.BatchNormalization(name=None if name is None else name + "_bn")(x)
    if act:
        x = L.Activation(act, name=None if name is None else name + "_act")(x)
    return x

def squeeze_excite(x, ratio=16, name=None):
    c = x.shape[-1]
    se = L.GlobalAveragePooling2D(name=None if name is None else name + "_gap")(x)
    se = L.Dense(max(c // ratio, 1), activation="relu",
                 name=None if name is None else name + "_fc1")(se)
    se = L.Dense(c, activation="sigmoid",
                 name=None if name is None else name + "_fc2")(se)
    se = L.Reshape((1,1,c))(se)
    return L.Multiply(name=None if name is None else name + "_scale")([x, se])

def residual_block(x_in, filters, s=1, act="relu", bn=True, preact=True, se_ratio=0, name=None):
    """
    Residual block with optional pre-activation and SE.
    """
    shortcut = x_in

    if preact:
        x = x_in
        if bn:
            x = L.BatchNormalization(name=None if name is None else name + "_pre_bn")(x)
        x = L.Activation(act, name=None if name is None else name + "_pre_act")(x)
        x = L.Conv2D(filters, 3, strides=s, padding="same",
                     kernel_initializer=HeNormal(),
                     use_bias=not bn,
                     name=None if name is None else name + "_conv1")(x)
        if bn:
            x = L.BatchNormalization(name=None if name is None else name + "_mid_bn")(x)
        x = L.Activation(act, name=None if name is None else name + "_mid_act")(x)
        x = L.Conv2D(filters, 3, strides=1, padding="same",
                     kernel_initializer=HeNormal(),
                     use_bias=not bn,
                     name=None if name is None else name + "_conv2")(x)
    else:
        x = conv_bn_act(x_in, filters, k=3, s=s, act=act, bn=bn, name=None if name is None else name + "_cba1")
        x = conv_bn_act(x,    filters, k=3, s=1, act=None, bn=bn, name=None if name is None else name + "_cba2")

    if se_ratio and se_ratio > 0:
        x = squeeze_excite(x, ratio=se_ratio, name=None if name is None else name + "_se")

    if (s != 1) or (shortcut.shape[-1] != filters):
        shortcut = L.Conv2D(filters, 1, strides=s, padding="same",
                            kernel_initializer=HeNormal(),
                            use_bias=not bn,
                            name=None if name is None else name + "_proj")(shortcut)
        if bn:
            shortcut = L.BatchNormalization(name=None if name is None else name + "_proj_bn")(shortcut)

    out = L.Add(name=None if name is None else name + "_add")([x, shortcut])
    return out

def upsample_block(x, skip, filters, up_mode="nearest", act="relu", bn=True, preact=True, se_ratio=0, name=None):
    """
    Decoder: upsample + concat skip + residual block
    up_mode: 'nearest' (UpSampling2D) or 'transposed' (Conv2DTranspose)
    """
    if up_mode == "transposed":
        x = L.Conv2DTranspose(filters, 2, strides=2, padding="same",
                              kernel_initializer=HeNormal(),
                              name=None if name is None else name + "_up")(x)
    else:
        x = L.UpSampling2D(size=(2,2), name=None if name is None else name + "_up")(x)
        x = conv_bn_act(x, filters, k=1, s=1, act=act, bn=bn, name=None if name is None else name + "_up_align")

    x = L.Concatenate(name=None if name is None else name + "_cat")([x, skip])
    x = residual_block(x, filters, s=1, act=act, bn=bn, preact=preact, se_ratio=se_ratio,
                       name=None if name is None else name + "_res")
    return x

# ------------------------
# Model builder
# ------------------------
class Config:
    input_shape      = (256, 256, 6)   # RGBx2
    num_classes      = 2               # softmax
    stem_filters     = 64
    encoder_filters  = [64, 128, 256, 512]
    decoder_filters  = [256, 128, 64]
    act              = "relu"
    bn               = True
    preact           = True
    se_ratio         = 0               # 0=off; 8/16=on
    up_mode          = "nearest"
    dropout_rate     = 0.0
    deep_supervision = False
    final_activation = "softmax"


def build_resunet_forest(cfg: Config):
    inputs = L.Input(shape=cfg.input_shape, name="input")

    # Stem
    x = conv_bn_act(inputs, cfg.stem_filters, k=3, s=1, act=cfg.act, bn=cfg.bn, name="stem1")
    x = conv_bn_act(x,      cfg.stem_filters, k=3, s=1, act=cfg.act, bn=cfg.bn, name="stem2")
    s1 = residual_block(x, cfg.encoder_filters[0], s=1, act=cfg.act, bn=cfg.bn,
                        preact=cfg.preact, se_ratio=cfg.se_ratio, name="enc1")

    # Encoder
    skips = [s1]
    x = residual_block(s1, cfg.encoder_filters[1], s=2, act=cfg.act, bn=cfg.bn,
                       preact=cfg.preact, se_ratio=cfg.se_ratio, name="enc2")
    skips.append(x)
    x = residual_block(x, cfg.encoder_filters[2], s=2, act=cfg.act, bn=cfg.bn,
                       preact=cfg.preact, se_ratio=cfg.se_ratio, name="enc3")
    skips.append(x)


    if len(cfg.encoder_filters) == 4:
        x = residual_block(x, cfg.encoder_filters[3], s=2, act=cfg.act, bn=cfg.bn,
                        preact=cfg.preact, se_ratio=cfg.se_ratio, name="enc4")
        bridge_in = x
        used_skips = [skips[2], skips[1], skips[0]]
        dec_filters = cfg.decoder_filter
    else:
        bridge_in = x
        used_skips = [skips[1], skips[0]]
        dec_filters = cfg.decoder_filters

    # Bridge
    b = residual_block(bridge_in, bridge_in.shape[-1], s=1,
                       act=cfg.act, bn=cfg.bn, preact=cfg.preact, se_ratio=cfg.se_ratio, name="bridge")

    if cfg.dropout_rate and cfg.dropout_rate > 0:
        b = L.Dropout(cfg.dropout_rate, name="bridge_dropout")(b)

    # Decoder
    x = b
    for i, (sk, f) in enumerate(zip(used_skips, dec_filters), start=1):
        x = upsample_block(x, sk, f, up_mode=cfg.up_mode, act=cfg.act, bn=cfg.bn,
                           preact=cfg.preact, se_ratio=cfg.se_ratio, name=f"dec{i}")

    # Head
    logits = L.Conv2D(cfg.num_classes, 1, padding="same",
                    kernel_initializer=HeNormal(), name="logits")(x)
    outputs = L.Activation(cfg.final_activation, name="pred", dtype='float32')(logits)


    if cfg.deep_supervision:
        aux = []
        aux1 = L.Conv2D(cfg.num_classes, 1, padding="same",
                        kernel_initializer=HeNormal(), name="aux1")(x)
        aux1 = L.Activation(cfg.final_activation, name="aux1_pred")(aux1)
        aux.append(aux1)
        model = Model(inputs, [outputs] + aux, name="ResUNet_Forest_DS")
    else:
        model = Model(inputs, outputs, name="ResUNet_Forest")

    return model


def build_res_unet(input_shape, nClasses=2, **kwargs):
    cfg = Config()
    cfg.input_shape = input_shape
    cfg.num_classes = nClasses

    for k, v in kwargs.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)

    depth = kwargs.get("depth", None)
    if depth is not None:
        if depth == 4:
            cfg.encoder_filters = [64, 128, 256, 512]
            cfg.decoder_filters = [256, 128, 64]
        elif depth == 3:
            cfg.encoder_filters = [64, 128, 256]
            cfg.decoder_filters = [128, 64]

    if kwargs.get("act", None) == "gelu":
        try:
            _ = L.Activation("gelu")
        except Exception:
            cfg.act = "relu"

    return build_resunet_forest(cfg)
