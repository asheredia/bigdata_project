import tensorflow as tf
from tensorflow.keras import layers, Model


# =========================================================
# Squeeze-and-Excitation Block
# =========================================================
def se_block(x, reduction=8):

    filters = x.shape[-1]

    se = layers.GlobalAveragePooling1D()(x)

    se = layers.Dense(filters // reduction,
                      activation='relu')(se)

    se = layers.Dense(filters,
                      activation='sigmoid')(se)

    se = layers.Reshape((1, filters))(se)

    x = layers.Multiply()([x, se])

    return x


# =========================================================
# Residual Block 1D
# =========================================================
def residual_block_1d(
        x,
        filters,
        stride=1,
        use_se=False,
        dropout_rate=0.0):

    shortcut = x

    # -----------------------------
    # Conv 1
    # -----------------------------
    x = layers.Conv1D(
        filters=filters,
        kernel_size=3,
        strides=stride,
        padding='same',
        use_bias=False
    )(x)

    x = layers.BatchNormalization()(x)

    x = layers.Activation("gelu")(x)

    # -----------------------------
    # Conv 2
    # -----------------------------
    x = layers.Conv1D(
        filters=filters,
        kernel_size=3,
        strides=1,
        padding='same',
        use_bias=False
    )(x)

    x = layers.BatchNormalization()(x)

    # -----------------------------
    # Optional SE block
    # -----------------------------
    if use_se:
        x = se_block(x)

    # -----------------------------
    # Shortcut projection
    # -----------------------------
    if stride != 1 or shortcut.shape[-1] != filters:

        shortcut = layers.Conv1D(
            filters=filters,
            kernel_size=1,
            strides=stride,
            padding='same',
            use_bias=False
        )(shortcut)

        shortcut = layers.BatchNormalization()(shortcut)

    # -----------------------------
    # Residual connection
    # -----------------------------
    x = layers.Add()([x, shortcut])

    x = layers.Activation("gelu")(x)

    # -----------------------------
    # Optional dropout
    # -----------------------------
    if dropout_rate > 0:
        x = layers.Dropout(dropout_rate)(x)

    return x


# =========================================================
# ResNet18-1D for IQ signals
# =========================================================
def ResNet18_1D(
        input_shape=(1024, 2),
        num_classes=40,
        use_se=False,
        dropout_rate=0.1, 
        name="resnet18"):

    inputs = layers.Input(shape=input_shape)

    # =====================================================
    # STEM
    # =====================================================
    x = layers.Conv1D(
        filters=32,
        kernel_size=15,
        strides=2,
        padding='same',
        use_bias=False
    )(inputs)

    x = layers.BatchNormalization()(x)

    x = layers.Activation("gelu")(x)

    x = layers.MaxPooling1D(
        pool_size=3,
        strides=2,
        padding='same'
    )(x)

    # =====================================================
    # STAGE 1
    # =====================================================
    x = residual_block_1d(
        x,
        filters=32,
        stride=1,
        use_se=use_se,
        dropout_rate=dropout_rate
    )

    x = residual_block_1d(
        x,
        filters=32,
        stride=1,
        use_se=use_se,
        dropout_rate=dropout_rate
    )

    # =====================================================
    # STAGE 2
    # =====================================================
    x = residual_block_1d(
        x,
        filters=64,
        stride=2,
        use_se=use_se,
        dropout_rate=dropout_rate
    )

    x = residual_block_1d(
        x,
        filters=64,
        stride=1,
        use_se=use_se,
        dropout_rate=dropout_rate
    )

    # =====================================================
    # STAGE 3
    # =====================================================
    x = residual_block_1d(
        x,
        filters=128,
        stride=2,
        use_se=use_se,
        dropout_rate=dropout_rate
    )

    x = residual_block_1d(
        x,
        filters=128,
        stride=1,
        use_se=use_se,
        dropout_rate=dropout_rate
    )

    # =====================================================
    # STAGE 4
    # =====================================================
    x = residual_block_1d(
        x,
        filters=256,
        stride=2,
        use_se=use_se,
        dropout_rate=dropout_rate
    )

    x = residual_block_1d(
        x,
        filters=256,
        stride=1,
        use_se=use_se,
        dropout_rate=dropout_rate
    )

    # =====================================================
    # CLASSIFIER
    # =====================================================
    x = layers.GlobalAveragePooling1D()(x)

    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(
        num_classes,
        activation='softmax'
    )(x)

    # =====================================================
    # MODEL
    # =====================================================
    model = Model(inputs, outputs, name = name)

    return model


# =========================================================
# Create model
# =========================================================
def build_resnet_model(parameters_dict):
    tf.keras.backend.clear_session()
    input_shape=parameters_dict["input_shape"]
    num_classes=parameters_dict["num_classes"]
    name=parameters_dict["name"]
    use_se=parameters_dict["use_se"]
    model = ResNet18_1D(
        input_shape=input_shape,
        num_classes=num_classes,
        use_se=use_se,     # baseline
        dropout_rate=0.1,
        name=name)
    return model
