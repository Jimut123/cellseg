def Classification(H,W,C):

    input_layer = tf.keras.Input(shape=(H, W, C))

    m1_1 = tf.keras.layers.Conv2D(32, 3, activation='relu', strides=(1, 1), name="m1_1", padding='same')(input_layer)
    m1_2 = tf.keras.layers.Conv2D(32, 3, activation='relu', strides=(1, 1), name="m1_2", padding='same')(m1_1)
    m1_3 = tf.keras.layers.Conv2D(32, 3, activation='relu', strides=(2, 2), name="m1_3", padding='same')(m1_2)

    l1_1 = tf.keras.layers.Conv2D(32, 3, activation='relu', strides=(1, 1), name="l1_1", padding='same')(m1_3)
    l1_2 = tf.keras.layers.Conv2D(32, 3, activation='relu', strides=(1, 1), name="l1_2", padding='same')(l1_1)
    l1_3 = tf.keras.layers.Conv2D(32, 3, activation='relu', strides=(1, 1), name="l1_3", padding='same')(l1_2)

    r1_1 = tf.keras.layers.Conv2D(32, 1, activation='relu', strides=(1, 1), name="r1_1", padding='same')(m1_3)
    r1_2 = tf.keras.layers.Conv2D(32, 1, activation='relu', strides=(1, 1), name="r1_2", padding='same')(r1_1)
    r1_3 = tf.keras.layers.Conv2D(32, 1, activation='relu', strides=(1, 1), name="r1_3", padding='same')(r1_2)


    c1 = concatenate([l1_3, r1_3, m1_3])

    m2_1 = tf.keras.layers.Conv2D(64, 3, activation='relu', strides=(1, 1), name="m2_1", padding='same')(c1)
    m2_2 = tf.keras.layers.Conv2D(64, 3, activation='relu', strides=(1, 1), name="m2_2", padding='same')(m2_1)
    m2_3 = tf.keras.layers.Conv2D(64, 3, activation='relu', strides=(2, 2), name="m2_3", padding='same')(m2_2)

    l2_1 = tf.keras.layers.Conv2D(64, 3, activation='relu', strides=(1, 1), name="l2_1", padding='same')(m2_3)
    l2_2 = tf.keras.layers.Conv2D(64, 3, activation='relu', strides=(1, 1), name="l2_2", padding='same')(l2_1)
    l2_3 = tf.keras.layers.Conv2D(64, 3, activation='relu', strides=(1, 1), name="l2_3", padding='same')(l2_2)

    r2_1 = tf.keras.layers.Conv2D(64, 1, activation='relu', strides=(1, 1), name="r2_1", padding='same')(m2_3)
    r2_2 = tf.keras.layers.Conv2D(64, 1, activation='relu', strides=(1, 1), name="r2_2", padding='same')(r2_1)
    r2_3 = tf.keras.layers.Conv2D(64, 1, activation='relu', strides=(1, 1), name="r2_3", padding='same')(r2_2)

    c2 = concatenate([l2_3, r2_3, m2_3])

    m3_1 = tf.keras.layers.Conv2D(128, 3, activation='relu', strides=(1, 1), name="m3_1", padding='same')(c2)
    m3_2 = tf.keras.layers.Conv2D(128, 3, activation='relu', strides=(1, 1), name="m3_2", padding='same')(m3_1)
    m3_3 = tf.keras.layers.Conv2D(128, 3, activation='relu', strides=(2, 2), name="m3_3", padding='same')(m3_2)

    l3_1 = tf.keras.layers.Conv2D(128, 3, activation='relu', strides=(1, 1), name="l3_1", padding='same')(m3_3)
    l3_2 = tf.keras.layers.Conv2D(128, 3, activation='relu', strides=(1, 1), name="l3_2", padding='same')(l3_1)
    l3_3 = tf.keras.layers.Conv2D(128, 3, activation='relu', strides=(1, 1), name="l3_3", padding='same')(l3_2)

    r3_1 = tf.keras.layers.Conv2D(128, 1, activation='relu', strides=(1, 1), name="r3_1", padding='same')(m3_3)
    r3_2 = tf.keras.layers.Conv2D(128, 1, activation='relu', strides=(1, 1), name="r3_2", padding='same')(r3_1)
    r3_3 = tf.keras.layers.Conv2D(128, 1, activation='relu', strides=(1, 1), name="r3_3", padding='same')(r3_2)


    c3 = concatenate([l3_3, r3_3, m3_3])

    m4_1 = tf.keras.layers.Conv2D(256, 3, activation='relu', strides=(2, 2), name="m4_1")(c3)
    m4_2 = tf.keras.layers.Conv2D(256, 3, activation='relu', strides=(2, 2), name="m4_2")(m4_1)
    m4_3 = tf.keras.layers.Conv2D(256, 3, activation='relu', strides=(2, 2), name="m4_3")(m4_2)
    m4_4 = tf.keras.layers.Conv2D(256, 3, activation='relu', strides=(2, 2), name="m4_4")(m4_3)
    # m4_5 = tf.keras.layers.Conv2D(128, 3, activation='relu', strides=(2, 2), name="m4_5")(m4_4)
    # m4_6 = tf.keras.layers.Conv2D(128, 3, activation='relu', strides=(2, 2), name="m4_6")(m4_5)
    # m4_7 = tf.keras.layers.Conv2D(128, 3, activation='relu', strides=(2, 2), name="m4_7")(m4_6)


    x = tf.keras.layers.SpatialDropout2D(0.15, name="dropout_3")(m4_4)
    x = tf.keras.layers.Flatten(name="flatten")(x)
    x = tf.keras.layers.Dense(N_LABELS, activation='softmax', name="output_layer")(x)

    model = tf.keras.models.Model(inputs=input_layer, outputs=x)
    return model