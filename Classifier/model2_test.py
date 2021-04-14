def Classification(H,W,C):
    
    input_layer = tf.keras.Input(shape=(H, W, C))

    m1_1 = BatchNormalization()(Conv2D(32, 3, activation='relu', strides=(1, 1), name="m1_1", padding='same')(input_layer))
    m1_2 = BatchNormalization()(Conv2D(32, 3, activation='relu', strides=(1, 1), name="m1_2", padding='same')(m1_1))
    m1_3 = MaxPool2D((2, 2))(BatchNormalization()(Conv2D(32, 3, activation='relu', strides=(1, 1), name="m1_3", padding='same')(m1_2)))

    l1_1 = BatchNormalization()(Conv2D(32, 3, activation='relu', strides=(1, 1), name="l1_1", padding='same')(m1_3))
    l1_2 = BatchNormalization()(Conv2D(32, 3, activation='relu', strides=(1, 1), name="l1_2", padding='same')(l1_1))
    l1_3 = MaxPool2D((2, 2))(BatchNormalization()(Conv2D(32, 3, activation='relu', strides=(1, 1), name="l1_3", padding='same')(l1_2)))

    m2_1 = BatchNormalization()(Conv2D(64, 3, activation='relu', strides=(1, 1), name="m2_1", padding='same')(l1_3))
    m2_2 = BatchNormalization()(Conv2D(64, 3, activation='relu', strides=(1, 1), name="m2_2", padding='same')(m2_1))
    m2_3 = MaxPool2D((2, 2))(BatchNormalization()(Conv2D(64, 3, activation='relu', strides=(1, 1), name="m2_3", padding='same')(m2_2)))

    l2_1 = BatchNormalization()(Conv2D(64, 3, activation='relu', strides=(1, 1), name="l2_1", padding='same')(m2_3))
    l2_2 = BatchNormalization()(Conv2D(64, 3, activation='relu', strides=(1, 1), name="l2_2", padding='same')(l2_1))
    l2_3 = MaxPool2D((2, 2))(BatchNormalization()(Conv2D(64, 3, activation='relu', strides=(1, 1), name="l2_3", padding='same')(l2_2)))


    m3_1 = BatchNormalization()(Conv2D(128, 3, activation='relu', strides=(1, 1), name="m3_1", padding='same')(l2_1))
    m3_2 = BatchNormalization()(Conv2D(128, 3, activation='relu', strides=(1, 1), name="m3_2", padding='same')(m3_1))
    m3_3 = MaxPool2D((2, 2))(BatchNormalization()(Conv2D(128, 3, activation='relu', strides=(1, 1), name="m3_3", padding='same')(m3_2)))

    l3_1 = BatchNormalization()(Conv2D(128, 3, activation='relu', strides=(1, 1), name="l3_1", padding='same')(m3_3))
    l3_2 = BatchNormalization()(Conv2D(128, 3, activation='relu', strides=(1, 1), name="l3_2", padding='same')(l3_1))
    l3_3 = MaxPool2D((2, 2))(BatchNormalization()(Conv2D(128, 3, activation='relu', strides=(1, 1), name="l3_3", padding='same')(l3_2)))

    m4_1 = BatchNormalization()(Conv2D(256, 3, activation='relu', strides=(2, 2), name="m4_1")(l3_3))
    m4_2 = BatchNormalization()(Conv2D(256, 3, activation='relu', strides=(2, 2), name="m4_2")(m4_1))
    # m4_3 = BatchNormalization()(Conv2D(512, 3, activation='relu', strides=(2, 2), name="m4_3")(m4_2))
    # m4_4 = BatchNormalization()(Conv2D(512, 3, activation='relu', strides=(2, 2), name="m4_4")(m4_3))


    #x = SpatialDropout2D(0.5, name="dropout_3")(m4_4)
    x = Flatten(name="flatten")(m4_2)
    x = Dense(512, activation='relu', name="dense_512")(x)
    x = Dense(N_LABELS, activation='softmax', name="output_layer")(x)

    model = tf.keras.models.Model(inputs=input_layer, outputs=x)
    return model
