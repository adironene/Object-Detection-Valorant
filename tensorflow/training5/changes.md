- learning_rate=LEARNING_RATE * 0.3

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=1e-6,
        verbose=1  # Print when learning rate changes
    )

    took out augment data