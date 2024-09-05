from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.metrics import Precision, Recall

def model_trainer(all_input_sequence_data, all_label, model, epochs=50, batch_size=32, model_save_path='full_model_subclass'):
    """
    Train a TensorFlow/Keras model with callbacks and save the trained model.

    Args:
        all_input_sequence_data (dict): Dictionary with keys 'train' and 'val' for training and validation input data.
        all_label (dict): Dictionary with keys 'train' and 'val' for training and validation labels.
        model (tf.keras.Model): The model to be trained.
        epochs (int, optional): Number of epochs to train the model. Default is 50.
        batch_size (int, optional): Batch size for training. Default is 32.
        model_save_path (str, optional): Path to save the trained model. Default is 'full_model_subclass'.

    Returns:
        history (tf.keras.callbacks.History): History object containing training metrics.
        model (tf.keras.Model): The trained model.
    """
    print("=" * 52)
    print("Step 4: Start training Model :)")
    print("=" * 52)

    # Callbacks for learning rate reduction and early stopping
    learning_rate_reduction = ReduceLROnPlateau(
        monitor='val_accuracy',
        patience=2,
        factor=0.5,
        min_lr=0.00005,
        verbose=1
    )

    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=3,
        restore_best_weights=True,
        verbose=1
    )

    try:
        # Compile the model
        model.compile(
            optimizer="adam",
            loss='categorical_crossentropy',
            metrics=['accuracy', Precision(), Recall()]
        )

        # Train the model
        history = model.fit(
            all_input_sequence_data['train'], all_label['train'],
            validation_data=(all_input_sequence_data['val'], all_label['val']),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, learning_rate_reduction],
        )

        # Save the model
        # Ensure the path has a valid extension
        if model_save_path.endswith('.keras') or model_save_path.endswith('.h5'):
            model.save(model_save_path)
        else:
            model.save(model_save_path + '.keras')

        print(f"Model saved as '{model_save_path}'")
    except Exception as e:
        print(f"An error occurred during model training or saving: {e}")
        return None, None

    print("=" * 52)
    print("Finish training Model :)")
    print("=" * 52)

    return history, model
