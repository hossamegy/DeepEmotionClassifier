import matplotlib.pyplot as plt

def model_evaluation(his, model, all_input_sequence_data, all_label):
    """
    Evaluates the model on test data and plots training history.

    Args:
        his: History object containing training metrics.
        model: The trained model to be evaluated.
        all_input_sequence_data: Dictionary containing input data for evaluation.
        all_label: Dictionary containing labels for evaluation.
    """
    print("=" * 52)
    print("Step 5: Start evaluation model :)")
    print("=" * 52)

    try:
        # Evaluate the model on the test data
        test_results = model.evaluate(all_input_sequence_data['test'], all_label['test'])

        print("-" * 45)
        print("Test evaluation results:")
        print("-" * 45)
    
        metrics_names = model.metrics_names
        for name, result in zip(metrics_names, test_results):
            print(f"{name.capitalize()}: {result:.4f}")
        print("=" * 52)
    except Exception as e:
        print(f"An error occurred during model evaluation: {e}")
        return

    # Plot training history
    plt.figure(figsize=(12, 4))

    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(his.history.get('accuracy', []), label='Train Accuracy')
    plt.plot(his.history.get('val_accuracy', []), label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(his.history.get('loss', []), label='Train Loss')
    plt.plot(his.history.get('val_loss', []), label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
