import numpy as np
import lime


class LIME:
    """
    LIME explainer.
    """
    def __init__(self, model, kernel_fn, kernel_width=0.75, verbose=False):
        """
        Initialize LIME explainer.
        Args:
            model (function): Model function.
            kernel_fn (function): Kernel function.
            kernel_width (float): Kernel width.
            verbose (bool): Verbosity.
        """
        self.model = model
        self.kernel_fn = kernel_fn
        self.kernel_width = kernel_width
        self.verbose = verbose

    def explain(self, instance, n_samples=5000, n_features=10, labels=None, **kwargs):
        """
        Explain an instance.
        Args:
            instance (np.ndarray): Instance.
            n_samples (int): Number of samples.
            n_features (int): Number of features.
            labels (list): List of labels.
        Returns:
            np.ndarray: Explanations.
        """
        if labels is None:
            labels = [1]
        explanations = np.zeros((len(labels), instance.shape[0], n_features))
        for i, label in enumerate(labels):
            if self.verbose:
                print("Explaining label {}...".format(label))
            explanations[i] = self.explain_label(instance, label, n_samples, n_features, **kwargs)
        return explanations

    def explain_label(self, instance, label, n_samples, n_features, **kwargs):
        """
        Explain an instance for a given label.
        Args:
            instance (np.ndarray): Instance.
            label (int): Label.
            n_samples (int): Number of samples.
            n_features (int): Number of features.
        Returns:
            np.ndarray: Explanations.
        """
        # Sample instances
        samples = self.sample(instance, n_samples, **kwargs)

        # Predictions
        predictions = self.predict(samples)

        # Weights
        weights = self.weights(samples, predictions, label)

        # Fit linear model
        linear_model = self.fit_linear_model(samples, weights, label)

        # Explanations
        explanations = self.explanations(linear_model, instance, n_features)

        return explanations

    def sample(self, instance, n_samples, **kwargs):
        """
        Sample instances.
        Args:
            instance (np.ndarray): Instance.
            n_samples (int): Number of samples.
        Returns:
            np.ndarray: Samples.
        """
        raise NotImplementedError
