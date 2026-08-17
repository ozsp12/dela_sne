import unittest
import numpy as np

from lac import LAC
from tsne import TSNE


class AlgorithmSmokeTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        self.X = np.vstack(
            [
                rng.normal(-2.0, 0.3, size=(15, 4)),
                rng.normal(0.0, 0.3, size=(15, 4)),
                rng.normal(2.0, 0.3, size=(15, 4)),
            ]
        )

    def test_lac_shapes_and_weights(self):
        model = LAC(n_clusters=3, h=0.5, random_state=7).fit(self.X)
        self.assertEqual(model.labels_.shape, (45,))
        self.assertEqual(model.feature_weights_.shape, (3, 4))
        np.testing.assert_allclose(model.feature_weights_.sum(axis=1), 1.0)

    def test_tsne_shape_and_finiteness(self):
        embedding = TSNE(
            perplexity=10,
            learning_rate=50,
            n_iter=30,
            exaggeration_iter=10,
            random_state=7,
        ).fit_transform(self.X)
        self.assertEqual(embedding.shape, (45, 2))
        self.assertTrue(np.all(np.isfinite(embedding)))


if __name__ == "__main__":
    unittest.main()
