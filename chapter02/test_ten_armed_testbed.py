import unittest

import numpy as np

from ten_armed_testbed import Bandit


class TestGradientBandit(unittest.TestCase):
    def test_common_preference_offset_preserves_action_probabilities(self):
        for offset in (0., 1000., -1000.):
            with self.subTest(offset=offset):
                bandit = Bandit(k_arm=2, initial=offset, gradient=True)
                bandit.reset()
                bandit.q_estimation[1] += np.log(3.)
                bandit.act()
                np.testing.assert_allclose(bandit.action_prob, [0.25, 0.75])


if __name__ == '__main__':
    unittest.main()
