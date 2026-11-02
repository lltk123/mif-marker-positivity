import unittest, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from continuation_core import *


class ContinuationChecks(unittest.TestCase):
    def test_signed_distance_and_ring_candidate(self):
        x = np.zeros((3, 32, 32), np.float32)
        x[1, 8:24, 8:24] = 1
        x[2, 8:24, 25:30] = 1
        g = geometry(x)
        self.assertEqual(g.shape, (4, 32, 32))
        self.assertEqual(g[1, 8, 10], 0)
        self.assertGreater(g[1, 16, 16], 0)
        self.assertLess(g[1, 0, 0], 0)
        self.assertEqual(g[3, 15, 23], 0)
        self.assertEqual(g[3, 15, 8], 1)

    def test_orientation_preserves_target_signal_assignment(self):
        x = np.zeros((3, 32, 32), np.float32)
        x[1, 8:24, 8:24] = 1
        x[0] = x[1] * 0.6
        for epoch in [4, 5, 6, 7]:
            a = oriented(x, 93, 2, epoch)
            self.assertTrue(np.array_equal(a[0], a[1] * 0.6))

    def test_sdf_added_channel_zero_initialization_and_common_weights(self):
        cc.seed_all(3)
        state = cc.Network().state_dict()
        cc.seed_all(4)
        a = network("control", state)
        cc.seed_all(4)
        b = network("sdf", state)
        self.assertTrue(torch.equal(a.encoder[0].weight, b.encoder[0].weight[:, :3]))
        self.assertEqual(int(torch.count_nonzero(b.encoder[0].weight[:, 3])), 0)
        self.assertEqual(
            sum(p.numel() for p in b.parameters())
            - sum(p.numel() for p in a.parameters()),
            288,
        )

    def test_physics_loss_rejects_ineligible_transformed_view(self):
        target = np.ones((8, 8), bool)
        a = np.full((8, 8), 0.0001, np.float32)
        self.assertFalse(eligible_physics(a, a.copy(), target, "noise_low"))
        bright = np.full((8, 8), 0.5, np.float32)
        self.assertTrue(eligible_physics(bright, bright.copy(), target, "noise_low"))
        self.assertFalse(eligible_physics(bright, bright * 0.1, target, "zero_0.7"))
        out = torch.tensor([[1.0, 2.0], [3.0, 1.0]], requires_grad=True)
        other = torch.tensor([[5.0, -2.0], [-2.0, 5.0]], requires_grad=True)
        y = torch.tensor([1, 0])
        ok = torch.tensor([False, False])
        ce = nn.functional.cross_entropy(out, y, reduction="none")
        lc = (
            0.5 * ce
            + 0.5 * nn.functional.cross_entropy(other, y, reduction="none")
            + 0.1
            * nn.functional.smooth_l1_loss(
                out[:, 1] - out[:, 0], other[:, 1] - other[:, 0], reduction="none"
            )
        )
        loss = torch.where(ok, lc, ce).mean()
        self.assertTrue(torch.equal(loss, ce.mean()))
        loss.backward()
        self.assertTrue(torch.equal(other.grad, torch.zeros_like(other)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
