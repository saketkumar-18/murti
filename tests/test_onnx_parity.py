"""ONNX export parity: exported graphs must match PyTorch outputs exactly.

Self-contained — builds fresh (untrained) models, exports to a temp dir,
and compares onnxruntime outputs against PyTorch for the decoder and the
diffusion UNet across several inputs. This is what guarantees the browser
runtime sees the same math as the Python pipeline.
"""
import numpy as np
import pytest
import torch

from murti.models import LatentDiffusion, VAE3D
from murti.onnx_export import export_onnx
from murti.text import MAX_LEN


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    vae = VAE3D().eval()
    diff = LatentDiffusion().eval()
    out_dir = str(tmp_path_factory.mktemp("onnx"))
    paths = export_onnx(vae, diff, out_dir)
    return vae, diff, paths


def test_decoder_parity(exported):
    import onnxruntime as ort
    vae, diff, paths = exported
    sess = ort.InferenceSession(paths["decoder"], providers=["CPUExecutionProvider"])
    for seed in (0, 7):
        z = torch.randn(1, vae.latent_ch, 8, 8, 8, generator=torch.manual_seed(seed))
        with torch.no_grad():
            expected = torch.sigmoid(vae.decode(z)).numpy()
        got = sess.run(None, {"latent": z.numpy()})[0]
        assert got.shape == expected.shape
        np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-5)


def test_diffusion_parity(exported):
    import onnxruntime as ort
    vae, diff, paths = exported
    sess = ort.InferenceSession(paths["diffusion"], providers=["CPUExecutionProvider"])
    for t_val in (0, 500, 999):
        z = torch.randn(1, vae.latent_ch, 8, 8, 8, generator=torch.manual_seed(t_val))
        t = torch.tensor([t_val], dtype=torch.long)
        toks = torch.randint(0, 40, (1, MAX_LEN), generator=torch.manual_seed(t_val + 1))
        with torch.no_grad():
            expected = diff.unet(z, t, toks).numpy()
        got = sess.run(None, {
            "z_noisy": z.numpy(),
            "t": t.numpy().astype(np.int64),
            "tokens": toks.numpy().astype(np.int64),
        })[0]
        np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-5)


def test_encoder_parity(exported):
    import onnxruntime as ort
    vae, diff, paths = exported
    sess = ort.InferenceSession(paths["encoder"], providers=["CPUExecutionProvider"])
    x = (torch.rand(1, 1, 32, 32, 32, generator=torch.manual_seed(3)) > 0.8).float()
    with torch.no_grad():
        expected = vae.encode(x)[0].numpy()
    got = sess.run(None, {"voxels": x.numpy()})[0]
    np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-5)
