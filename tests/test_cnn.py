import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnxruntime")

from luem.models.cnn import VOCAB_SIZE, CNNModel, encode  # noqa: E402
from luem.models.cnn_net import CharCNN, CharGRU, WithSigmoid  # noqa: E402


@pytest.fixture(scope="module", params=["cnn", "gru"])
def net(request):
    torch.manual_seed(0)
    if request.param == "gru":
        return CharGRU(emb=8, hidden=8).eval()
    return CharCNN(emb=8, channels=8, kernels=(2, 3), hidden=8).eval()


def test_vocab_covers_both_layouts():
    assert VOCAB_SIZE < 256
    assert 1 not in encode("l;ylfu") + encode("สวัสดี")


def test_padded_prefix_scores_like_prefix_alone(net):
    ids = encode("l;ylfu")
    with torch.no_grad():
        batch = net(
            torch.tensor(
                [encode("l;ylfu"[:k], len(ids)) for k in range(1, len(ids) + 1)]
            )
        )
        alone = torch.stack(
            [net(torch.tensor([ids[:k]]))[0] for k in range(1, len(ids) + 1)]
        )
    assert torch.allclose(batch, alone, atol=1e-6)


def test_onnx_runtime_matches_torch(net, tmp_path):
    path = tmp_path / "cnn.onnx"
    model = WithSigmoid(net).eval()
    torch.onnx.export(
        model,
        (torch.tensor([encode("abc")]),),
        str(path),
        input_names=["ids"],
        output_names=["p_wrong"],
        dynamic_axes={"ids": {0: "batch", 1: "length"}},
        opset_version=17,
        dynamo=False,
    )
    runtime = CNNModel(path)
    text = "l;ylfu"
    with torch.no_grad():
        ref = [
            model(torch.tensor([encode(text[:k])])).item()
            for k in range(1, len(text) + 1)
        ]
    assert np.allclose(runtime.predict_prefixes(text), ref, atol=1e-5)
    assert runtime.predict(text) == pytest.approx(ref[-1], abs=1e-5)
    assert runtime.predict("555") == 0.0
    assert runtime.predict_prefixes("(abc")[0] == 0.0
