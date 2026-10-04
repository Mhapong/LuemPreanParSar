# --doctest-modules imports every module in src/: skip the PyTorch-only ones where torch is not installed
import importlib.util

collect_ignore = [] if importlib.util.find_spec("torch") else ["src/luem/models/cnn_net.py"]
