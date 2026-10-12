import importlib.util

collect_ignore = [] if importlib.util.find_spec("torch") else ["src/luem/models/cnn_net.py"]
