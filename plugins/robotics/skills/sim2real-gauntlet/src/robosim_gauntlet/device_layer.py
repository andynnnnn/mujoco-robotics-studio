"""Select Playground's wrapper when available, with a native MuJoCo fallback."""

try:
  from mujoco_playground.wrapper import DeviceLayerWrapper
except ModuleNotFoundError:
  class DeviceLayerWrapper:
    def __init__(self, device_layer):
      self._device_layer = device_layer

    @property
    def unwrapped(self):
      return getattr(self._device_layer, "unwrapped", self._device_layer)

    def __getattr__(self, name):
      return getattr(self._device_layer, name)
