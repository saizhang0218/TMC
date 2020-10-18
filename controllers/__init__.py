REGISTRY = {}

from .basic_controller_corridor import BasicMAC_corridor
from .basic_controller_2c_vs_64zg import BasicMAC_2c_vs_64zg
from .basic_controller_6h_vs_8z import BasicMAC_6h_vs_8z
from .basic_controller_3s5z import BasicMAC_3s5z
from .basic_controller_3s_vs_4z import BasicMAC_3s_vs_4z
from .basic_controller_3s_vs_5z import BasicMAC_3s_vs_5z

REGISTRY["basic_mac_corridor"] = BasicMAC_corridor
REGISTRY["basic_mac_2c_vs_64zg"] = BasicMAC_2c_vs_64zg
REGISTRY["basic_mac_3s5z"] = BasicMAC_3s5z
REGISTRY["basic_mac_6h_vs_8z"] = BasicMAC_6h_vs_8z
REGISTRY["basic_mac_3s_vs_4z"] = BasicMAC_3s_vs_4z
REGISTRY["basic_mac_3s_vs_5z"] = BasicMAC_3s_vs_5z