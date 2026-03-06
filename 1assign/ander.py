
# LIBRARIES
import torch

# CONFIGS
csv_path = "./data/insurance.csv"
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"


