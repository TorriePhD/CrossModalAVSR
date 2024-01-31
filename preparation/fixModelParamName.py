from pathlib import Path
import torch
from tqdm import tqdm

parentPath = Path("/home/st392/fsl_groups/grp_lip/compute/results")
for ckptPath in tqdm(list(parentPath.rglob("*.ckpt"))):
    ckpt = torch.load(ckptPath, map_location="cpu")
    #replace "audioEcoder" with "audioEncoder"
    newDict = {k.replace("audioEcoder", "audioEncoder"): v for k, v in ckpt['state_dict'].items()}
    ckpt['state_dict'] = newDict
    torch.save(ckpt, ckptPath)
    print(f"Fixed {ckptPath}")
    
