from tqdm import tqdm
from joint_names import Body
import numpy as np
import json
import os

def main(input_dir: str, output_dir: str):
    files = os.listdir(input_dir)
    for file in files:
        path = os.path.join(input_dir, file)
        dance = np.load(path)

        output = []
        for smplh in tqdm(dance):
            l = len(smplh)
            smplh = smplh.reshape(l // 3, 3)
            smpl = Body.from_smplh(smplh).as_smpl()
            assert len(smpl) == 24, f"output data length is incorrect, expected 24, received: {len(smpl)}"
            output.append(smpl.tolist())
        
        output_prefix = ".".join(file.split(".")[:-1])
        output_name = f"{output_prefix}.json"

        path = os.path.join(output_dir, output_name)
        with open(path, "w") as f:
            f.write(json.dumps(output))


if __name__ == "__main__":
    main("test_smplh", "outdir")