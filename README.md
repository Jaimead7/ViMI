<div align="center">
    <h1 style= "margin 0px; padding: 0px;">VIMI - <i>Vision Models Interface</i></h1>
    <hr style="width: 100%; height: 1px; margin: 15px;">
    <a href="https://github.com/Jaimead7/ViMI/actions/workflows/py-tests.yml"><img src="https://github.com/Jaimead7/ViMI/actions/workflows/py-tests.yml/badge.svg"></a>
    <a href="https://github.com/Jaimead7/ViMI/blob/master/LICENSE"><img src="https://img.shields.io/static/v1.svg?label=LICENSE&message=MIT&color=2dba4e&colorA=2b3137"></a>
    <a href="https://pypi.org/project/jaimead7-vimi/"><img src="https://img.shields.io/pypi/v/jaimead7-vimi.svg?color=2b3137"></a>
</div> 

Interface for AI vision models.  

## Authors
> Jaime Alvarez Diaz  
> [![email](https://img.shields.io/static/v1.svg?label=Gmail&message=alvarez.diaz.jaime1@gmail.com&logo=gmail&color=2dba4e&logoColor=white&colorA=c71610)](mailto:alvarez.diaz.jaime1@gmail.com)  
[![GitHub Profile](https://img.shields.io/static/v1.svg?label=GitHub&message=Jaimead7&logo=github&color=2dba4e&colorA=2b3137)](https://github.com/Jaimead7)  

## Installation
Install as a package from source files
```powershell
git clone https://github.com/Jaimead7/ViMI.git
cd ViMI
py -m pip install .
```

Install as a package from pypi
```
py -m pip install jaimead7-vimi
```

## Usage
Create a model folder with a valid structure. You can find examples in [./test/models/](./test/models/).

```python
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from vimi.engines import EnginesReg, ModelEngine
from vimi.results import Results


MODEL_PATH: Path = Path('./test/models/DetectModelNCNN')
IMGS_PATHS: list[Path] = [
    Path(Path('./test/imgs/coco/bike.jpg')),
    Path(Path('./test/imgs/coco/planes.jpg'))
]

model: Optional[ModelEngine] = EnginesReg.get_model(MODEL_PATH)
if model is None:
    raise RuntimeError(f'Cant\'t open the model "{MODEL_PATH}".')

results: list[Results] = model(IMGS_PATHS)

if len(results) != len(IMGS_PATHS):
    raise RuntimeError(f'Error processing images: Expected {len(IMGS_PATHS)} Results, got {len(results)}.')
for result, img in zip(results, IMGS_PATHS):
    cv2.imshow(f'{img.name}', result.plot())
_: int = cv2.waitKey(0)
cv2.destroyAllWindows()
```

## Available model interfaces
### [YOLO11](https://docs.ultralytics.com/es/models/yolo11/) models:
| *type*       | **ncnn** | **pyTorch** |
| ------------ | :------: | :---------: |
| classify     | ✅ | ❌ |
| detect       | ✅ | ❌ |
| obb          | ❌ | ❌ |
| pose         | ❌ | ❌ |
| segmentation | ❌ | ❌ |

## License
This project is licensed under the [MIT](./LICENSE) license.  

**Third-party models:**  
The `./test/` directory contains model files that are subject to their own respective licenses. Each model includes a `metadata.yaml` file specifying its applicable license terms. Please refer to these metadata files for details.  

## Contributing
Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.
