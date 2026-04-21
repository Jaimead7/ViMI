from pathlib import Path
from typing import Optional

import cv2

from vimi.engines import EnginesReg, ModelEngine
from vimi.results import Results


def main() -> None:
    TESTS_PATH: Path = Path(__file__).absolute().parents[2] / 'test'
    IMGS_PATH: Path = TESTS_PATH / 'imgs' / 'dota'
    MODEL_PATH: Path = TESTS_PATH / 'models' / 'OBBModelNCNN'

    IMGS: dict[Path, dict] = {
        IMGS_PATH / 'airport.png': {
            'result': ['plain', 'plain', 'plain'],
        },
        IMGS_PATH / 'roundabout.png': {
            'result': ['roundabout', 'small vehicle'],
        }
    }

    model: Optional[ModelEngine] = EnginesReg.get_model(model_path= MODEL_PATH)
    if model is None:
        raise SystemError(f'Cant\'t open the model "{MODEL_PATH}".')

    for img_path, expected_res in IMGS.items():
        results: list[Results] = model(img_path)
        if len(results) == 0:
            raise ValueError(f'Error processing image "{img_path}".')
        res: Results = results[0]
        print(f'"{img_path}"\n')
        print(res)
        print(f'***********************\n')
        cv2.namedWindow(img_path.stem, cv2.WINDOW_FULLSCREEN)
        cv2.imshow(img_path.stem, res.plot())

    cv2.waitKey(-1)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    print('\n\n---------- OBB INTERFERENCE EXAMPLE ----------\n')
    main()
