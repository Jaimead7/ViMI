# MIT License

# Copyright (c) 2026 Jaime Álvarez Díaz
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the “Software”), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

 
from pathlib import Path

import cv2

from vimi import EnginesReg, ModelEngine, Results


def main() -> None:
    TESTS_PATH: Path = Path(__file__).absolute().parents[2] / 'test'
    IMGS_PATH: Path = TESTS_PATH / 'imgs' / 'coco'
    MODEL_PATH: Path = TESTS_PATH / 'models' / 'DetectModelNCNN'

    IMGS: dict[Path, dict] = {
        IMGS_PATH / 'bike.jpg': {
            'result': ['bicycle', 'horse'],
        },
        IMGS_PATH / 'planes.jpg': {
            'result': ['airplane', 'airplane'],
        }
    }

    model: ModelEngine | None = EnginesReg.get_model(model_path= MODEL_PATH)
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
    print('\n\n---------- CLASSYFY INTERFERENCE EXAMPLE ----------\n')
    main()
