## Main Usage Scenario

> NOTE: using python 3.10

> NOTE: This usage scenario may alternate with your needs - maybe you want a different pattern size or shape or maybe you want more coupling fps, or you want more than four cameras etc. But scripts are flexible enough to do so without code modification. To do so, modify the [radon_checkerboard.yaml](checkerboard/radon_checkerboard.yaml), `cameras.def.json5` and inspect `calibrate --help` and `capture --help` to get a clue what args to modify.

### 1. Make calibration pattern
The default pattern to use is already rendered onto a pdf: [radon_checkerboard.pdf](checkerboard/radon_checkerboard.pdf), [radon_checkerboard_flipped.pdf](checkerboard/radon_checkerboard_flipped.pdf) - these two images should be printed on both sides of an A4 sheet with an exact square alignment so that when viewed in the light, the opposite image overlaps with the image on the other side. The resulting paper can be strengthened with a frame.

[irl example](https://www.youtube.com/shorts/DMcCJ7dV_Po)

### 2. Place cameras and make calibration
1. Place cameras in a circle around the place where hand is supposed to be laying
2. Install [calibrate/requirements.txt](calibrate/requirements.txt), [test/requirements.txt](test/requirements.txt) into your venv or global python installation
3. Test what ids are assigned to cameras by probing different values with `python -m test --cam {id}` from the repository root (start with `--cam 0` and increment)
4. Make a file `cameras.def.json5` in the repository root and fill it with something like [this](cameras.def.example.json5) (read important notes about .def and .calib files from the example def)
5. Run `python -m calibrate` from the repository root and follow the instructions dropped down in the console:
    ```
    === Camera Calibration Script ===
    Instructions:
    1. Ensure that the calibration pattern (9x13 chessboard of 13mm squares) is visible in all cameras you want to calibrate.
    2. Press 'c' to capture calibration images.
    The script will print which cameras detected the pattern.
    3. Press 's' to perform calibration when ready.
    Calibration requires at least 12 captures when all cameras detect the pattern.   Captures when a camera does not detect the pattern will be skipped.
    4. After calibration, the script will write the intrinsic and extrinsic parameters back to the cameras file.
    ```
6. After finishing calibration there will be created `cameras.calib.json5` with contents like [here](cameras.calib.example.json5)

[irl example](https://www.youtube.com/shorts/nwtn0VRGkjQ)

### 3. Capture hand
1. Make sure all the hardware is ready - cameras are connected and calibrated
2. Install [capture/requirements.txt](capture/requirements.txt) into your venv or global python installation
3. Run `python -m capture` from the repository root

[irl example](https://youtube.com/shorts/QCHkzZVtM5I)