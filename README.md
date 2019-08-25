# GURU_Video
GURU video processing for pupil data

Dependencies:
- Numpy
- OpenCV

## Usage

Display video:

```bash
python3 video_display.py --data_folder kitchen --bucket 1
```

Clipping Video into Numbers of small videos:

```bash
python3 video_save.py --data_folder --num 3
```

Parsing Pupil Data
```bash
python3 parser.py --data_folder ... --save_folder ...
```
