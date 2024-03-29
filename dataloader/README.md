# Singleton Usage
Parsing Pupil Data
```bash
python3 parser.py --data_folder ...
```

Clipping Data:

The algorithm will create folder 0, 1, 2, 3... each with world_bucket0.mp4 world_pupil_data_bucket0.npy as its data. At the root, directory it will save a txt file to specify the start_frame and end_frame of each folders in relationship with the root files.

# Available API for Use:
...









# Data Format from 'pupil_data'
```python
Serialized_Dict(
    mappingproxy({
        'confidence': 0.685075099793111, 
        'norm_pos': (0.3031836657455456, 0.3923454326122271), 
        'base_data': 
            (
            mappingproxy({
                'model_confidence': 0.6112187135145275, 
                'theta': 1.5536346365166254, 
                'diameter': 37.39571439536745, 
                'sphere': 
                    mappingproxy({
                        'center': (-3.165777031441055, 3.5184313101748654, 59.34805428154027), 
                        'radius': 12.0
                    }), 
                'method': '3d c++', 
                'confidence': 0.685075099793111, 
                'projected_sphere': 
                    mappingproxy({
                        'center': (166.92761400091996, 236.75651103842392), 
                        'axes': (250.72431068103782, 250.72431068103782), 
                        'angle': 90.0}
                    ), 
                'timestamp': 693.883931, 
                'model_birth_timestamp': 676.08475, 
                'circle_3d': 
                    mappingproxy({
                        'center': (-6.067447602154793, 3.312501135732803, 47.705979695698275), 
                        'radius': 1.436479369847379, 
                        'normal': (-0.24180588089281152, -0.017160847870171875, -0.9701728821534997)
                    }), 
                'norm_pos': (0.3031836657455456, 0.3923454326122271), 
                'topic': 'pupil', 
                'ellipse': 
                    mappingproxy({
                        'center': (121.27346629821825, 243.06182695510915), 
                        'axes': (35.06866619396925, 37.39571439536745), 
                        'angle': -4.522513134651533}
                    ), 
                'diameter_3d': 2.872958739694758, 
                'phi': -1.8150595599806159, 
                'model_id': 1, 
                'id': 1})
            ), 
            'timestamp': 693.883931, 
            'topic': 'gaze.2d.1.'})
)
```