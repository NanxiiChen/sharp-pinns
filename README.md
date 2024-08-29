# PF-PINNs

## Configs

### 1d-activation driven

```ini
[PARAM]
ALPHA_PHI = 1.03e-4
OMEGA_PHI = 1.76e7
MM = 7.94e-18
DD = 8.5e-10
AA = 5.35e7
LP = 1e-11
CSE = 1.
CLE = 5100/1.43e5


[TRAIN]
DIM = 1
DRIVEN = "activation"
GEO_COEF = 1e4
TIME_COEF = 1e-5
TIME_SPAN = (0, 1)
GEO_SPAN = (-0.5, 0.5)
REF_PATH = ./data/results-fenics-active.csv
ALPHA = 1.0
LR = 5e-4
RESUME = None
NUM_SEG = 5

NETWORK_SIZE = [2] + [16]*8 + [2]
NTK_BATCH_SIZE = 128
NTK_MODE = "mini"
BREAK_INTERVAL = 100
EPOCHS = 4000
GEOTIME_SHAPE = [25, 25]
BCDATA_SHAPE = 128
ICDATA_SHAPE = 128
SAMPLING_STRATEGY = ["grid_transition"] * 3
RAR_BASE_SHAPE = 5000
RAR_SHAPE = 512
ADAPTIVE_SAMPLING = "rar"

LOG_NAME = None
```

### 1d-dissolution driven
 
```ini
[PARAM]
ALPHA_PHI = 1.03e-4
OMEGA_PHI = 1.76e7
MM = 7.94e-18
DD = 8.5e-10
AA = 5.35e7
LP = 2.0
CSE = 1.
CLE = 5100/1.43e5

[TRAIN]
DIM = 1
DRIVEN = "dissolution"
GEO_COEF = 1e4
TIME_COEF = 1e-2
TIME_SPAN = (0, 1)
GEO_SPAN = (-0.5, 0.5)
NETWORK_SIZE = [2] + [16]*8 + [2]
REF_PATH = "./data/results-fenics-diffusion.csv"
NTK_BATCH_SIZE = 400
BREAK_INTERVAL = 200
EPOCHS = 500000
ALPHA = 1.0
LR = 5e-4

GEOTIME_SHAPE = [15, 15]
BCDATA_SHAPE = 128
ICDATA_SHAPE = 256
SAMPLING_STRATEGY = ["grid_transition"] * 3

RAR_BASE_SHAPE = 20000
RAR_SHAPE = 4000

RESUME = None
ADAPTIVE_SAMPLING = "rar"
```

### 2d-dissolution 

```ini
[PARAM]
ALPHA_PHI = 1.03e-4
OMEGA_PHI = 1.76e7
MM = 7.94e-18
DD = 8.5e-10
AA = 5.35e7
LP = 2.0
CSE = 1.
CLE = 5100/1.43e5

[TRAIN]
DIM = 2
DRIVEN = "dissolution"
GEO_COEF = 1e4
TIME_COEF = 5e-2
TIME_SPAN = (0, 1)
GEO_SPAN = ((-0.5, 0.5), (0, 0.5))
NETWORK_SIZE = [3] + [16]*8 + [2]
NUM_SEG = 5

MESH_POINTS = "./data/2d/mesh_points.npy"
REF_PREFIX = "./data/2d/sol-"
TARGET_TIMES = [0.00, 5.12, 10.24, 18.43]

NTK_BATCH_SIZE = 300
BREAK_INTERVAL = 10
EPOCHS = 800000
ALPHA = 1.0
LR = 5e-4

GEOTIME_SHAPE = 3000
BCDATA_SHAPE = 128
ICDATA_SHAPE = 256
SAMPLING_STRATEGY = ["lhs", "lhs", "lhs"]

RAR_BASE_SHAPE = 20000
RAR_SHAPE = 4000

RESUME = None
ADAPTIVE_SAMPLING = "rar"
FORWARD_BATCH_SIZE = 2000
```

### 2d-dissolution-2pits

```ini
[PARAM]
ALPHA_PHI = 1.03e-4
OMEGA_PHI = 1.76e7
MM = 7.94e-18
DD = 8.5e-10
AA = 5.35e7
LP = 2.0
CSE = 1.
CLE = 5100/1.43e5

[TRAIN]
DIM = 2
DRIVEN = "dissolution"
GEO_COEF = 1e4
TIME_COEF = 1e-2
TIME_SPAN = (0, 0.20)
GEO_SPAN = ((-0.5, 0.5), (0, 0.5))
NETWORK_SIZE = [3] + [80]*4 + [2]
NUM_SEG = 10

MESH_POINTS = "./data/2d-2pits/mesh_points.npy"
REF_PREFIX = "./data/2d-2pits/sol-"
TARGET_TIMES = [0.00, 2.56, 9.84, 19.99]

NTK_BATCH_SIZE = 100
BREAK_INTERVAL = 200
EPOCHS = 800000
ALPHA = 1.0
LR = 5e-4

GEOTIME_SHAPE = [30, 15, 50]
BCDATA_SHAPE = 200
ICDATA_SHAPE = 500
SAMPLING_STRATEGY = ["grid_transition", "lhs", "lhs"]

RAR_BASE_SHAPE = 20000
RAR_SHAPE = 4000


RESUME = None
ADAPTIVE_SAMPLING = "rar"
FORWARD_BATCH_SIZE = 2000
```

### 3pits
```ini
[PARAM]
ALPHA_PHI = 1.03e-4
OMEGA_PHI = 1.76e7
MM = 7.94e-18
DD = 8.5e-10
AA = 5.35e7
LP = 2.0
CSE = 1.
CLE = 5100/1.43e5

[TRAIN]
DIM = 2
DRIVEN = "dissolution"
GEO_COEF = 1e4
TIME_COEF = 1e-2
TIME_SPAN = (0, 0.10)
GEO_SPAN = ((-0.5, 0.5), (0, 0.5))
NETWORK_SIZE = [3] + [80]*4 + [2]
NUM_SEG = 10

MESH_POINTS = "./data/2d-3pits/mesh_points.npy"
REF_PREFIX = "./data/2d-3pits/sol-"
TARGET_TIMES = [0.00, 2.56, 5.82, 9.79]

NTK_BATCH_SIZE = 100
BREAK_INTERVAL = 1
EPOCHS = 800000
ALPHA = 1.0
LR = 5e-4

GEOTIME_SHAPE = [30, 15, 40]
BCDATA_SHAPE = 200
ICDATA_SHAPE = 200
SAMPLING_STRATEGY = ["grid_transition", "lhs", "lhs"]

RAR_BASE_SHAPE = 20000
RAR_SHAPE = 4000
NTK_MODE = "random"
RESUME = None
;RESUME = "/root/tf-logs/3pits-gradient-2024-07-28-00-03-55/model-11000.pt"
ADAPTIVE_SAMPLING = "gar"
FORWARD_BATCH_SIZE = 2000
```

## Citation
```bibtex
@misc{chenPfPinnsPhysicsInformedNeural2024,
  title = {Pf-Pinns: Physics-Informed Neural Networks for Solving Coupled Allen-Cahn and Cahn-Hilliard Phase Field Equations},
  author = {Chen, Nanxi and Lucarini, Sergio and Ma, Rujin and Chen, Airong and Cui, Chuanjie},
  year = {2024},
  month = jan,
  doi = {10.2139/ssrn.4761824},
  keywords = {/unread},
}

```
