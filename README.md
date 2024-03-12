Xian Wang, "*Enhancing Vocal Melody Extraction by Exploring Multiple Levels of Contexts*," under review.

This is the accompanying code for the above paper.

## Hardware Requirements:
- one GPU with 11 GB VRAM
- 64 GB RAM

## Main Python Packages Used
- tensorflow 2.x
- librosa
- soundfile
- mir_eval
- medleydb

## Code Structure
- [demo.ipynb] Jupyter notebook demonstrating how to load the model for inference. 
- [main.py] all-in-one file for training and inference
- [pitch_local.py] implementation of the proposed deep learning model
    - [lstm_sub.py] implementation of the LSTM layer used in the model
- [nsgt.py] implementation of a tf function for calculating the VQT spectrogram
- [self_defined] some custom utilities
- [ckpts] stores the best checkpoint of the model

## Datasets
- for training: MedleyDB
- for testing: MedleyDB, ADC04, MIREX05, MIR-1K, RWC

## Use the Code for Inference
For your convenience, I have uploaded a checkpoint of the trained model, which is in the folder *ckpts*.
The script *main.py* has been properly configured to run in inference mode using the above checkpoint, so you do not need
to train the model yourself from scratch.

If you would like to train the model yourself, please follow the training process given in the paper, step by step in an
incremental manner.

## Detailed Performance Measures
In the paper when comparing the performance of different models, due to space limit we only gave the overall accuracy (OA).
Below we present more detailed performance measures:
- voicing recall (VR)
- voicing false alarm (VFA)
- raw pitch accuracy (RPA)
- raw chroma accuracy (RCA)

Measures are in percentage.

Models compared: 
- [kknet] K. Shao, K. Chen, T. Berg-Kirkpatrick, and S. Dubnov, “Towards improving harmonic sensitivity and prediction stability for singing melody extraction,” in Proc. Int. Soc. Music Inf. Retriev. Conf., 2023, pp. 1–7. 
- [mtanet] Y. Gao, Y. Hu, L. Wang, H. Huang, and L. He, “MTANet: Multi-band time-frequency attention network for singing melody extraction from polyphonic music,” in Proc. INTERSPEECH, 2023, pp. 5396–5400. 
- [rmvpe] H. Wei, X. Cao, T. Dan, and Y. Chen, “RMVPE: A robust model for vocal pitch estimation in polyphonic music,” in Proc. INTERSPEECH, 2023, pp. 5421–5425.


 
### ADC04 
|          | VR    | VFA         | RPA   | RCA   | OA        |
|----------|-------|-------------|-------|-------|-----------|
| kknet    | 86.33 | 19.69       | 81.83 | 83.72 | 80.96     |
| mtanet   | 85.67 | &nbsp; 6.00 | 82.14 | 83.50 | **83.26** |
| rmvpe    | 76.56 | 11.10       | 89.82 | 91.20 | 76.48     |
| proposed | 81.78 | 17.27       | 88.89 | 90.92 | 77.99     |

### MIREX05 
|          | VR    | VFA         | RPA   | RCA   | OA    |
|----------|-------|-------------|-------|-------|-------|
| kknet    | 82.70 | 11.23       | 79.41 | 81.58 | 83.12 |
| mtanet   | 82.79 | &nbsp; 4.88 | 79.44 | 79.54 | 85.40 |
| rmvpe    | 85.73 | &nbsp; 5.36 | 91.81 | 91.95 | 87.13 |
| proposed | 84.85 | &nbsp; 4.55 | 88.84 | 89.26 | 86.55 |

### MDB
|          | VR    | VFA         | RPA   | RCA   | OA    |
|----------|-------|-------------|-------|-------|-------|
| kknet    | 70.46 | 14.38       | 62.28 | 66.17 | 76.76 |
| mtanet   | 77.23 | 11.28       | 69.93 | 70.92 | 82.65 |
| rmvpe    | 72.57 | &nbsp; 9.36 | 85.10 | 87.02 | 83.25 |
| proposed | 76.06 | 10.52       | 85.79 | 87.06 | 83.85 |

### MIR-1K
|          | VR    | VFA   | RPA   | RCA   | OA    |
|----------|-------|-------|-------|-------|-------|
| kknet    | 70.87 | 22.60 | 61.63 | 66.91 | 66.41 |
| mtanet   | 73.66 | 13.97 | 67.56 | 69.00 | 72.80 |
| rmvpe    | 63.98 | &nbsp; 4.42  | 87.77 | 91.09 | 72.21 |
| proposed | 74.37 |  12.45     |  84.30     | 86.90      |    74.78   |

### RWC 
|          | VR    | VFA         | RPA   | RCA   | OA    |
|----------|-------|-------------|-------|-------|-------|
| kknet    | 76.19 | 8.40        | 61.99 | 65.17 | 76.97 |
| mtanet   | 79.47 | 10.64       | 62.79 | 63.20 | 76.49 |
| rmvpe    | 75.84 | &nbsp; 8.70 | 75.90 | 76.63 | 77.54 |
| proposed | 77.53 | 10.36       | 75.42 | 76.08 | 77.47 |