# Industry Notes (Practical Direction)

This project is now positioned as a production-oriented prototype.  
The next strongest upgrades are:

1. Continuous Sign Language Recognition (CSLR) with temporal models.
2. Edge-optimized deployment (ONNX/TensorRT/OpenVINO).
3. Signer adaptation and confidence calibration.
4. Better sentence-level decoding for low-latency streaming.

## References used for planning

1. WLASL benchmark paper (word-level ASL):  
https://arxiv.org/abs/1910.11006

2. How2Sign dataset and paper (continuous, multimodal ASL):  
https://how2sign.github.io/  
https://arxiv.org/abs/2008.08143

3. RWTH-PHOENIX-Weather dataset page (CSLR/SLT benchmark):  
https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX/

4. Recent review on sign language recognition trends and challenges:  
https://www.mdpi.com/2227-7390/11/2/322

## Applied in current codebase

1. Temporal sequence dataset + temporal model training.
2. Hybrid decision stack (rules + temporal + ML fallback).
3. Runtime quality hints and adaptive performance control.
4. Offline inference/reporting and deployable release bundles.

