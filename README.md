### This is a collection of nodes that are a Work in Progress -- these may break or change behavior in non-backwards compatible ways 


## Instance Diffusion Setup

This is a WIP and a standalone node setup will come eventually.

### Temporary Installation Setup
1. Create "InstanceDiffusion" directory in your Comfy models directory
2. Download the 3 models from https://huggingface.co/spaces/logtd/instancediffusion/tree/main into that directory
3. Create a "yolov8" directory in your Comfy models directory
4. Download yolov8 (any size you want) from https://github.com/ultralytics/ultralytics
5. pip install ultralytics supervision
6. Replace the sampling.py in AnimateDiff-evolved with the `sampling_animate_diff.py` file found in the `instancediffusion/` directory in this repo
7. git clone this repo into your custom_nodes