from .tokenflow.tokenflow_node2 import KSamplerTFNode
from .tracker.tracker_node import TrackerNode
from .tracker.hand_node import HandTrackerNode
from .shared.batched_ksampler_node import KSamplerBatchedNode
from .tracker.video_tracker_prompt_node import VideoTrackerPromptNode
from .rave.rave_ksampler_node import KSamplerRaveNode
from .shared.temporal_node import TemporalNetPreprocessor
from .instancediffusion.instance_tracker_prompt_node import InstanceTrackerPromptNode
from .instancediffusion.instance_model_node import InstanceDiffusionLoaderNode

NODE_CLASS_MAPPINGS = {
    "KSamplerTF": KSamplerTFNode,
    "TrackerNode": TrackerNode,
    "KSamplerBatchedNode": KSamplerBatchedNode,
    "VideoTrackerPromptNode": VideoTrackerPromptNode,
    "KSamplerRAVE": KSamplerRaveNode,
    "TemporalNetPreprocessor": TemporalNetPreprocessor,
    "InstanceTrackerPrompt": InstanceTrackerPromptNode,
    "InstanceDiffusionLoader": InstanceDiffusionLoaderNode,
    "HandTrackerNode": HandTrackerNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "KSamplerRAVE": "KSampler (RAVE)",
    "KSamplerTF": "KSampler (TF)",
    "TrackerNode": "Object Tracker",
    "KSamplerBatchedNode": "KSampler Batched",
    "VideoTrackerPromptNode": "Video Tracker Prompt",
    "TemporalNetPreprocessor": "TemporalNet Preprocessor",
    "InstanceTrackerPrompt": "Instance Tracker Prompt",
    "InstanceDiffusionLoader": "Instance Diffusion Loader",
    "HandTrackerNode": "Hand Tracker Node"
}
