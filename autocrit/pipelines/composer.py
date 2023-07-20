from dataclasses import dataclass
from abc import abstractmethod
from typing import List, Dict, Any, Optional
from copy import copy, deepcopy
import numpy as np
import re
import json

from autocrit.pipelines.datatypes import PipelineState, NestedPipelineState, ComposerStoreElement
from autocrit.pipelines.utils import jsonl_to_dict, dict_to_jsonl, same_state
from autocrit.pipelines.pipelines import InferenceComponent, InferencePipeline


class InferenceComposer:
    def __init__(self, inference_pipeline : InferencePipeline):
        self.inference_pipeline = inference_pipeline
        # Assign depths to each component of inference pipeline
        self.set_depths(inference_pipeline, 0)
        self.store = {}
        self.id = 0

    def set_depths(self, pipeline : InferencePipeline, d : int):
        pipeline.set_depth(d)
        if hasattr(pipeline, "inference_components"):
            for component in pipeline.inference_components.values():
                self.set_depths(component, d+1)
    
    def validate_state(self, state : NestedPipelineState, pipeline : InferencePipeline):
        state = copy(state)
        if state != []:
            pipeline_state = state[0]
            if pipeline.name == pipeline_state.name and pipeline_state.step <= pipeline.max_steps:
                state.pop(0)
                if state != []:
                    sub_pipeline_state = state[0]
                    if sub_pipeline_state.name in pipeline.inference_components.keys():
                        self.validate_state(state, pipeline.inference_components[sub_pipeline_state.name])
                    else:
                        raise ValueError(f"Invalid state encountered.\nComponent name: {sub_pipeline_state.name}\nPipeline components: {list(pipeline.inference_components.keys())}")
            else:
                raise ValueError(f"Invalid state encountered.\nPipeline name, max steps: {pipeline.name}, {pipeline.max_steps}\nState: {state}")
            
    def update_store(self, data, state):
        samples = dict_to_jsonl(data)
        for s in samples:
            self.store[s["id"]] = ComposerStoreElement(s["id"], state, data)

    def expand_state(self, state):
        """
        Expand the given state to the full nested depth.
        """
        state = deepcopy(state)
        assert state is not None and len(state) > 0
        cur_pipeline = self.inference_pipeline
        # Skip top level state as this corresponds to the top level pipeline
        for pipeline_state in state[1:]:
            cur_pipeline = cur_pipeline.inference_components[pipeline_state.name]
        # Descend pipeline until we reach first inference component
        while isinstance(cur_pipeline, InferencePipeline):
            cur_pipeline = list(cur_pipeline.inference_components.values())[0]
            state.append(PipelineState(cur_pipeline.name, 0))
        return state

    def dump_json(self):
        pass

    def from_json(self):
        pass

    def __call__(self, data, start : Optional[NestedPipelineState]=None, end : Optional[NestedPipelineState]=None, step=None):
        """
        : param step: specifies name of pipeline in start to execute once.
        """
        # If requested is already present in the store, retrieve it
        if data.get("id", None):
            # Check that requested data shares same state
            assert False not in [same_state(self.store[data["id"][0]], self.store[id]) for id in data["id"]]
            data = {k: [self.store[id].data[k] for id in data["id"]] for k in self.store[data["id"][0]]}
            start = self.store[data["id"][0]]
        # Assign ids to new data
        else:
            num_samples = len(list(data.values())[0])
            data["id"] = [self.id+i for i in range(num_samples)]
            self.id += num_samples
            if start is None:
                start = [PipelineState(self.inference_pipeline.name, 0)]
            if end is None:
                end = [PipelineState(self.inference_pipeline.name, 1)]
        if step is not None:
            # Require we step a pipeline we are currently in
            start = self.expand_state(start)
            step_depth = None
            for depth, pipeline_state in enumerate(start):
                if pipeline_state.name == step:
                    step_depth = depth
                    break
            if step_depth is None:
                raise ValueError(f"Invalid start, step pair given: \n\
                                    Start: {start}\n\
                                    Step: {step}")
            end = deepcopy(start)
            # TODO: Deal with case we exceed max steps, and need to make transition, or given batch is done with this part of pipeline
            end[step_depth].step += 1
        self.validate_state(start, self.inference_pipeline)
        self.validate_state(end, self.inference_pipeline)
        data, state = self.inference_pipeline(data, start, end)
        self.update_store(data, state)
        return data
