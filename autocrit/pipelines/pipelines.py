from dataclasses import dataclass
from abc import abstractmethod
from typing import List, Dict, Any, Optional
from copy import copy, deepcopy
import numpy as np
import re
import json

from autocrit.pipelines.datatypes import PipelineState, NestedPipelineState
from autocrit.pipelines.utils import jsonl_to_dict, dict_to_jsonl, same_state


class InferenceComponent:
    def __init__(self, name : str, max_steps : int):
        self.name = name
        self.max_steps = max_steps
        self.depth = None

    def set_depth(self, depth : int):
        self.depth = depth

    @abstractmethod
    def __call__(self, data : Dict[str, List[Any]], start, end) -> Dict[str, Any]:
        pass


# The assumption is that all the components in the pipeline are distinct
class InferencePipeline(InferenceComponent):
    def __init__(self, name : str, max_steps : int, inference_components : List[InferenceComponent], stopping_criteria=None):
        super().__init__(name, max_steps)
        self.inference_components = {comp.name: comp for comp in inference_components}
        # Map prev component to next
        self.next_component = {inference_components[i].name: inference_components[i+1].name for i in range(len(self.inference_components)-1)}
        # Last component stays fixed (and will be updated at a higher level)
        self.next_component[inference_components[-1].name] = "end"
        # Set stopping critera if given otherwise set always false criteria
        self.stopping_criteria = stopping_criteria if stopping_criteria is not None else lambda data: False

    def next_state(self, state : NestedPipelineState, end : NestedPipelineState):
        # Do not step state if exit state has been reached
        if same_state(state, end):
            return state[self.depth+1], state
        # Otherwise we step the pipeline to the next component
        pipeline_state = state[self.depth+1]
        pipeline_state.step += 1
        cur_comp = self.inference_components[pipeline_state.name]
        # Move to next component if we have reached max_steps
        if pipeline_state.step >= cur_comp.max_steps:
            pipeline_state.name = self.next_component[pipeline_state.name]
            pipeline_state.step = 0
        return state[self.depth+1], state

    def get_first_component_name(self):
        return list(self.inference_components.values())[0].name

    def get_pipeline_state(self, data : Dict[str, Any], state : NestedPipelineState):
        # If nested pipeline state does not include this depth, start from beginning
        if len(state) <= self.depth+1:
            pipeline_state = PipelineState(self.get_first_component_name(), 0)
            state.append(pipeline_state)
        # If the nested pipeline state does include this depth, but not this component, then skip
        elif state[self.depth+1].name not in list(self.inference_components.keys()):
            pipeline_state = PipelineState("end", 0)
        # If both the depth and correct component are present, return
        else:
            pipeline_state = state[self.depth+1]
        # Compute the batch_mask from the data
        pipeline_state.batch_mask = [self.stopping_criteria(s) for s in dict_to_jsonl(data)]
        # If entire batch is masked, go to end state
        if False not in pipeline_state.batch_mask:
            pipeline_state.name = "end"
        return pipeline_state, state

    def exit_state(self, pipeline_state : PipelineState, state : NestedPipelineState, end : NestedPipelineState):
        # If the pipeline has completed and this matches the nested state, 
        # we can remove this depth from the nested state
        if pipeline_state.name == "end" and state[self.depth+1].name == "end":
            state.pop(-1)
            assert len(state) == self.depth+1
            return state
        # If pipeline has completed but this is not matched in nested state,
        # this means are skipping over this pipeline entirely
        # and should not change state
        elif pipeline_state.name == "end" and state[self.depth+1].name != "end":
            return state
        # If we have reached the end state, do not change state
        elif same_state(state, end):
            return state
        # This case should never occur
        else:
            raise RuntimeError("Invalid pipeline exit state!")

    def __call__(self, data : Dict[str, Any], start : NestedPipelineState, end : NestedPipelineState):
        # Get local state of current pipeline
        state = start
        pipeline_state, state = self.get_pipeline_state(data, start)
        # Mask data which should not be passed through pipeline
        # NOTE: This means list lengths in data should be independent of the number of pipeline calls
        orig_data = data
        data = jsonl_to_dict([s for masked, s in zip(pipeline_state.batch_mask, dict_to_jsonl(data)) if not masked])
        while pipeline_state.name != "end" and not same_state(state, end):
            component = self.inference_components[pipeline_state.name]
            # Call component
            data, state = component(data=data, start=state, end=end)
            # Update top level of state after last component call
            pipeline_state, state = self.next_state(state, end)
        # If pipeline has fully completed, resest state
        # Update unmasked data
        orig_data_to_data = np.cumsum(~np.array(pipeline_state.batch_mask))-1
        data = dict_to_jsonl(data)
        orig_data = jsonl_to_dict([data[ind] if not masked else s for ind, masked, s in zip(orig_data_to_data, pipeline_state.batch_mask, dict_to_jsonl(orig_data))])
        state = self.exit_state(pipeline_state, state, end)
        return orig_data, state
