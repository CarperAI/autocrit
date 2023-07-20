from typing import List, Dict, Any, Optional
import re

from autocrit.pipelines.pipelines import InferenceComponent, InferencePipeline, PipelineState
from autocrit.pipelines.composer import InferenceComposer


class DummySentence(InferenceComponent):
    def __init__(self, name, max_steps):
        super().__init__(name, max_steps)
        self.response = "S: Alex has 5 apples and Josh has 6 bananas, so they have 5*6=30 fruits total\n"

    def __call__(self, data : Dict[str, List[Any]], start, end) -> Dict[str, Any]:
        data["response"] = [response + self.response for response in data["response"]]
        return data, start

class DummyCritique(InferenceComponent):
    def __init__(self, name, max_steps):
        super().__init__(name, max_steps)
        self.response = "C: You should use +, not *\n"

    def __call__(self, data : Dict[str, List[Any]], start, end) -> Dict[str, Any]:
        data["response"] = [response + self.response for response in data["response"]]
        return data, start

class DummyRefinement(InferenceComponent):
    def __init__(self, name, max_steps):
        super().__init__(name, max_steps)
        self.response = "R: Alex has 5 apples and Josh has 6 bananas, so they have 5+6=11 fruits total\n"

    def __call__(self, data : Dict[str, List[Any]], start, end) -> Dict[str, Any]:
        data["response"] = [response + self.response for response in data["response"]]
        return data, start

def dummy_refinement_stopping_criteria(sample):
    latest_sent = sample["response"].split("S: ")[-1]
    return len(re.findall("R: ", latest_sent)) >= 3

def run_dummy_pipeline():
    dummy_pipeline = InferenceComposer(
                        InferencePipeline("dummy_answer", 1,  [
                            InferencePipeline("dummy_sentence", 3, [
                                DummySentence("dummy_initial_sentence", 1),
                                InferencePipeline("dummy_critique_refine", 5, [
                                    DummyCritique("dummy_critique", 1),
                                    DummyRefinement("dummy_refine", 1),
                                ], stopping_criteria=dummy_refinement_stopping_criteria),
                            ]),
                        ])
                    )

    # A core problem here is that the state isn't uniform across depth. Will need to address this.
    # NOTE: Some bug where I do at least one round even if it's all complete
    start_state = [PipelineState("dummy_answer", 0), PipelineState("dummy_sentence", 0)]#, PipelineState("dummy_refine", 0)]
    end_state = [PipelineState("dummy_answer", 0), PipelineState("dummy_sentence", 2), PipelineState("dummy_critique_refine", 1)]
    data = {"response": [""]}
    data = dummy_pipeline(data, start_state, end_state)
    print(data["response"][0])

if __name__ == "__main__":
    run_dummy_pipeline()