from autocrit.pipelines.datatypes import PipelineState, NestedPipelineState, ComposerStoreElement


def dict_to_jsonl(d):
    if len(d) == 0:
        return []
    else:
        return [{k: d[k][i] for k in d.keys()} for i in range(len(d["id"]))]


def jsonl_to_dict(l):
    if len(l) == 0:
        return {}
    else:
        return {k: [s[k] for s in l] for k in l[0].keys()}


def same_state(s1 : NestedPipelineState, s2 : NestedPipelineState):
    if len(s1) != len(s2):
        return False
    for ss1, ss2 in zip(s1, s2):
        if ss1.name != ss2.name or ss1.step != ss2.step:
            return False
    return True