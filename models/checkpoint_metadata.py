from dataclasses import dataclass
from enum import Enum, auto

class ModelType(Enum):
    UNIGRAM = 'unigram'
    
@dataclass
class CheckpointMetadata:
    wandb_id: str
    step: int
    cfg: dict
    
    def to_dict(self):
        return {'wandb_id': self.wandb_id, 'step': self.step, 'cfg': self.cfg}
    
    def from_dict(d):
        return CheckpointMetadata(wandb_id=d['wandb_id'], step=d['step'], cfg=d['cfg'])