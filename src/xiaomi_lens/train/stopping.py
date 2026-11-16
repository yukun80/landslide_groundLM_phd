"""独立于显卡的停止规则；epoch 从0开始，0仅训练区域，不计联合训练。"""

from dataclasses import dataclass, asdict
import math


@dataclass
class StopState:
    delta: float = 0.002
    patience: int = 4
    minimum_joint: int = 5
    anchor: float | None = None
    bad_epochs: int = 0
    joint_epochs: int = 0
    last_epoch: int = -1
    best: float = -1.0
    best_epoch: int = -1
    scores: list | None = None

    def update(self, epoch, score):
        if epoch != self.last_epoch + 1 or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("轮次必须连续，分数必须有效")
        self.last_epoch = epoch
        self.scores = (self.scores or []) + [float(score)]
        if score > self.best:
            self.best, self.best_epoch = float(score), epoch
        if epoch == 0:
            return False
        self.joint_epochs += 1
        if self.anchor is None or score - self.anchor >= self.delta - 1e-12:
            self.anchor, self.bad_epochs = float(score), 0
        else:
            self.bad_epochs += 1
        return self.joint_epochs >= self.minimum_joint and self.bad_epochs >= self.patience

    def outcome(self, stopped):
        if not stopped:
            return "epoch_limit_not_confirmed_stable"
        if self.best <= 0:
            return "no_detection_progress"
        recent = self.scores[-self.patience:]
        if max(recent) < self.best - self.delta:
            return "no_improvement_with_regression"
        if max(recent) - min(recent) >= self.delta:
            return "no_improvement_but_fluctuating"
        return "validation_score_plateau"

    def to_dict(self):
        return asdict(self)
