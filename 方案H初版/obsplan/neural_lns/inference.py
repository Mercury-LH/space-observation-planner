"""Standalone checkpoint scoring with the training-time feature transformation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from ..famou_h_eval import FEATURES
from .model import CandidateMLP
from .static_data import file_sha256, transform


class Ranker:
    def __init__(self, checkpoint, device='cpu', *, expected_sha256=None, expected_versions=None, allow_static_domain_transfer=False):
        begin = time.perf_counter()
        self.checkpoint = Path(checkpoint)
        self.sha256 = file_sha256(self.checkpoint)
        if expected_sha256 is not None and self.sha256 != expected_sha256:
            raise ValueError('checkpoint SHA-256 mismatch')
        self.payload = torch.load(self.checkpoint, map_location='cpu', weights_only=True)
        if self.payload.get('format_version') != 1:
            raise ValueError('unsupported checkpoint format')
        self.normalizer = self.payload['normalizer']
        if self.normalizer.get('feature_names') != list(FEATURES) or self.normalizer.get('source_split') != 'train':
            raise ValueError('checkpoint lacks canonical train-only normalization')
        self.dynamic_normalizer = self.payload['dynamic_normalizer']
        self.feature_versions = self.payload['feature_versions']
        self.domain_transfer = None
        if expected_versions and any(self.feature_versions.get(k) != v for k, v in expected_versions.items()):
            if self.payload['stage'] == 'static' and allow_static_domain_transfer:
                self.domain_transfer = {'source': self.feature_versions, 'target': dict(expected_versions),
                                        'explicitly_allowed': True}
            else:
                raise ValueError('checkpoint feature/pool/price versions differ from inference')
        if self.payload['schema'].get('static_names') != list(FEATURES):
            raise ValueError('checkpoint static schema column order differs')
        if self.payload['dynamic_dim'] and len(self.payload['schema'].get('dynamic_names', [])) != self.payload['dynamic_dim']:
            raise ValueError('checkpoint dynamic schema column count differs')
        self.device = device
        self.model = CandidateMLP(self.payload['dynamic_dim']).to(device)
        self.model.load_state_dict(self.payload['model_state'])
        self.model.eval()
        self._synchronize()
        self.load_s = time.perf_counter() - begin
        self.inference_s = 0.
        self.calls = 0

    def _synchronize(self):
        if str(self.device).startswith('cuda'):
            torch.cuda.synchronize(self.device)

    @torch.no_grad()
    def score(self, static_features, dynamic_features=None):
        begin = time.perf_counter()
        x = torch.from_numpy(transform(static_features, self.normalizer)).to(self.device)
        d = None
        if self.model.dynamic_dim:
            if dynamic_features is None:
                raise ValueError('dynamic checkpoint requires dynamic inputs')
            d = torch.from_numpy(transform(dynamic_features, self.dynamic_normalizer,
                                          dim=self.model.dynamic_dim)).to(self.device)
        elif dynamic_features is not None:
            raise ValueError('static checkpoint accepts only static inputs')
        result = self.model(x, d).cpu().numpy()
        self._synchronize()
        self.inference_s += time.perf_counter() - begin
        self.calls += 1
        if not np.isfinite(result).all():
            raise FloatingPointError('nonfinite scores; use deterministic rule fallback')
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--input', required=True, type=Path, help='NPZ: features [N,20], optional dynamic_features [N,D]')
    parser.add_argument('--output', required=True, type=Path, help='new NPZ containing scores and original candidate_id')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--sha256')
    parser.add_argument('--batch-size', type=int, default=8192)
    args = parser.parse_args(argv)
    if args.batch_size <= 0 or args.output.exists():
        raise ValueError('positive batch size and new output required')
    ranker = Ranker(args.checkpoint, args.device, expected_sha256=args.sha256)
    with np.load(args.input, allow_pickle=False) as z:
        x = z['features']
        d = z['dynamic_features'] if 'dynamic_features' in z else None
        ids = z['candidate_id'] if 'candidate_id' in z else np.arange(len(x), dtype=np.int64)
    if ids.shape != (len(x),):
        raise ValueError('candidate_id must preserve one id per input row')
    scores = [ranker.score(x[i:i + args.batch_size], d[i:i + args.batch_size] if d is not None else None)
              for i in range(0, len(x), args.batch_size)]
    with args.output.open('xb') as fh:
        np.savez_compressed(fh, scores=np.concatenate(scores) if scores else np.empty(0, np.float32), candidate_id=ids)
    print(json.dumps({'checkpoint_sha256': ranker.sha256, 'rows': len(x), 'load_s': ranker.load_s,
                      'inference_s': ranker.inference_s, 'output_sha256': file_sha256(args.output),
                      'semantics': 'ranking scores, not probabilities or legality labels'}, indent=2))


if __name__ == '__main__':
    main()
