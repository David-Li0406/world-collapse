"""Run the REGULAR upstream iVideoGPT MBPO (train_metaworld_mbpo.py) with the
FP32 VideoPredictor.rollout fix applied at the class level.

The upstream WM generation wraps the LLM generate() in bf16 autocast, which on
this hardware produces inf/NaN logits -> torch.multinomial -> SIGFPE/CUDA assert
right at start_mbpo. This is a pure numerical-stability fix (FP32 + logits
sanitizer); it does NOT change the training dynamics, so the run remains the
regular wm-policy training. Same patch our src/wcollapse pipeline already uses.

Usage (from iVideoGPT/mbrl, with mbrl+iVideoGPT on PYTHONPATH):
    python <repo>/scripts/run_regular_mbpo.py <hydra overrides...>
"""
import contextlib
import torch
from transformers import LogitsProcessor, LogitsProcessorList

import video_predictor  # noqa: F401  (iVideoGPT/mbrl on PYTHONPATH)
from video_predictor import VideoPredictor, symexp


class _NoOpScaler:
    def scale(self, loss): return loss
    def unscale_(self, optimizer): pass
    def step(self, optimizer): optimizer.step()
    def update(self): pass


class _SanitizeLogits(LogitsProcessor):
    def __call__(self, input_ids, scores):
        return torch.where(torch.isfinite(scores), scores,
                           torch.full_like(scores, -1e4))


_SAFE_LOGITS = LogitsProcessorList([_SanitizeLogits()])

_orig_init = VideoPredictor.__init__


def _patched_init(self, *args, **kwargs):
    _orig_init(self, *args, **kwargs)
    # FP32 model+tokenizer (bf16 generate() SIGFPEs on this hardware).
    self.model.float()
    self.tokenizer.float()
    self.tok_scaler = _NoOpScaler()
    self.model_scaler = _NoOpScaler()


def _patched_rollout(self, obs, policy, horizon):
    with contextlib.nullcontext():
        B = obs.shape[0]
        args = self.args
        obs = obs.to(self.device) / 255.
        init_obs = obs
        current_frames = list(torch.chunk(obs, 3, dim=1))
        tokens_per_ctx = 256
        tokens_per_dyn = 16
        context_frames = torch.stack(current_frames[-args.context_length:], dim=1)
        tokens, _ = self.tokenizer.tokenize(
            torch.cat((context_frames, torch.zeros_like(context_frames)), dim=1),
            args.context_length,
        )
        tokens = tokens[:, :args.context_length * (tokens_per_ctx + 1)]
        init_tokens = tokens
        embeds = self.model.get_input_embeddings(tokens)
        cache = None
        obss, actions, rewards = [], [], []
        obs = init_obs
        for t in range(horizon):
            action = policy(obs, t)
            action_embeds = self.model.action_linear(action)
            embeds[:, -1] += action_embeds
            result = self.model.llm.generate(
                inputs_embeds=embeds,
                do_sample=True, temperature=1.0,
                pad_token_id=50256, top_k=100,
                use_cache=True,
                max_new_tokens=tokens_per_dyn + 1,
                return_dict_in_generate=True,
                output_hidden_states=True,
                logits_processor=_SAFE_LOGITS,
            )
            predicted_token = result.sequences[:, :-1]
            last_layer_hidden_states = result.hidden_states[-1]
            last_token_states = last_layer_hidden_states[-1]
            reward = self.model.reward_linear(last_token_states).squeeze(-2)
            cat_predicted_token = (torch.concat(
                [predicted_token,
                 (torch.ones(B) * self.model.token_for_sdf).unsqueeze(1).to(self.device)],
                dim=1).to(predicted_token.dtype))
            embeds = torch.concat(
                [embeds, self.model.get_input_embeddings(cat_predicted_token)], dim=1)
            fmap, cache = self.tokenizer.detokenize(
                torch.concat([init_tokens, predicted_token], dim=1),
                args.context_length, cache=cache, return_cache=True,
            )
            fmap = fmap.clamp(0.0, 1.0)
            current_frames.append(fmap[:, -1])
            current_frames.pop(0)
            obs = torch.cat(current_frames, dim=1)
            obss.append(obs)
            actions.append(action)
            rewards.append(reward)
    obss = [init_obs] + obss
    actions = [torch.zeros_like(actions[0])] + actions
    rewards = [torch.zeros_like(rewards[0])] + rewards
    if self.args.symlog:
        rewards = [symexp(r) for r in rewards]
    return (torch.stack(obss, 1).float(),
            torch.stack(actions, 1).float(),
            torch.stack(rewards, 1).float())


VideoPredictor.__init__ = _patched_init
VideoPredictor.rollout = _patched_rollout

import os  # noqa: E402
import sys  # noqa: E402
import runpy  # noqa: E402

# ---------------------------------------------------------------------------
# Optional collapse_prone recency-WM: restrict the world model's segment sampler
# to the most recent WMC_WM_RECENT_WINDOW transitions, so the WM forgets older
# data (incl. held-out region B) — the variance/collapse driver. Patches ONLY
# ReplaySegmentBuffer (the WM loader); the policy's replay stays full. Class-
# level patch persists through sys.modules into both the runpy and import paths.
# ---------------------------------------------------------------------------
_RECENT = os.environ.get("WMC_WM_RECENT_WINDOW", "").strip()
if _RECENT:
    import random as _random
    from replay_buffer import ReplaySegmentBuffer, episode_len as _eplen
    _W = int(_RECENT)
    _rp = {"done": False}

    def _recent_sample_episode(self):
        fns = self._episode_fns
        chosen, total = [], 0
        for fn in reversed(fns):           # _episode_fns is chronological
            chosen.append(fn)
            total += _eplen(self._episodes[fn])
            if total >= _W:
                break
        if not _rp["done"]:
            print(f"[recency-wm] WM trains on last {len(chosen)} eps "
                  f"(~{total} transitions, window={_W}); total stored={len(fns)}",
                  flush=True)
            _rp["done"] = True
        return self._episodes[_random.choice(chosen)]

    ReplaySegmentBuffer._sample_episode = _recent_sample_episode

# ---------------------------------------------------------------------------
# Optional "our wm-policy setting": goal-bias trained/held-out (A/B) split.
# Enabled by WMC_GOAL_BIAS_FRACTION (e.g. 0.5). Data collection is restricted to
# region A (rand_vec dim-0 lower fraction); eval reports SR_A (trained) and SR_B
# (held-out upper region). Minimal runtime patch — upstream files untouched.
# ---------------------------------------------------------------------------
_BIAS = os.environ.get("WMC_GOAL_BIAS_FRACTION", "").strip()


def _run_biased():
    import numpy as np
    import torch
    import drq_utils
    import metaworld_env
    from metaworld_env import MetaWorld
    import train_metaworld_mbpo as T
    from hydra import compose, initialize_config_dir

    frac = float(_BIAS)
    region = {"mode": "A"}          # training collects in region A
    printed = {"done": False}
    _orig_reset = MetaWorld.reset

    def _biased_reset(self):
        m = region["mode"]
        if m in ("A", "B"):
            rs = self._env._random_reset_space
            lo = np.asarray(rs.low, dtype=np.float64)
            hi = np.asarray(rs.high, dtype=np.float64)
            rv = np.random.uniform(lo, hi)
            split = lo[0] + frac * (hi[0] - lo[0])
            rv[0] = (np.random.uniform(lo[0], split) if m == "A"
                     else np.random.uniform(split, hi[0]))
            self._env._freeze_rand_vec = True
            self._env._last_rand_vec = rv.astype(np.float32)
            if not printed["done"]:
                print(f"[goal-bias] rand_vec dim0=[{lo[0]:.3f},{hi[0]:.3f}] "
                      f"split@{split:.3f} frac={frac}  A=lower, B=upper", flush=True)
                printed["done"] = True
        return _orig_reset(self)

    MetaWorld.reset = _biased_reset

    def _ab_eval(self):
        # Evaluate trained region A and held-out region B; print SR for each so
        # it lands in train.log (the reliable channel). Restore A for training.
        for reg in ("A", "B"):
            region["mode"] = reg
            episode = 0
            total_success = 0
            until = drq_utils.Until(self.cfg.num_eval_episodes, bar_name=f"eval_{reg}")
            while until(episode):
                ts = self.eval_env.reset()
                ep_succ = 0
                while not ts.last():
                    with torch.no_grad(), drq_utils.eval_mode(self.agent):
                        action = self.agent.act(ts.observation, self.global_step,
                                                eval_mode=True)
                    ts = self.eval_env.step(action)
                    ep_succ += ts.success
                total_success += float(ep_succ >= 1.0)
                episode += 1
            sr = total_success / episode
            print(f"[ab-eval] | eval | F: {self.global_frame} | "
                  f"SS_{reg}: {sr:.4f}", flush=True)
        region["mode"] = "A"

    T.Workspace.eval = _ab_eval

    # Manual hydra compose (we drop hydra.* overrides; output dir via WMC_OUT).
    out = os.environ["WMC_OUT"]
    os.makedirs(out, exist_ok=True)
    cfgs_dir = os.path.join(os.path.dirname(os.path.abspath(T.__file__)), "cfgs")
    overrides = [a for a in sys.argv[1:] if not a.startswith("hydra.")]
    with initialize_config_dir(version_base=None, config_dir=cfgs_dir):
        cfg = compose(config_name="mbpo_config", overrides=overrides)
    os.chdir(out)  # Workspace uses Path.cwd() as work_dir (logs/eval here)
    ws = T.Workspace(cfg)
    ws.train()


if __name__ == "__main__":
    if _BIAS:
        _run_biased()
    else:
        # Regular training: run upstream AS __main__ via runpy (hydra's
        # config_path='cfgs' only resolves as a dir under __main__).
        script = os.path.join(os.getcwd(), "train_metaworld_mbpo.py")
        runpy.run_path(script, run_name="__main__")
