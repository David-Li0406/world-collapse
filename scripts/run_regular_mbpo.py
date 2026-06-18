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

# Run the upstream script AS __main__ via runpy (not import+call): hydra's
# @hydra.main(config_path='cfgs') only resolves 'cfgs' as a filesystem dir when
# __name__=='__main__'; importing it makes hydra look for a 'cfgs' package and
# fail. The VideoPredictor patch above persists through sys.modules, so the
# runpy'd script picks up the FP32 class. cwd is iVideoGPT/mbrl (set by the
# caller), where train_metaworld_mbpo.py + cfgs/ live.
import os  # noqa: E402
import runpy  # noqa: E402

if __name__ == "__main__":
    script = os.path.join(os.getcwd(), "train_metaworld_mbpo.py")
    runpy.run_path(script, run_name="__main__")
