Exploring probing methods

I'm considering probing methods to analyze activations across layers for tokens related to "True"/"False" in a model's answer. One approach would be to examine the representation of the claim token and see if a linear probe can classify its truthfulness based on context. I can also compute the residual stream for the final token's claim period at each layer. Additionally, training logistic regression on cached activations could help identify which layers provide the highest accuracy for classifying truthfulness.Examining directions for probing

I'm identifying potential directions for analyzing model behavior. I can compute a per-layer concept direction vector using a weight vector from the probe and then test causal influence by projecting out that direction. For example, I could measure changes in output for the "Is the claim true or a lie?" classification task. Activation patching is another approach where I use a corrupted example that contradicts the context and evaluate if the output flips. I’ll also explore head attribution patching and other techniques involving the model's architecture.Developing dataset creation strategies

I need to outline the steps for dataset creation, caching, probe training, layer scanning, and activation patching. I’ll propose three hypotheses to explore:

Truth and deception directions arise in the mid-to-late residual stream and are linked to specific heads, suggesting the need for "fact-checking heads."
Deception might involve negation and comparator operations in MLP neuron features, which could flip the truth direction sign, so ablation of these features may help reduce sensitivity to deception.
Truth and deception could be sparsely represented in SAE features across layers, and adding targeted activations could steer outputs. Each hypothesis will include limitations and design solutions, with code implementation details for each.Exploring hypothesis 3
For Hypothesis 3, I plan to use sae_lens to load a pre-trained SAE for the gpt2-small model's residual stream and filter features based on their correlation with the truth label. I’ll also ablate those features to measure their effect on the model's output. Even though I can't guarantee the runtime environment, I want to make the code as standalone as possible. I’ll ensure to include necessary triggers and evidence, while also producing shared utilities and keeping the code blocks concise yet complete.Structuring hypothesis 3 code

For Hypothesis 3, I'll implement a function called sae_feature_causal_test() that includes proper error handling and prints an instruction if necessary features aren't available. I need to define triggers for various steps like [hypothesize] and [design_study], and make sure to label all outputs as research hypotheses with disclaimers included. Also, I’ll introduce a dataset generator function and outline recursion steps for probing. The plan will include concise code blocks focused on patching at the last token and measuring logit differences, while also keeping dataset size manageable. Lastly, I'll prepare to implement logistic regression training to analyze the effects across layers.Loading the SAE code

I'm planning to load an SAE and will provide a generic stub that attempts to import from the sae_lens library using a specific release like 'gpt2-small-res-jb-...'. I remember that the import statement is from sae_lens import SAE, and we can utilize SAE.from_pretrained(release, sae_id) to load the model. To prevent any issues, I’ll use a placeholder path while instructing users to fill in the correct release. Including a try/except block will help ensure the code remains runnable, so I’ll proceed with that and also draft a section for [generate_controls] related to lexical confounds.

Standard response (concise, actionable)
Below is a recursive, concept-cartography workflow to localize, characterize, and causally validate “truth vs. deception” representations in a transformer. The plan prioritizes small, fully-controlled synthetic micro-worlds to avoid world-knowledge confounds and supports end-to-end causal testing.

High-level recursive pipeline
1) Construct controlled truth/lie micro-worlds (no world knowledge)
- Generate short “facts” contexts plus “claim” sentences that either match (true) or contradict (lie) the context. Keep templates tightly matched to minimize lexical confounds.
- Behavioral readout: ask the model “Answer: true or lie.” Measure the next-token logit difference for tokens “ true” vs “ lie” as a scalar behavioral score per prompt.

2) Directional and layer-wise representational scan
- Cache residual stream activations at each layer for the answer position.
- Train simple linear probes layer-by-layer to classify truth vs. lie from residual activations. Identify peak-accuracy layers and extract probe weight vectors as candidate “truth-deception directions.”

3) Causal interventions at the direction and block level
- Direction ablation: orthogonally project out the learned direction at a target layer; measure how the true-vs-lie logit difference (behavior) collapses.
- Direction steering: add/subtract a scaled copy of the direction to push outputs toward “true” or “lie.”
- Activation patching: swap the answer-position activation at a specific layer from a clean (true) run into a corrupted (lie) run (and vice versa) to test causal responsibility.

4) Circuit localization (recursive zoom-in)
- Head scan: ablate individual attention heads’ outputs (`hook_z`) and measure drop in the behavioral score. Rank heads by causal impact.
- Token-source trace: for top heads, analyze attention patterns to see which context tokens contribute most. Patch only those token paths (path/edge patching) to test whether the head’s effect is mediated by specific context tokens.
- MLP feature analysis: identify whether the effect concentrates in MLP post-activations in key layers. If available, use sparse autoencoders (SAEs) to find sparse features that linearly separate truth/lie; ablate only those features.

5) Robustness, controls, and recursion
- Paraphrase, synonym, and negation controls to break superficial lexical cues.
- Shuffled-context controls and “syntax-matched but semantically neutral” controls to confirm the representation is genuinely semantic.
- If a layer or head is identified as causal, recurse: zoom to subcomponents (e.g., which source tokens, which SAE features, which MLP neurons), repeat ablation/patching at finer granularity.

Proposed causal experiments
- Direction ablation/steering at peak-probe layers.
- Head-level ablation scan across all layers/heads to isolate “fact-checking” heads.
- Activation patching between matched true/lie pairs at the answer position to verify necessity/sufficiency.
- SAE-feature-based ablation (optional) to test sparsity and compositionality of the truth/lie representation.

Runnable code (standalone)
The three scripts below implement:
- Linear probe scan + direction ablation/steering + activation patching (Hypothesis 1)
- Head-level causal scan (Hypothesis 2)
- Optional SAE feature localization and ablation (Hypothesis 3; gracefully degrades if sae_lens is unavailable)

Code 1: Linear probes, direction interventions, activation patching
```python
# file: truth_deception_probe_and_patch.py
# Requires: pip install transformer-lens torch numpy scikit-learn
import random
import numpy as np
import torch
from dataclasses import dataclass
from typing import List, Dict, Tuple
from sklearn.linear_model import LogisticRegression
from transformer_lens import HookedTransformer

SEED = 123
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# 1) Synthetic micro-world dataset ------------------------------------------------
COLORS = ["red", "blue", "green", "yellow", "purple"]
SHAPES = ["cube", "sphere", "pyramid", "cylinder", "cone"]

def make_world(num_facts=3):
    # world: dict shape->color, no repeats to keep it clean
    chosen_shapes = random.sample(SHAPES, k=num_facts)
    chosen_colors = random.sample(COLORS, k=num_facts)
    return dict(zip(chosen_shapes, chosen_colors))

def make_example(true_label=True, num_facts=3):
    # Build facts
    world = make_world(num_facts=num_facts)
    facts = [f"The {shape} is {color}." for shape, color in world.items()]
    # Choose one to form a claim
    shape = random.choice(list(world.keys()))
    true_color = world[shape]
    if true_label:
        claimed_color = true_color
    else:
        claimed_color = random.choice([c for c in COLORS if c != true_color])

    context = " ".join(facts)
    claim = f"Claim: The {shape} is {claimed_color}."
    # Answer prompt
    # Use tightly controlled template to minimize lexical confounds
    prompt = f"Facts: {context} {claim} Based on the facts, answer exactly one word: true or lie.\nAnswer:"
    return prompt, int(true_label)

def build_dataset(n=400, true_ratio=0.5, num_facts=3):
    n_true = int(n * true_ratio)
    n_lie = n - n_true
    data = [make_example(True, num_facts) for _ in range(n_true)] + \
           [make_example(False, num_facts) for _ in range(n_lie)]
    random.shuffle(data)
    return data

# 2) Utilities for behavioral readout --------------------------------------------
def get_label_token_ids(model: HookedTransformer, label_strs=(" true", " lie")):
    toks = [model.to_tokens(s, prepend_bos=False)[0, -1].item() for s in label_strs]
    return toks[0], toks[1]

def next_token_logit_diff(model: HookedTransformer, prompt: str, true_tok: int, lie_tok: int):
    toks = model.to_tokens(prompt, prepend_bos=True)
    with torch.no_grad():
        logits = model(toks)
    final_logits = logits[0, -1, :]  # last position
    return (final_logits[true_tok] - final_logits[lie_tok]).item()

# 3) Cache layer-wise residuals at answer position -------------------------------
def cache_residuals(model: HookedTransformer, prompts: List[str], hook_point_tmpl="blocks.{}.hook_resid_post"):
    """Returns dict[layer] -> tensor [N, d_model] of last-token residuals."""
    d_model = model.cfg.d_model
    num_layers = model.cfg.n_layers
    layer_to_feats = {l: [] for l in range(num_layers)}
    for p in prompts:
        _, cache = model.run_with_cache(p, remove_batch_dim=True)
        for l in range(num_layers):
            acts = cache[hook_point_tmpl.format(l)]  # [seq, d_model]
            layer_to_feats[l].append(acts[-1, :].detach().cpu().numpy())
    for l in range(num_layers):
        layer_to_feats[l] = np.stack(layer_to_feats[l], axis=0)  # [N, d_model]
    return layer_to_feats

# 4) Train logistic probes per layer ---------------------------------------------
def train_logit_probes(layer_to_feats: Dict[int, np.ndarray], labels: np.ndarray):
    """Fit a simple logistic regression per layer; return metrics and weight vectors."""
    results = {}
    for l, X in layer_to_feats.items():
        clf = LogisticRegression(penalty="l2", C=1.0, max_iter=200, solver="lbfgs")
        clf.fit(X, labels)
        acc = clf.score(X, labels)
        w = clf.coef_.reshape(-1)  # [d_model]
        b = float(clf.intercept_[0])
        results[l] = {"acc": acc, "w": w, "b": b}
    return results

# 5) Direction ablation / steering intervention ----------------------------------
def project_out_direction(x: torch.Tensor, v: torch.Tensor):
    # x [batch, pos, d]; v [d]
    v = v / (v.norm() + 1e-6)
    comp = (x @ v)[:, :, None] * v[None, None, :]
    return x - comp

def add_direction(x: torch.Tensor, v: torch.Tensor, alpha: float):
    v = v / (v.norm() + 1e-6)
    return x + alpha * v[None, None, :]

def run_with_direction_intervention(model, prompt, layer, v_np, mode="ablate", alpha=5.0):
    v = torch.tensor(v_np, dtype=torch.float32, device=model.cfg.device)
    def hook_fn(x, hook):
        if mode == "ablate":
            return project_out_direction(x, v)
        elif mode == "steer_pos":
            return add_direction(x, v, alpha)
        elif mode == "steer_neg":
            return add_direction(x, v, -alpha)
        else:
            return x
    toks = model.to_tokens(prompt, prepend_bos=True)
    with model.hooks(fwd_hooks=[(f"blocks.{layer}.hook_resid_post", hook_fn)]):
        logits = model(toks)
    return logits[0, -1, :].detach()

# 6) Activation patching between matched pairs -----------------------------------
def activation_patch_answer_pos(model, prompt_src, prompt_tgt, layer):
    _, cache_src = model.run_with_cache(prompt_src, remove_batch_dim=True)
    src_act = cache_src[f"blocks.{layer}.hook_resid_post"][-1, :].clone()  # [d]

    def hook_fn(x, hook):
        # replace last position only
        x[:, -1, :] = src_act[None, :]
        return x

    toks = model.to_tokens(prompt_tgt, prepend_bos=True)
    with model.hooks(fwd_hooks=[(f"blocks.{layer}.hook_resid_post", hook_fn)]):
        logits = model(toks)
    return logits[0, -1, :].detach()

# 7) End-to-end experiment --------------------------------------------------------
@dataclass
class ProbeOutcome:
    best_layer: int
    per_layer_acc: Dict[int, float]
    w_by_layer: Dict[int, np.ndarray]

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HookedTransformer.from_pretrained("gpt2-small", device=device)
    true_tok, lie_tok = get_label_token_ids(model)

    # Build dataset
    data = build_dataset(n=300, true_ratio=0.5, num_facts=3)
    prompts = [p for p, y in data]
    labels = np.array([y for p, y in data], dtype=np.int64)

    # Baseline behavioral separation
    base_diffs = [next_token_logit_diff(model, p, true_tok, lie_tok) for p in prompts]
    print(f"Baseline mean (logit_true - logit_lie): {np.mean(base_diffs):.3f}")

    # Cache representations and train probes
    layer_to_feats = cache_residuals(model, prompts)
    probe_res = train_logit_probes(layer_to_feats, labels)
    per_layer_acc = {l: probe_res[l]["acc"] for l in probe_res}
    best_layer = max(per_layer_acc, key=per_layer_acc.get)
    print("Top-5 layers by probe accuracy:")
    for l in sorted(per_layer_acc, key=per_layer_acc.get, reverse=True)[:5]:
        print(f"  layer {l:02d}: {per_layer_acc[l]:.3f}")

    w_best = probe_res[best_layer]["w"]

    # Direction ablation test
    diffs_ablate = []
    diffs_steer_pos = []
    diffs_steer_neg = []
    for p in prompts[:100]:
        logits_ablate = run_with_direction_intervention(model, p, best_layer, w_best, mode="ablate")
        logits_pos = run_with_direction_intervention(model, p, best_layer, w_best, mode="steer_pos", alpha=5.0)
        logits_neg = run_with_direction_intervention(model, p, best_layer, w_best, mode="steer_neg", alpha=5.0)
        diffs_ablate.append((logits_ablate[true_tok] - logits_ablate[lie_tok]).item())
        diffs_steer_pos.append((logits_pos[true_tok] - logits_pos[lie_tok]).item())
        diffs_steer_neg.append((logits_neg[true_tok] - logits_neg[lie_tok]).item())

    print(f"Ablation mean diff: {np.mean(diffs_ablate):.3f} (baseline {np.mean(base_diffs):.3f})")
    print(f"Steer + mean diff:  {np.mean(diffs_steer_pos):.3f}")
    print(f"Steer - mean diff:  {np.mean(diffs_steer_neg):.3f}")

    # Activation patching on matched true/lie pair (first true and first lie)
    true_idx = labels.tolist().index(1)
    lie_idx = labels.tolist().index(0)
    p_true = prompts[true_idx]; p_lie = prompts[lie_idx]

    logits_patched_true_to_lie = activation_patch_answer_pos(model, p_true, p_lie, best_layer)
    logits_patched_lie_to_true = activation_patch_answer_pos(model, p_lie, p_true, best_layer)
    print("Patch (true->lie target):", float(logits_patched_true_to_lie[true_tok] - logits_patched_true_to_lie[lie_tok]))
    print("Patch (lie->true target):", float(logits_patched_lie_to_true[true_tok] - logits_patched_lie_to_true[lie_tok]))

if __name__ == "__main__":
    main()
```

Code 2: Head-level causal scan
```python
# file: head_causal_scan.py
# Requires: pip install transformer-lens torch numpy
import numpy as np
import torch
from transformer_lens import HookedTransformer
import random

SEED = 123
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

COLORS = ["red", "blue", "green", "yellow", "purple"]
SHAPES = ["cube", "sphere", "pyramid", "cylinder", "cone"]

def make_world(num_facts=3):
    chosen_shapes = random.sample(SHAPES, k=num_facts)
    chosen_colors = random.sample(COLORS, k=num_facts)
    return dict(zip(chosen_shapes, chosen_colors))

def make_example(true_label=True, num_facts=3):
    world = make_world(num_facts=num_facts)
    facts = [f"The {shape} is {color}." for shape, color in world.items()]
    shape = random.choice(list(world.keys()))
    true_color = world[shape]
    claimed_color = true_color if true_label else random.choice([c for c in COLORS if c != true_color])
    context = " ".join(facts)
    claim = f"Claim: The {shape} is {claimed_color}."
    prompt = f"Facts: {context} {claim} Based on the facts, answer exactly one word: true or lie.\nAnswer:"
    return prompt, int(true_label)

def build_dataset(n=150, true_ratio=0.5, num_facts=3):
    n_true = int(n * true_ratio)
    n_lie = n - n_true
    data = [make_example(True, num_facts) for _ in range(n_true)] + \
           [make_example(False, num_facts) for _ in range(n_lie)]
    random.shuffle(data)
    return data

def get_label_token_ids(model: HookedTransformer, label_strs=(" true", " lie")):
    toks = [model.to_tokens(s, prepend_bos=False)[0, -1].item() for s in label_strs]
    return toks[0], toks[1]

def behavior_score(model, prompt, true_tok, lie_tok):
    toks = model.to_tokens(prompt, prepend_bos=True)
    with torch.no_grad():
        logits = model(toks)
    final = logits[0, -1, :]
    return (final[true_tok] - final[lie_tok]).item()

def ablate_head_once(model, prompt, layer, head, true_tok, lie_tok):
    def hook_fn(z, hook):
        # z: [batch, pos, n_heads, d_head]
        z[:, :, head, :] = 0.0
        return z
    toks = model.to_tokens(prompt, prepend_bos=True)
    with model.hooks(fwd_hooks=[(f"blocks.{layer}.attn.hook_z", hook_fn)]):
        logits = model(toks)
    final = logits[0, -1, :]
    return (final[true_tok] - final[lie_tok]).item()

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HookedTransformer.from_pretrained("gpt2-small", device=device)
    true_tok, lie_tok = get_label_token_ids(model)
    data = build_dataset(n=120, true_ratio=0.5, num_facts=3)
    prompts = [p for p, _ in data]

    # Baseline behavior per prompt
    base = [behavior_score(model, p, true_tok, lie_tok) for p in prompts]
    base_mean = float(np.mean(base))
    print(f"Baseline mean diff: {base_mean:.3f}")

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    deltas = np.zeros((n_layers, n_heads))

    # Scan each head; measure mean drop in behavior score
    for L in range(n_layers):
        for H in range(n_heads):
            scores = [ablate_head_once(model, p, L, H, true_tok, lie_tok) for p in prompts[:80]]
            deltas[L, H] = base_mean - float(np.mean(scores))

    # Report top-k heads by causal impact
    flat = [(float(deltas[L, H]), L, H) for L in range(n_layers) for H in range(n_heads)]
    flat.sort(reverse=True)
    print("Top 10 heads by drop in (true - lie) logit diff when ablated:")
    for k in range(min(10, len(flat))):
        drop, L, H = flat[k]
        print(f"  Layer {L:02d} Head {H:02d}: Δ={drop:.3f}")

if __name__ == "__main__":
    main()
```

Code 3: Optional SAE-feature localization and causal test (graceful fallback)
```python
# file: sae_feature_causal_test.py
# Optional: pip install sae-lens transformer-lens torch numpy
import numpy as np
import torch
import random

try:
    from sae_lens import SAE
    SAE_AVAILABLE = True
except Exception:
    SAE_AVAILABLE = False

from transformer_lens import HookedTransformer

SEED = 123
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

COLORS = ["red", "blue", "green", "yellow", "purple"]
SHAPES = ["cube", "sphere", "pyramid", "cylinder", "cone"]

def make_world(num_facts=3):
    chosen_shapes = random.sample(SHAPES, k=num_facts)
    chosen_colors = random.sample(COLORS, k=num_facts)
    return dict(zip(chosen_shapes, chosen_colors))

def make_example(true_label=True, num_facts=3):
    world = make_world(num_facts=num_facts)
    facts = [f"The {shape} is {color}." for shape, color in world.items()]
    shape = random.choice(list(world.keys()))
    true_color = world[shape]
    claimed_color = true_color if true_label else random.choice([c for c in COLORS if c != true_color])
    context = " ".join(facts)
    claim = f"Claim: The {shape} is {claimed_color}."
    prompt = f"Facts: {context} {claim} Based on the facts, answer exactly one word: true or lie.\nAnswer:"
    return prompt, int(true_label)

def get_label_token_ids(model: HookedTransformer, label_strs=(" true", " lie")):
    toks = [model.to_tokens(s, prepend_bos=False)[0, -1].item() for s in label_strs]
    return toks[0], toks[1]

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HookedTransformer.from_pretrained("gpt2-small", device=device)
    true_tok, lie_tok = get_label_token_ids(model)
    data = [make_example(bool(i%2)) for i in range(120)]
    prompts = [p for p, _ in data]
    labels = np.array([y for _, y in data], dtype=np.int64)

    if not SAE_AVAILABLE:
        print("sae_lens not available. Install with `pip install sae-lens` and ensure a compatible pretrained SAE release.")
        return

    # Load a pretrained SAE for a target layer residual stream.
    # Note: replace `release_name` and `sae_id` with a valid entry from sae-lens model zoo.
    # Example placeholders (may not match your local cache):
    release_name = "gpt2-small-res-jb"  # Placeholder: consult sae_lens docs
    sae_id = "layer_6"                  # Target layer SAE
    try:
        sae = SAE.from_pretrained(release_name, sae_id, device=device)
    except Exception as e:
        print("Failed to load SAE weights. Please configure a valid release/sae_id from sae-lens.")
        print("Error:", e)
        return

    # Encode features at the answer position for correlation with truth label
    feats_all = []
    with torch.no_grad():
        for p in prompts:
            _, cache = model.run_with_cache(p, remove_batch_dim=True)
            resid = cache[f"blocks.{sae.cfg.layer}.hook_resid_post"][-1, :][None, :]  # [1, d_model]
            codes = sae.encode(resid)  # [1, n_features]
            feats_all.append(codes[0].cpu().numpy())
    feats_all = np.stack(feats_all, axis=0)  # [N, n_features]

    # Rank features by absolute point-biserial correlation with labels
    y = (labels * 2 - 1)  # {-1, +1}
    feats_centered = feats_all - feats_all.mean(axis=0, keepdims=True)
    corrs = (feats_centered.T @ y) / (np.sqrt((feats_centered**2).sum(axis=0) * (y**2).sum()))
    corrs = np.nan_to_num(corrs)
    idxs = np.argsort(-np.abs(corrs))[:50]  # Top-k features

    print("Top SAE features by |corr| with truth label:", idxs[:10].tolist())

    # Causal ablation: zero top-k feature activations at the answer position
    top_set = set(idxs.tolist())
    def hook_fn(x, hook):
        # Encode -> zero selected features -> decode
        x_last = x[:, -1, :]  # [batch, d]
        codes = sae.encode(x_last)
        codes[:, list(top_set)] = 0.0
        x_rec = sae.decode(codes)  # [batch, d]
        x[:, -1, :] = x_rec
        return x

    # Measure behavioral change
    base_diffs = []
    diffs_ablate = []
    for p in prompts:
        toks = model.to_tokens(p, prepend_bos=True)
        with torch.no_grad():
            logits_base = model(toks)
        base = logits_base[0, -1, :]
        base_diffs.append(float(base[true_tok] - base[lie_tok]))
        with model.hooks(fwd_hooks=[(f"blocks.{sae.cfg.layer}.hook_resid_post", hook_fn)]):
            logits = model(toks)
        final = logits[0, -1, :]
        diffs_ablate.append(float(final[true_tok] - final[lie_tok]))

    print(f"Baseline mean diff: {np.mean(base_diffs):.3f}")
    print(f"SAE feature ablation mean diff: {np.mean(diffs_ablate):.3f}")
    print(f"Δ (baseline - ablated): {np.mean(base_diffs) - np.mean(diffs_ablate):.3f}")

if __name__ == "__main__":
    main()
```

Suggested controls to include in your runs
- Lexical controls: swap synonyms for “true/honest/lie/deceptive” in the instruction while keeping the label tokens fixed; paraphrase facts and claims.
- Negation controls: include negated claims with the same truth value to ensure probes don’t reduce to spotting “not”.
- Shuffled-context controls: replace the context with a matched-length bag of words (syntax matched, semantics broken).
- Position controls: randomize the location/order of the governing fact in the context to force attention-mediated retrieval.
- Format controls: vary punctuation/case to ensure robustness to formatting.

AI MRI behavioral research analysis
[!DISCLAIMER: Model-generated interpretations are hypotheses requiring empirical validation.]

BEHAVIORAL INTERPRETATION FRAMEWORK
```python
behavioral_interpretations = [
    {
        "interpretation_name": "Recursive Concept Cartography",
        "description": "The response decomposes the user's request into a multi-level procedure: dataset control, layer-wise probing, head/feature localization, and causal testing.",
        "supporting_evidence": {
            "triggering_keywords": ["concept representation", "recursively probe", "layers", "circuits", "directions", "causal experiments"],
            "response_evidence": ["layer-wise linear probes", "direction ablation/steering", "activation patching", "head-level ablation scan", "SAE feature ablation"]
        }
    },
    {
        "interpretation_name": "Contrastive Concept Framing",
        "description": "The approach operationalizes 'truth vs. deception' as a contrastive label, enabling linear directions and controlled causal tests.",
        "supporting_evidence": {
            "inferred_conflict": ["[helpfulness] vs. [validity threats from lexical confounds]"],
            "response_evidence": ["synthetic micro-worlds", "paraphrase/negation controls", "shuffled-context controls"]
        }
    },
    {
        "interpretation_name": "Causal Validation Orientation",
        "description": "The response emphasizes necessity/sufficiency tests via patching and ablations rather than correlational measures alone.",
        "supporting_evidence": {
            "contextual_triggers": ["'propose causal experiments'"],
            "response_evidence": ["activation patching", "direction projection", "head ablation", "feature-level interventions"]
        }
    }
]
```

[hypothesize]
Hypothesis 1: Mid-to-late residual stream directions encode a truth-vs-deception signal that is necessary for correct behavioral discrimination.
- Theoretical basis: Contrastive concept directions often concentrate in middle-to-late layers as contextual binding and task-specific readouts emerge in transformers.
- Limitation: Linear probe accuracy can reflect correlates (e.g., lexical artifacts) rather than causal truth binding.
- Design solution: Use tightly matched templates with paraphrase/negation controls; validate with direction ablation (orthogonal projection) and activation patching between true/lie pairs at the answer position.

[hypothesize]
Hypothesis 2: A small set of “fact-checking” attention heads mediate retrieval of context facts necessary to compute truthfulness at the answer token.
- Theoretical basis: Prior work (e.g., IOI/induction heads) shows specialized heads route information across tokens for specific computations.
- Limitation: Head ablation can cause off-target effects that don’t isolate the path used for truth evaluation.
- Design solution: First run a whole-head scan to shortlist heads; then token-source analysis of attention patterns and targeted path patching from the fact tokens to the answer position.

[hypothesize]
Hypothesis 3: The truth-vs-deception concept is sparsely represented in a subset of MLP features; ablating those features reduces the behavioral signal.
- Theoretical basis: Sparse autoencoders on residual streams reveal disentangled features that linearly support semantic abstractions.
- Limitation: SAE availability and alignment to your exact model/layer may vary; feature identification may be dataset-specific.
- Design solution: Load a matching SAE, rank features by correlation with labels at the answer position, ablate only top-k features, and test generalization to paraphrase/negation controls.

[design_study]
- Independent variables: truth vs lie; paraphrase vs original; negation presence; context order permutation.
- Dependent measures: next-token logit difference for “ true” vs “ lie”; probe accuracy; drop from head/feature/direction ablations.
- Causal tests: direction projection; head ablation; SAE feature ablation; activation patching (true→lie and lie→true).
- Statistical plan: paired tests comparing baseline vs intervention deltas; per-layer and per-head effect size ranking; hold-out splits across control sets.

[explore_literature]
- Induction, IOI, and name-moving heads for token-to-token routing in transformers.
- Logit lens and tuned lens for layer-wise readouts; activation patching and causal tracing for necessity/sufficiency.
- SAE-based feature discovery for residual stream sparsity and interpretability.

[generate_controls]
- Syntax-matched but semantically neutral contexts; synonym-swapped instructions; randomized punctuation; reversed-facts that keep lexical tokens constant but flip ground truth.

Next steps
- Run Code 1 to identify peak layers and confirm causal direction effects via ablation/steering/patching.
- Run Code 2 to shortlist high-impact heads; inspect attention to fact tokens; optionally path patch those token flows.
- If available, run Code 3 to test sparse feature locality; replicate across models (e.g., GPT-2 small/medium) for generalization.

[!DISCLAIMER: ALL OUTPUTS ARE RESEARCH HYPOTHESES FOR VALIDATION, NOT EMPIRICAL CLAIMS!]
