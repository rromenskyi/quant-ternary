# Notes: dealignai's "CRACK" abliteration — the weight-level mechanism

Side note, unrelated to this project's quantization work — kept here because
it came up while comparing against `dealignai/Nemotron-3.5-Lightning-30B-A3B-JANG_2L-CRACK`
as a size/quality reference point. Focuses on *how the weights are physically
changed*, not the eval/harness wrapper around it.

**Short version:** CRACK is essentially advanced abliteration / directional
ablation. There may be no full fine-tune involved at all. They measure which
direction the residual stream moves in when the model decides to "refuse,"
then modify a few weight matrices so the model can no longer physically write
a component along that direction. The change is baked directly into the
`.safetensors` file.

## Their pipeline

1. Take the original BF16 model and a set of prompt pairs. For CRACK
   specifically, they use **structurally mirrored pairs**: pairs as
   structurally identical as possible, where one triggers a refusal and the
   other gets a normal answer. Earlier dealignai model cards explicitly
   mentioned **512 pairs**. This is so the activation difference reflects
   `refusal` specifically, not just topic difference.

2. Run both halves through the model and, at each layer of interest, capture
   the residual activation. For layer `l`:

   ```
   μ_bad,l  = mean(h_bad,l)
   μ_good,l = mean(h_good,l)
   r_l = normalize(μ_bad,l − μ_good,l)
   ```

   This is the **refusal direction** for that layer. Arditi et al.'s original
   work showed refusal in many instruction models is concentrated in a very
   low-dimensional, often nearly one-dimensional, direction of the residual
   stream.

3. Determine **at which layers `r_l` actually causally affects refusal** —
   this matters more than computing the vector itself. At inference time you
   can temporarily project it out:

   ```
   h'_l = h_l − (h_l^T r_l) r_l
   ```

   and check whether the model stops refusing. This narrows the target down
   to maybe 5-10 layers where the decision actually happens, instead of
   touching all of them.

   Their card for the Nemotron 3.5 Lightning version describes this as
   **"early decision-zone abliteration with per-layer refusal directions"**
   — i.e. they target the zone where the refusal decision forms, not a
   blanket orthogonalization of the whole 30B model.

4. Once the layers and directions are known, no runtime hook is needed —
   the operation is folded directly into the weights. For an output
   projection `o_proj` (`y = Wx`, writing attention's result back into the
   residual stream):

   ```
   W' = (I − s_l r_l r_l^T) W  =  W − s_l r_l (r_l^T W)
   ```

   `s_l` is a per-layer strength. After this, for any input `x`, the matrix
   can no longer produce a residual component along the refusal direction.
   This is why it's a permanent weight modification — no hooks needed after
   the model is saved. This is the mechanism from the original abliteration
   paper.

   (dealignai's git history reportedly once contained the line "calibrated
   per-layer strengths based on projection magnitude analysis," later
   scrubbed from the README.)

## Where exactly they touch the weights

An older README for their Qwen 3.5 VL version reportedly stated the surgery
directly: **512 mirrored pairs → per-layer projected vectors → only `o_proj`
on full-attention layers.** Later removed from the card, but visible in git
history.

```
Transformer block
      attention
         |
      Q - K - V -+
                  v
               o_proj    <- this is what gets modified
                  |
                  v
           residual stream
```

Why `o_proj` specifically: `q_proj`/`k_proj`/`v_proj` form attention's
internal computation, while `o_proj` is what writes the attention result
directly back into the residual stream. There's no need to destroy knowledge
inside attention — it's enough to prevent attention from writing out a
specific direction externally.
