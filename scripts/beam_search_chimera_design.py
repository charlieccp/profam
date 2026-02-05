from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Set, Tuple, Union
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from dataclasses import dataclass
from score_sequences import score_variants_ensemble

"""
Script to generate 'best' chimeric sequences using a beam search.
The best chimeras is those with the highest conditional likelihood scores.
"""

@dataclass
class Beam:
    seq: str
    score: float
    n_switched: bool
    c_switched: bool
    emitted_len: int  # number of residues actually emitted/scored


def _allowed_parent_choices(pos: int, beam: Beam,
                            nterm_positions: Set[int], cterm_positions: Set[int]) -> Tuple[bool, bool]:
    allow_A = True
    allow_B = True
    if pos in nterm_positions and beam.n_switched:
        allow_A = False
    if pos in cterm_positions and beam.c_switched:
        allow_B = False
    return allow_A, allow_B


def _update_switch_flags(pos: int, picked_parent: str, beam: Beam,
                         nterm_positions: Set[int], cterm_positions: Set[int]) -> Tuple[bool, bool]:
    n_sw = beam.n_switched
    c_sw = beam.c_switched
    if pos in nterm_positions and (not n_sw) and picked_parent == "B":
        n_sw = True
    if pos in cterm_positions and (not c_sw) and picked_parent == "A":
        c_sw = True
    return n_sw, c_sw


def _score_last_residue_lls(
    model,
    seqs: List[str],
    *,
    tokenized_conditioning_sequences: Optional[List[List[int]]] = None,
    ensemble_size: Optional[int] = None,
    scoring_max_tokens: int = 64000,
    max_tokens_override: int = 8192,
    start_tokens: Optional[List[int]] = None,
    weights: Optional[np.ndarray] = None,
) -> np.ndarray:

    #Need to adapt format?
    # cand_names, cand_seqs = read_fasta(
    #     args.candidates_file, keep_insertions=False, to_upper=True
    # )

    # Encode completions with BOS/EOS = [SEP]
    comp_tok = model.tokenizer.encode_completions(
        seqs,
        bos_token=model.tokenizer.sep_token,
        eos_token=model.tokenizer.sep_token,
    )
    completion_ids = (
        torch.as_tensor(comp_tok["input_ids"], dtype=torch.long)
        .unsqueeze(0)
        .to(model.device)
    ) # (1, n, L)

    with torch.no_grad():
        _, mean_lls_per_pos, _ = score_variants_ensemble(
            model=model,
            completion_ids=completion_ids,
            scoring_max_tokens=scoring_max_tokens,
            max_tokens_override=max_tokens_override,
            tokenized_conditioning_sequences=tokenized_conditioning_sequences,
            ensemble_size=ensemble_size,
            start_tokens=start_tokens,
            weights=weights,
        )

    return mean_lls_per_pos[:, -2].astype(np.float64)


def beam_search_chimera_template(
    *,
    model,
    chimera_template: str,
    hole_options: Dict[int, Tuple[Union[str, None], Union[str, None]]],
    beam_width: int = 16,
    nterm_positions: Optional[Set[int]] = None,
    cterm_positions: Optional[Set[int]] = None,
    tokenized_conditioning_sequences: Optional[List[List[int]]] = None,
    ensemble_size: Optional[int] = None,
    scoring_max_tokens: int = 64000,
    max_tokens_override: int = 8192,
    start_tokens: Optional[List[int]] = None,
    weights: Optional[np.ndarray] = None,
) -> List[Tuple[str, float, float, int]]:
    """
    Supports empty options at '_' holes:
      option == '' or None => emit nothing (sequence shortens), no model score added.

    Returns a list of:
      (sequence, total_score, avg_score_per_emitted_residue, emitted_len)
    sorted best-first by `rank_by`.
    """
    nterm_positions = nterm_positions or set()
    cterm_positions = cterm_positions or set()

    L = len(chimera_template)

    for pos, (a, b) in hole_options.items():
        if not (0 <= pos < L):
            raise ValueError(f"hole_options position {pos} out of range (len={L})")
        if chimera_template[pos] != "_":
            raise ValueError(f"hole_options has pos {pos} but template at pos is '{chimera_template[pos]}', not '_'")

        # allow None for gaps; otherwise must be single residue
        for opt in (a, b):
            if opt is None:
                continue
            if len(opt) != 1:
                raise ValueError(f"Options at pos {pos} must be single residues or empty, got {opt!r}")

    beams: List[Beam] = [Beam(seq="", score=0.0, n_switched=False, c_switched=False, emitted_len=0)]

    for pos in range(L):
        template_char = chimera_template[pos]

        # expansions entries:
        # (prev_beam, new_seq, picked_parent, emitted_residue_or_none)
        expansions: List[Tuple[Beam, str, Optional[str], Optional[str]]] = []

        if template_char != "_":
            # Fixed residue: always emit and score
            for b in beams:
                expansions.append((b, b.seq + template_char, None, template_char))
        else:
            # Hole: choose option A or B, either can be residue or empty
            optA, optB = hole_options[pos]
            for b in beams:
                allow_A, allow_B = _allowed_parent_choices(pos, b, nterm_positions, cterm_positions)

                if allow_A:
                    if optA is None:
                        expansions.append((b, b.seq, "A", None))       # deletion
                    else:
                        expansions.append((b, b.seq + optA, "A", optA))  # emit residue

                if allow_B:
                    if optB is None:
                        expansions.append((b, b.seq, "B", None))
                    else:
                        expansions.append((b, b.seq + optB, "B", optB))

        if not expansions:
            raise RuntimeError(f"No valid expansions at position {pos} under the prior.")

        # Score only the expansions that emitted a residue
        emit_indices = [i for i, (_, _, _, emitted) in enumerate(expansions) if emitted is not None]
        emit_seqs = [expansions[i][1] for i in emit_indices]

        emit_ll = None
        if emit_seqs:
            emit_ll = _score_last_residue_lls(
                model,
                emit_seqs,
                tokenized_conditioning_sequences=tokenized_conditioning_sequences,
                ensemble_size=ensemble_size,
                scoring_max_tokens=scoring_max_tokens,
                max_tokens_override=max_tokens_override,
                start_tokens=[47, 63],
                weights=weights,
            )

        # Build new beams
        new_beams: List[Beam] = []
        emit_cursor = 0
        for i, (prev, new_seq, picked_parent, emitted) in enumerate(expansions):
            # update switch flags only when there was a choice (picked_parent != None)
            if picked_parent is None:
                n_sw, c_sw = prev.n_switched, prev.c_switched
            else:
                n_sw, c_sw = _update_switch_flags(pos, picked_parent, prev, nterm_positions, cterm_positions)

            if emitted is None:
                # deletion: no score added, emitted_len unchanged
                new_beams.append(
                    Beam(seq=new_seq, score=prev.score, n_switched=n_sw, c_switched=c_sw, emitted_len=prev.emitted_len)
                )
            else:
                ll = float(emit_ll[emit_cursor])
                emit_cursor += 1
                new_beams.append(
                    Beam(
                        seq=new_seq,
                        score=prev.score + ll,
                        n_switched=n_sw,
                        c_switched=c_sw,
                        emitted_len=prev.emitted_len + 1,
                    )
                )

        # Prune: rank beams during search
        new_beams.sort(key=lambda x: (x.score / max(1, x.emitted_len)), reverse=True)
        beams = new_beams[:beam_width]

    # Final ranking
    out = []
    for b in beams:
        avg = b.score / max(1, b.emitted_len)
        out.append((b.seq, b.score, avg, b.emitted_len))


    out.sort(key=lambda x: x[2], reverse=True)  # average per emitted residue

    return out


def main():
    parser = argparse.ArgumentParser(
        description="Compute list of chimeric candidates ranked based on their conditional likelihoods"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="model_checkpoints/profam-1",
        help="Checkpoint run directory (contains checkpoints/last.ckpt)",
    )
    parser.add_argument(
        "--conditioning_fasta",
        type=str,
        default=None,
        help="Path to conditioning FASTA/MSA file",
    )
    parser.add_argument(
        "--template_chimera",
        type=str,
        default=None,
        help="(Unaligned) sequence of chimera with '_' to indicate positions to define.",
    )
    parser.add_argument(
        "--cutting_pt_options",
        type=dict,
        default=None,
        help="Dictionary with as keys, the positions (0-based) to determine and as values,\
        tuple with the 2 residues to pick between (use None if residue is gap in alignment\
        parents.",
    )
    parser.add_argument(
        "--NtermLoop_positions",
        type=dict,
        default=None,
        help="Positions (0-based) that are part of the Nterm of the loop (this is important\
        to respect the biological prior)",
    )
    parser.add_argument(
        "--CtermLoop_positions",
        type=dict,
        default=None,
        help="Positions (0-based) that are part of the Cterm of the loop (this is important\
        to respect the biological prior)",
    )
    parser.add_argument(
        "--beam_width",
        type=int,
        default=16,
        help="Number of constructs to keep at every step.",
    )
    parser.add_argument(
        "--save_dir", type=str, default="outputs", help="Directory to save output files"
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=8192,
        help="Token budget (prompt+completion) used for batch size heuristics",
    )
    parser.add_argument(
        "--scoring_max_tokens",
        type=int,
        default=64000,
        help=(
            "Token budget used ONLY to dynamically set the scoring batch size to stay within memory "
            "constraints. This is typically higher than --max_tokens. "
        ),
    )
    parser.add_argument(
        "--ensemble_number",
        type=int,
        default=3,
        help="Number of prompts used to generate the ensemble score",
    )
    parser.add_argument(
        "--use_diversity_weights",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If set, sample conditioning sequences with homology-based diversity weights (1/#neighbors).",
    )
    parser.add_argument(
        "--diversity_theta",
        type=float,
        default=0.2,
        help="Theta used for homology neighbor definition when computing diversity weights.",
    )
    parser.add_argument(
        "--recompute_diversity_weights",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If set, ignore any on-disk cached weights and recompute.",
    )
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="sdpa",
        choices=["sdpa", "flash_attention_2", "eager"],
        help="Override attention implementation before model init (e.g. flash_attention_2)",
    )
    args = parser.parse_args()

    seed_all(args.seed)

    ckpt_path = os.path.join(args.checkpoint_dir, "checkpoints/last.ckpt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found at {ckpt_path}. Run `python scripts/hf_download_checkpoint.py` to download the checkpoint."
        )
    attn_impl = args.attn_implementation

    try:
        import flash_attn
    except ImportError:
        if attn_impl == "flash_attention_2":
            raise ImportError(
                "Flash attention is not installed. "
                "select an alternative attention implementation such as:\n`--attn_implementation sdpa`.\n"
                "Or install it with:\n`pip install flash-attn --no-build-isolation`. "
            )

    try:
        ckpt_blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        hyper_params = ckpt_blob.get("hyper_parameters", {})
        cfg_obj = hyper_params.get("config", None)
        if cfg_obj is None:
            raise RuntimeError(
                "Could not find 'config' in checkpoint hyper_parameters to override attn implementation"
            )
        setattr(cfg_obj, "attn_implementation", attn_impl)
        setattr(cfg_obj, "_attn_implementation", attn_impl)
        # We handle ensemble size explicitly now, but setting it here doesn't hurt
        if hasattr(cfg_obj, "gym_subsamples_per_n"):
            setattr(cfg_obj, "gym_subsamples_per_n", args.ensemble_number)
        model: LlamaLitModule = LlamaLitModule.load_from_checkpoint(
            ckpt_path, config=cfg_obj, strict=False
        )
    except Exception as e:
        raise RuntimeError(f"Failed to override attention implementation: {e}")
    model.eval()
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    model.to(args.device, dtype=dtype_map[args.dtype])

    weights = None
    if args.use_diversity_weights and args.conditioning_fasta:
        print(
            f"Computing diversity (homology) weights for {args.conditioning_fasta}...",
            file=sys.stderr,
        )
        _, aligned_sequences = read_fasta(
            args.conditioning_fasta,
            keep_insertions=False,
            to_upper=True,
            keep_gaps=True,
        )
        weights = compute_homology_sequence_weights_with_cache(
            msa_file=args.conditioning_fasta,
            sequences=aligned_sequences,
            theta=args.diversity_theta,
            force_recalc=args.recompute_diversity_weights,
        )

    # Build ProteinDocument objects (just to read sequences nicely)
    len_cond_doc = None
    if args.conditioning_fasta:
        cond_doc = build_pool_from_fasta(args.conditioning_fasta)
        # Tokenize conditioning sequences individually
        print(
            f"Tokenizing {len(cond_doc.sequences)} conditioning sequences...",
            file=sys.stderr,
        )
        # Using the tokenizer directly on strings to get IDs.
        # NOTE: verify if we need spaces or not. The tokenizer in debug worked on "ACDEFGH".
        tokenized_conditioning_sequences = [
            model.tokenizer(
                seq.upper().replace("-", "").replace(".", ""), add_special_tokens=False
            )["input_ids"]
            for seq in cond_doc.sequences
        ]
        len_cond_doc = len(cond_doc.sequences)

    
    # Encode completions with BOS/EOS = [SEP]
    comp_tok = model.tokenizer.encode_completions(
        cand_seqs,
        bos_token=model.tokenizer.sep_token,
        eos_token=model.tokenizer.sep_token,
    )
    completion_ids = (
        torch.as_tensor(comp_tok["input_ids"], dtype=torch.long)
        .unsqueeze(0)
        .to(model.device)
    )  # (1, n, L)

    with torch.no_grad():
        beam_search_chimera_template(
                model = model,
                chimera_template = args.template_chimera,
                hole_options = args.cutting_pt_options,
                beam_width = args.beam_width,
                nterm_positions = args.NtermLoop_positions,
                cterm_positions = args.CtermLoop_positions,
                tokenized_conditioning_sequences = tokenized_conditioning_sequences,
                ensemble_size = args.ensemble_number,
                scoring_max_tokens = args.scoring_max_tokens,
                start_tokens=[47, 63],
                weights = weights,
            )
    print(beam_search_chimera_template)
#     # Output handling
#     os.makedirs(args.save_dir, exist_ok=True)
#     candidate_basename = os.path.splitext(os.path.basename(args.candidates_file))[0]

#     csv_path = os.path.join(args.save_dir, f"{candidate_basename}_scores.csv")
#     json_path = os.path.join(args.save_dir, f"{candidate_basename}_metadata.json")

#     df_out = pd.DataFrame(
#         {"id": cand_names, "mutated_sequence": cand_seqs, "score": lls.tolist()}
#     )
#     if dms_scores is not None:
#         df_out["DMS_score"] = dms_scores
#     df_out.to_csv(csv_path, index=False)

#     # Save per-position mean log-likelihood per mutant.
#     # lls_per_pos has shape (n_mutants, L_seq) where L_seq == len(mutated_sequence).
#     for ll_per_pos, cand in zip(lls_per_pos,cand_names):

#         df = pd.DataFrame({
#             "position": [i+1 for i in range(len(ll_per_pos))],
#             "mean_log_likelihood": ll_per_pos,
#         })

#         os.makedirs(os.path.join(args.save_dir, "ll_per_pos_per_cand"), exist_ok=True)
#         per_pos_csv_path = os.path.join(args.save_dir, "ll_per_pos_per_cand", f"{cand}_per_pos_ll.csv")
#         df.to_csv(per_pos_csv_path, index=False)

#     print(f"Scores saved to {csv_path}...")

#     # Calculate metrics
#     corr = None
#     if dms_scores is not None:
#         corr, _ = spearmanr(lls, dms_scores)
#         print(f"Spearman correlation: {corr}", file=sys.stderr)

#     print(n_opt_list)
#     metadata = {
#         "n_sequences_evaluated": len(cand_seqs),
#         "ensemble_number": args.ensemble_number,
#         "timestamp": datetime.now().isoformat(),
#         "conditioning_fasta": args.conditioning_fasta,
#         # "n_conditioning_sequences": int(len_cond_doc) if len_cond_doc is not None else None,
#         "n_conditioning_sequences":",".join(n_opt_list),
#         "candidates_file": args.candidates_file,
#         "mean_likelihood_score": float(np.mean(lls)),
#         "spearman_correlation": float(corr) if corr is not None else None,
#         "checkpoint": args.checkpoint_dir,
#     }

#     with open(json_path, "w") as f:
#         json.dump(metadata, f, indent=4)

#     print(f"Metadata saved to {json_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
