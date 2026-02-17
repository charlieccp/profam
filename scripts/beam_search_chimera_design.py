from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from scripts.score_sequences import score_variants_ensemble


@dataclass
class Beam:
    seq: str
    score: float
    score_pos: float
    n_switched: bool
    c_switched: bool
    filled_str: str

    # per-loop accumulators used for ranking (None refs are skipped via counts)
    dLL_parentA_loop_sum: float = 0.0
    dLL_parentB_loop_sum: float = 0.0
    n_parentA_loop: int = 0
    n_parentB_loop: int = 0

    # optional per-position deltas (debug/trace)
    dLL_parentA_pos: float = 0.0
    dLL_parentB_pos: float = 0.0


def _allowed_parent_choices(
    pos: int, beam: Beam, nterm_positions: List[int], cterm_positions: List[int]
) -> Tuple[bool, bool]:
    allow_A = True
    allow_B = True
    if pos in nterm_positions and beam.n_switched:
        allow_A = False
    if pos in cterm_positions and beam.c_switched:
        allow_B = False
    return allow_A, allow_B


def _update_switch_flags(
    pos: int, picked_parent: str, beam: Beam, nterm_positions: List[int], cterm_positions: List[int]
) -> Tuple[bool, bool]:
    n_sw = beam.n_switched
    c_sw = beam.c_switched
    if pos in nterm_positions and (not n_sw) and picked_parent == "B":
        n_sw = True
    if pos in cterm_positions and (not c_sw) and picked_parent == "A":
        c_sw = True
    return n_sw, c_sw


def _normalize_switch_flags(
    pos: int, n_sw: bool, c_sw: bool, nterm_positions: List[int], cterm_positions: List[int]
) -> Tuple[bool, bool]:
    # Only enforce switch constraints while we are inside the constrained region
    if pos not in nterm_positions:
        n_sw = False
    if pos not in cterm_positions:
        c_sw = False
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
    )  # (1, n, L)

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

    # last residue before EOS/SEP (your original behavior)
    return mean_lls_per_pos[:, -2].astype(np.float64)


def loop_score_A(b: Beam) -> float:
    # average over positions where parentA exists in this loop
    return b.dLL_parentA_loop_sum / max(1, b.n_parentA_loop)


def loop_score_B(b: Beam) -> float:
    return b.dLL_parentB_loop_sum / max(1, b.n_parentB_loop)


def _dense_ranks(values: List[float], higher_is_better: bool = False) -> Dict[float, int]:
    uniq = sorted(set(values), reverse=higher_is_better)
    return {v: i + 1 for i, v in enumerate(uniq)}


def select_by_sum_of_ranks_with_ties(
    beams: List[Beam],
    beam_width: int,
    *,
    higher_is_better: bool = False,
) -> List[Beam]:
    """
    Dense ranks with ties for A and B, then minimize sum(rankA, rankB).
    Keep top beam_width and include all ties at the cutoff sum-rank.
    """
    if not beams:
        return []

    valsA = [loop_score_A(b) for b in beams]
    valsB = [loop_score_B(b) for b in beams]

    print(valsA)
    print(valsB)

    rA = _dense_ranks(valsA, higher_is_better=higher_is_better)
    rB = _dense_ranks(valsB, higher_is_better=higher_is_better)

    print(rA)
    print(rB)

    scored = []
    for b, a, c in zip(beams, valsA, valsB):
        ra = rA[a]
        rb = rB[c]
        scored.append((ra + rb, ra, rb, b))

    scored.sort(key=lambda x: (x[0], x[1], x[2]))
    print(scored)

    if len(scored) <= beam_width:
        return [b for *_, b in scored]

    cutoff_sum = scored[beam_width - 1][0]
    print(cutoff_sum)
    return [b for (s, _, _, b) in scored if s <= cutoff_sum]


def save_step_jsonl(f, pos, expansions, parentA_scores, parentB_scores):
    exp_rows = []

    for idx, (prev, new_seq, picked_parent, emitted, filled_str) in enumerate(expansions):
        exp_rows.append(
            {
                "new_seq": new_seq,
                "picked_parent": picked_parent,
                "emitted": emitted,
                "filled_str": filled_str,
                "dA_pos": None if parentA_scores[idx] is None else float(parentA_scores[idx]),
                "dB_pos": None if parentB_scores[idx] is None else float(parentB_scores[idx]),
            }
        )

    rec = {
        "pos": int(pos),
        "expansions": exp_rows,
    }

    f.write(json.dumps(rec) + "\n")
    f.flush()


def beam_search_chimera_template(
    *,
    model,
    name: str,
    chimera_template: str,
    hole_options: Dict[int, Tuple[Union[str, None], Union[str, None]]],
    aligned_ll_parentA: List[Optional[float]],
    aligned_ll_parentB: List[Optional[float]],
    beam_width: int = 16,
    nterm_positions: List[int],
    cterm_positions: List[int],
    tokenized_conditioning_sequences_parentA: Optional[List[List[int]]] = None,
    tokenized_conditioning_sequences_parentB: Optional[List[List[int]]] = None,
    ensemble_size: Optional[int] = None,
    scoring_max_tokens: int = 64000,
    max_tokens_override: int = 8192,
    start_tokens: Optional[List[int]] = None,
    weights_parentA: Optional[np.ndarray] = None,
    weights_parentB: Optional[np.ndarray] = None,
) -> List[Beam]:
    """
    Beam search over a chimera template with '_' holes.
    At the end of each *loop* (a consecutive run of holes), prune using:
      - dense rank under reference A loop score
      - dense rank under reference B loop score
      - minimize sum of ranks
      - keep top beam_width and include ties at cutoff
    Loop scores are computed only over positions where the reference has a value (None is skipped).

    Returns remaining beams (you can post-process into sequences/summaries).
    """
    L = len(chimera_template)

    # loop ends: end of each consecutive run of hole positions
    holes = sorted(hole_options.keys())
    loop_end_holes = set()
    for i, h in enumerate(holes):
        if i == len(holes) - 1 or holes[i + 1] != h + 1:
            loop_end_holes.add(h)

    # validate hole options
    for pos, (a, b) in hole_options.items():
        if not (0 <= pos < L):
            raise ValueError(f"hole_options position {pos} out of range (len={L})")
        if chimera_template[pos] != "_":
            raise ValueError(
                f"hole_options has pos {pos} but template at pos is '{chimera_template[pos]}', not '_'"
            )
        for opt in (a, b):
            if opt is None:
                continue
            if len(opt) != 1:
                raise ValueError(f"Options at pos {pos} must be single residues or empty, got {opt!r}")

    if len(aligned_ll_parentA) != L or len(aligned_ll_parentB) != L:
        raise ValueError("aligned_ll_parentA/B must have length equal to len(chimera_template)")

    beams: List[Beam] = [
        Beam(seq="", score=0.0, score_pos=0.0, n_switched=False, c_switched=False, filled_str="")
    ]

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"beam_trace_{name}_{beam_width}_{ts}.jsonl", "w") as trace_f:
        for pos in range(L):
            template_char = chimera_template[pos]

            # expansions entries: (prev_beam, new_seq, picked_parent, emitted_residue_or_none, filled_str)
            expansions: List[Tuple[Beam, str, Optional[str], Optional[str], str]] = []

            if template_char != "_":
                # fixed residue: you currently do NOT score it (keeps ll/deltas unchanged)
                for b in beams:
                    expansions.append((b, b.seq + template_char, None, None, b.filled_str))
            else:
                # hole: choose A or B (residue or deletion)
                optA, optB = hole_options[pos]
                for b in beams:
                    allow_A, allow_B = _allowed_parent_choices(pos, b, nterm_positions, cterm_positions)

                    if allow_A:
                        if optA is None:
                            expansions.append((b, b.seq, "A", None, b.filled_str + "A"))  # deletion
                        else:
                            expansions.append((b, b.seq + optA, "A", optA, b.filled_str + "A"))

                    if allow_B:
                        if optB is None:
                            expansions.append((b, b.seq, "B", None, b.filled_str + "B"))
                        else:
                            expansions.append((b, b.seq + optB, "B", optB, b.filled_str + "B"))

            if not expansions:
                raise RuntimeError(f"No valid expansions at position {pos} under the prior.")

            # Score only expansions that emitted a residue
            emit_indices = [i for i, (_, _, _, emitted, _) in enumerate(expansions) if emitted is not None]
            emit_seqs = [expansions[i][1] for i in emit_indices]

            emit_ll_parentA, emit_ll_parentB = None, None
            if emit_seqs:
                emit_ll_parentA = _score_last_residue_lls(
                    model,
                    emit_seqs,
                    tokenized_conditioning_sequences=tokenized_conditioning_sequences_parentA,
                    ensemble_size=ensemble_size,
                    scoring_max_tokens=scoring_max_tokens,
                    max_tokens_override=max_tokens_override,
                    start_tokens=start_tokens,
                    weights=weights_parentA,
                )

                emit_ll_parentB = _score_last_residue_lls(
                    model,
                    emit_seqs,
                    tokenized_conditioning_sequences=tokenized_conditioning_sequences_parentB,
                    ensemble_size=ensemble_size,
                    scoring_max_tokens=scoring_max_tokens,
                    max_tokens_override=max_tokens_override,
                    start_tokens=start_tokens,
                    weights=weights_parentB,
                )

            ll_by_exp_parentA = np.full(len(expansions), np.nan, dtype=float)
            if emit_ll_parentA is not None:
                for j, exp_i in enumerate(emit_indices):
                    ll_by_exp_parentA[exp_i] = float(emit_ll_parentA[j])

            ll_by_exp_parentB= np.full(len(expansions), np.nan, dtype=float)
            if emit_ll_parentB is not None:
                for j, exp_i in enumerate(emit_indices):
                    ll_by_exp_parentB[exp_i] = float(emit_ll_parentB[j])

            # Build new beams from expansions
            new_beams: List[Beam] = []
            parentA_ref = aligned_ll_parentA[pos]  # can be None
            parentB_ref = aligned_ll_parentB[pos]  # can be None
            parentA_scores = []
            parentB_scores = []
            for i, (prev, new_seq, picked_parent, emitted, filled_str) in enumerate(expansions):
                if picked_parent is None:
                    n_sw, c_sw = prev.n_switched, prev.c_switched
                else:
                    n_sw, c_sw = _update_switch_flags(pos, picked_parent, prev, nterm_positions, cterm_positions)

                n_sw, c_sw = _normalize_switch_flags(pos, n_sw, c_sw, nterm_positions, cterm_positions)

                if emitted is None:
                    ll = 0.0
                    dA = None
                    dB = None
                    addA = 0
                    addB = 0
                else:
                    ll_parentA = float(ll_by_exp_parentA[i])
                    ll_parentB = float(ll_by_exp_parentB[i])

                    if parentA_ref is None:
                        dA = None
                        addA = 0
                    else:
                        dA = ll_parentA - float(parentA_ref)
                        addA = 1

                    if parentB_ref is None:
                        dB = None
                        addB = 0
                    else:
                        dB = ll_parentB - float(parentB_ref)
                        addB = 1

                # ALWAYS append one entry per expansion
                parentA_scores.append(dA)
                parentB_scores.append(dB)

                new_beams.append(
                    Beam(
                        seq=new_seq,
                        score=prev.score + ll,
                        score_pos=ll,
                        n_switched=n_sw,
                        c_switched=c_sw,
                        filled_str=filled_str,
                        dLL_parentA_loop_sum=prev.dLL_parentA_loop_sum + (dA or 0.0),
                        dLL_parentB_loop_sum=prev.dLL_parentB_loop_sum + (dB or 0.0),
                        n_parentA_loop=prev.n_parentA_loop + addA,
                        n_parentB_loop=prev.n_parentB_loop + addB,
                        dLL_parentA_pos=dA or 0.0,
                        dLL_parentB_pos=dB or 0.0,
                    )
                )
                            
            save_step_jsonl(trace_f, pos, expansions, parentA_scores, parentB_scores )
            beams = new_beams

            # Prune only at end of a loop (end of a consecutive hole run)
            if pos in loop_end_holes:
                beams = select_by_sum_of_ranks_with_ties(beams, beam_width, higher_is_better=False)

                # reset per-loop accumulators for next loop
                for b in beams:
                    b.dLL_parentA_loop_sum = 0.0
                    b.dLL_parentB_loop_sum = 0.0
                    b.n_parentA_loop = 0
                    b.n_parentB_loop = 0

    return beams


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

    # seed_all(args.seed)

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

    
    # # Encode completions with BOS/EOS = [SEP]
    # comp_tok = model.tokenizer.encode_completions(
    #     cand_seqs,
    #     bos_token=model.tokenizer.sep_token,
    #     eos_token=model.tokenizer.sep_token,
    # )
    # completion_ids = (
    #     torch.as_tensor(comp_tok["input_ids"], dtype=torch.long)
    #     .unsqueeze(0)
    #     .to(model.device)
    # )  # (1, n, L)

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
