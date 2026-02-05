from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
import torch

from score_sequences import score_variants_ensemble


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
    start_tokens: Optional[List[int]] = None,
    weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    comp_tok = model.tokenizer.encode_completions(
        seqs,
        bos_token=model.tokenizer.sep_token,
        eos_token=model.tokenizer.sep_token,
    )
    completion_ids = (
        torch.as_tensor(comp_tok["input_ids"], dtype=torch.long)
        .unsqueeze(0)
        .to(model.device)
    )

    with torch.no_grad():
        _, mean_lls_per_pos, _ = score_variants_ensemble(
            model=model,
            completion_ids=completion_ids,
            scoring_max_tokens=scoring_max_tokens,
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
                start_tokens=start_tokens,
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
