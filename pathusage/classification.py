"""Sequence-level activity classification.

Assigns one activity class to a movement sequence from peak per-frame
detection counts (the maximum over all frames). A person-priority rule
prevents human passages from leaking into the wildlife class when a
spurious animal box co-occurs with a person.
"""

from __future__ import annotations


def classify_sequence(seq_dets, count_conf=0.40):
    """Return per-class counts and the primary activity for a sequence.

    Parameters
    ----------
    seq_dets : list[list[dict]]
        Detections per frame (see ``detection.Detector.detect``).
    count_conf : float
        Confidence threshold for counting persons / vehicles.

    Rule
    ----
        person + vehicle -> mtb
        person alone     -> on_foot
        lone vehicle     -> on_foot   (spurious box on an auto-free trail)
        animal           -> dog / horse / wildlife
        nothing          -> empty
    """
    peak_p = peak_v = 0
    animal_peak = {}
    n_empty = 0
    for frame in seq_dets:
        peak_p = max(peak_p, sum(1 for d in frame
                                 if d["category"] == 2 and d["conf"] >= count_conf))
        peak_v = max(peak_v, sum(1 for d in frame
                                 if d["category"] == 3 and d["conf"] >= count_conf))
        for d in frame:
            if d["category"] == 1:
                lbl = d.get("animal", "wildlife")
                animal_peak[lbl] = max(animal_peak.get(lbl, 0), 1)
        if not frame:
            n_empty += 1

    n_dogs = animal_peak.get("dog", 0)
    n_horse = animal_peak.get("horse", 0)
    n_wild = sum(c for l, c in animal_peak.items() if l not in ("dog", "horse"))

    if peak_p > 0 and peak_v > 0:
        primary = "mtb"
    elif peak_p > 0:
        primary = "on_foot"
    elif peak_v > 0:
        primary = "on_foot"
    elif n_dogs > 0:
        primary = "dog"
    elif n_horse > 0:
        primary = "horse"
    elif n_wild > 0:
        primary = "wildlife"
    else:
        primary = "empty"

    return {"n_total_persons": peak_p,
            "n_mtb": peak_p if (peak_p > 0 and peak_v > 0) else 0,
            "n_on_foot": peak_p if (peak_p > 0 and peak_v == 0) else 0,
            "n_dogs": n_dogs, "n_horses": n_horse, "n_other_wildlife": n_wild,
            "n_empty_frames": n_empty, "primary_activity": primary}
