"""Pattern classification of the fork prompt the main agent wrote for its subagent."""
import re

I = re.I
STANCE1 = {   # explicit stance words addressed to the subagent
    'critical': re.compile(r"\bcritical(ly)?\b", I),
    'skeptical': re.compile(r"\bs[kc]eptic\w*", I),
    'scrutinize': re.compile(r"\bscrutin\w*", I),
    'harsh_strict': re.compile(r"\b(harsh\w*|tough|stringent|ruthless|devil'?s advocate|do not be lenient|"
                               r"be strict|strict reviewer)\b", I),
}
STANCE2 = {   # directed fault-finding beyond the neutral "limitations"
    'flag_note_any': re.compile(r"\bflag\b|\bcall out\b|\bpoint out\b|\bidentify any\b|\bnote any\b|"
                                r"\bhighlight any\b|\blook (closely |carefully )?for\b", I),
    'overclaim_family': re.compile(r"overclaim\w*|over-claim\w*|overreach\w*|over-reach\w*|overstat\w*|"
                                   r"over-stat\w*|outrun\w*|outpace\w*|oversell\w*|oversold|exaggerat\w*|"
                                   r"unsupported|unsubstantiated|unjustified|overgeneral\w*", I),
    'inconsistency_error': re.compile(r"inconsisten\w*|contradict\w*|discrepanc\w*|\berrors?\b|\bmistakes?\b", I),
    'missing_lack': re.compile(r"\bmissing\b|\black(s|ing)?\b|\babsen(t|ce)\b|\bomitted\b|\bomissions?\b", I),
    'confound_unfair': re.compile(r"confound\w*|\bunfair\w*|cherry[- ]pick\w*", I),
    'flaws': re.compile(r"\bflaw\w*", I),
}
OTHER = {
    'rigorous': re.compile(r"\brigorous(ly)?\b", I),
    'actually_supports': re.compile(r"\b(actually|truly|genuinely|really)\b", I),
    'fair_stance': re.compile(r"\b(but|and|yet)\s+fair(ly)?\b(?!\s+(comparison|baseline|tun))|"
                              r"\bfair(ly)?\s+(and|but)\s+(rigorous|concrete|specific|critical|thorough)|"
                              r"\bbe fair\b|\bbalanced\b|\bobjective(ly)?\b|\bconstructive\b|"
                              r"\beven-handed\b|\bimpartial\w*", I),
    'strengths': re.compile(r"\bstrengths?\b", I),
    'limitations_weaknesses': re.compile(r"\bweakness\w*|\blimitations?\b", I),
}
RESTATE = {
    'reviewer_role': re.compile(r"\breviewer\b|\bpeer[- ]review", I),
    'iclr': re.compile(r"\bICLR\b"),
    'refers_to_criteria': re.compile(r"\bcriteria\b", I),
    'names_review_dimensions': None,   # >= 3 of the five merit words
    'rating_scale_values': re.compile(r"1\s*[/,|]\s*3\s*[/,|]\s*5\s*[/,|]\s*6\s*[/,|]\s*8\s*[/,|]\s*10", I),
    'rating_scale_labels': re.compile(r"strong reject|marginally (below|above)|strong accept", I),
    'score_ranges_1_4': re.compile(r"1\s*[-–]\s*4|1\s*/\s*2\s*/\s*3\s*/\s*4|\b1 to 4\b", I),
    'no_author_identification': re.compile(r"identify (the )?authors?|provenance|anonym", I),
    'ignore_embedded_instructions': re.compile(r"embedded|untrusted|instructions (inside|in) the paper", I),
    'paper_only': re.compile(r"only (on )?the (paper|submission|text)|paper text (only|provided)|"
                             r"solely on|no (outside|external)", I),
}
DIMS = ['novel', 'sound', 'rigor', 'clarity', 'significan']
SCOREWORD = re.compile(
    r"\brating\b|"
    r"\b(overall|iclr[- ]style|rough|suggested|your|accept(ance)?[ /-]+reject\w*)\s+(\w+\s+){0,2}recommendation\b|"
    r"\brecommendation\s*\((accept|reject)|\brecommendation on soundness|"
    r"\b(assign|give|provide|produce|output|include)\s+(\w+\s+){0,2}scores?\b|\bnumeric(al)? scores?\b|"
    r"(soundness|presentation|contribution|ICLR|suggested|recommended|tentative|preliminary|"
    r"provisional|proposed|numeric)[^.;]{0,80}\bscores?\b|"
    r"\bscores?\b[^.;]{0,40}(soundness|presentation|contribution|ICLR scale)|"
    r"\baccept \(true/false\)|\baccept/reject (leaning|recommendation)|"
    r"\bsuggested accept/reject\b", I)
JSONWORD = re.compile(r"\bJSON\b")
NEG = re.compile(r"\b(do not|don't|dont|no need|without|not|never|no)\b", I)
SPECIFIC = re.compile(r"\b(Table|Tab\.|Figure|Fig\.|Section|Sec\.|Theorem|Lemma|Eq\.|Equation|Appendix|"
                      r"Algorithm|Proposition)\s*[A-Z]?\.?\d|\"[^\"]{12,}\"|\be\.g\.", I)


def sentences(p: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.;:!?])\s+|\n+", p) if s.strip()]


def classify(p: str) -> dict:
    out = {'chars': len(p), 'words': len(p.split())}
    for grp in (STANCE1, STANCE2, OTHER, RESTATE):
        for k, rx in grp.items():
            if rx is not None:
                out[k] = bool(rx.search(p))
    low = p.lower()
    out['names_review_dimensions'] = sum(d in low for d in DIMS) >= 3
    ask_score = forbid_score = ask_json = forbid_json = False
    for s in sentences(p):
        if SCOREWORD.search(s):
            # negation that precedes the score word in the same sentence = "do not give scores"
            m = SCOREWORD.search(s)
            if NEG.search(s[:m.start()]) and not re.search(r"not attempt to identify", s[:m.start()], I):
                forbid_score = True
            else:
                ask_score = True
        if JSONWORD.search(s):
            m = JSONWORD.search(s)
            if NEG.search(s[:m.start()]):
                forbid_json = True
            elif re.search(r"\b(output|return|respond|reply|produce|write|provide|give)\b"
                           r"[^.(]{0,40}\bJSON\b", s, I) and not re.search(r"that I can use", s, I):
                ask_json = True
    out.update(asks_scores=ask_score, forbids_scores=forbid_score, asks_json=ask_json,
               forbids_json=forbid_json)
    out['stance1_any'] = any(out[k] for k in STANCE1)
    out['stance2_any'] = any(out[k] for k in STANCE2)
    out['stance_any'] = out['stance1_any'] or out['stance2_any']
    out['paper_specific_refs'] = bool(SPECIFIC.search(p))
    return out
