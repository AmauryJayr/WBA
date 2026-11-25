# Why it works: proofs behind the presentation

This file collects the arguments that make every claim in the deck valid. Each
result is labelled:

- **[cited]**: a published result we use without reproving.
- **[paper]**: proved in Garg & Bridgman or Ben-Or, Dolev & Hoch; we restate the proof in our notation.
- **[ours]**: our result, proved here. These have not been peer reviewed.
- **[approx]**: an approximation, with the assumption that makes it one.

## 0. Model and notation

- N processes, fully connected, synchronous lock-step rounds, reliable channels;
  a receiver knows who sent each message.
- Byzantine faults: a faulty process may send anything, or nothing. The faulty
  processes may collude and may see every honest message of a round before
  choosing their own (a *rushing* adversary).
- Weights: w[j] ≥ 0 and Σ_j w[j] = 1. For a set S, W(S) = Σ_{j∈S} w[j].
- ρ is the assumed bound: W(faulty) ≤ ρ. Let H be the honest set, so W(H) ≥ 1 − ρ.
- Thresholds are written in ρ. With w[j] = 1/N and ρ = t/N they become the
  unweighted ones: "weight ≥ 1 − ρ" ↔ "≥ N − t processes", "weight > ρ" ↔ "≥ t + 1 processes".
- Messages from processes with weight 0 are ignored by every algorithm here, so
  such processes only listen.
- t is the number of faults an unweighted algorithm is built for; f (or F for
  positive-weight processes) is the number that actually fail in a run.

The three goals: **agreement** (honest processes decide the same value),
**validity** (if every honest process starts with v, they decide v),
**termination** (every honest process decides).

## 1. Background results we rely on

**Theorem 1.1 [cited]** (Pease, Shostak & Lamport 1980). Without signatures,
Byzantine agreement is solvable only if N > 3t.

**Theorem 1.2 [cited]** (Fischer & Lynch 1982). Any Byzantine agreement protocol
needs t + 1 rounds in the worst case.

**Theorem 1.3 [cited]** (Fischer, Lynch & Paterson 1985). In an asynchronous
system no deterministic protocol solves consensus with even one crash. This is
why all algorithms here assume synchrony.

**Lemma 1.4 [paper]** (Garg & Bridgman Lemma 1). If ρ ≥ 1/3, some weight
assignment admits no weighted BA protocol.
*Sketch.* Take three processes of weight 1/3 each. Any one may be faulty
(weight 1/3 ≤ ρ), which is the N = 3, t = 1 case, impossible by Theorem 1.1.

## 2. The two weight facts everything uses

**Lemma 2.1 (heavy sets contain an honest process).** If W(S) > ρ then S
contains an honest process.
*Proof.* If S were all faulty, W(S) ≤ W(faulty) ≤ ρ. ∎

**Lemma 2.2 (quorums overlap in an honest process).** If ρ < 1/3 and W(S₁) ≥ 1 − ρ,
W(S₂) ≥ 1 − ρ, then W(S₁ ∩ S₂ ∩ H) > 0, so S₁ and S₂ share an honest process.
*Proof.* W(S₁ ∩ S₂) ≥ W(S₁) + W(S₂) − 1 ≥ 1 − 2ρ. Removing at most ρ of faulty
weight leaves at least 1 − 3ρ > 0 of honest weight in the intersection. ∎

This is where 1/3 comes from: the overlap argument needs 1 − 3ρ > 0.

**Corollary 2.3.** If ρ < 1/3 and some set S of *honest* processes has
W(S) ≥ 1 − 2ρ, then W(S) > ρ, so every honest process that hears from all of S
sees more than ρ of weight from it. (1 − 2ρ > ρ ⇔ ρ < 1/3.)

## 3. The anchor α_ρ

Order processes heaviest first, w[1] ≥ w[2] ≥ …, and let
α_ρ = min{k : w[1] + … + w[k] > ρ}.

**Lemma 3.1 [paper].** Among the α_ρ heaviest processes at least one is honest.
*Proof.* Their weight exceeds ρ; apply Lemma 2.1. ∎

**Lemma 3.2 [paper]** (Garg & Bridgman Lemma 5). With ρ = t/N, α_ρ ≤ t + 1 for
every weight assignment.
*Proof.* The t + 1 heaviest weights average at least the overall average 1/N,
so they total at least (t + 1)/N > t/N = ρ. ∎

**Lemma 3.3 [cited]** (Garg & Bridgman Lemma 2). Some executions force every
weighted BA protocol to run at least α_ρ rounds. (It generalises Theorem 1.2:
the adversary can make the α_ρ − 1 heaviest processes faulty, since together
they weigh at most ρ.)

## 4. Weighted Queen (ρ < 1/4) [paper]

Each round: exchange values; `myvalue` = the value with weight > 1/2 (0 if
none); `myweight` = its weight. The queen (the k-th heaviest) sends her
`myvalue`. Keep `myvalue` if `myweight > 1/2 + ρ`, otherwise take the queen's
value. Run α_ρ rounds.

**Lemma 4.1 (persistence).** If every honest process holds v at the start of a
round, they all hold v at its end.
*Proof.* Each honest process sees weight ≥ W(H) ≥ 1 − ρ for v. Since ρ < 1/4,
1 − ρ > 1/2 + ρ, so each keeps v. ∎ (This is the step that needs ρ < 1/4.)

**Lemma 4.2 (an honest queen forces agreement).** In a round with an honest
queen, all honest processes end the round with the same value.
*Proof.* If no honest process keeps its own value, all take the queen's single
value. Otherwise some honest p has `myweight > 1/2 + ρ` for v. Faulty processes
contribute at most ρ, so honest votes for v weigh more than 1/2. Then the queen
also sees more than 1/2 for v and sends v. Another honest process cannot keep a
different value v′, which would need honest votes for v′ above 1/2 as well. So
every honest process ends with v. ∎

**Theorem 4.3** (Garg & Bridgman Theorem 1). Weighted Queen solves weighted BA
for ρ < 1/4 in α_ρ rounds of 2 steps.
*Proof.* Lemma 3.1 gives a round with an honest queen; Lemma 4.2 gives agreement
there; Lemma 4.1 keeps it until the end. Validity is Lemma 4.1 from round 1.
Termination is the fixed loop. ∎

**Lemma 4.4 (detection, Garg & Bridgman Lemma 8).** If an honest process has
`myweight > 1/2 + ρ` for v and receives a queen value different from v, the queen
is faulty.
*Proof.* By the proof of Lemma 4.2, an honest queen would have sent v. ∎

## 5. Weighted King (ρ < 1/3) [paper]

This is the form in our code (Garg & Bridgman Fig. 2, written with ρ). Phase 1:
exchange V; set V to the value with weight ≥ 1 − ρ, else *undecided*.
Phase 2: exchange V; with s₀, s₁, s_u the weights seen (missing counts as
undecided), set V to the first of 0, 1, u whose weight exceeds ρ and remember
that weight as `myweight`. Phase 3: the king (the k-th heaviest) sends V; if V is
undecided or `myweight < 1 − ρ`, adopt the king's value (1 if it is undecided).

**Lemma 5.1.** After phase 1, honest processes hold at most one value v ∈ {0, 1},
the rest being undecided.
*Proof.* Two honest processes adopting 0 and 1 would each see a ≥ 1 − ρ quorum
for their value; by Lemma 2.2 the quorums share an honest process, which sent
only one value. ∎

**Lemma 5.2 (a strong honest process pins everyone).** If an honest p ends phase
2 with V = v ≠ u and `myweight ≥ 1 − ρ`, then every honest process (the king
included) ends phase 2 with V = v.
*Proof.* p saw weight ≥ 1 − ρ for v, of which at least 1 − 2ρ is honest. Every
honest q therefore sees more than ρ for v (Corollary 2.3). Honest phase-2 values
are v or u (Lemma 5.1), so the other value v̄ is sent only by faulty processes and
weighs at most ρ at q, never more than ρ. Whether q checks 0 or 1 first, the
first value whose weight exceeds ρ is v. ∎

**Theorem 5.3** (Garg & Bridgman Theorem 2). Weighted King solves weighted BA
for ρ < 1/3 in α_ρ rounds of 3 steps.
*Proof.* In a round with an honest king: if some honest process is strong,
Lemma 5.2 makes every honest process and the king hold v, so all end with v;
otherwise all adopt the king's one value. Persistence: if all honest hold v,
both phases show weight ≥ 1 − ρ for v, so everyone keeps v. Lemma 3.1 supplies an
honest king. ∎

**Lemma 5.4 (detection) [ours].** If an honest process ends phase 2 strong for
v and the king sends a value other than v, the king is faulty.
*Proof.* By Lemma 5.2 an honest king holds v and sends it. ∎

## 6. Weighted Gradecast (one leader) [ours, following Feldman & Micali]

Leader L sends v. Each positive-weight process echoes what it got from L. A
positive-weight process *supports* m if the echoes for m weigh ≥ 1 − ρ. Each
process takes m′ = the value with the most supporting weight and outputs
(m′, 2) if that weight is ≥ 1 − ρ, (m′, 1) if it is > ρ, and (⊥, 0) otherwise.
Assume ρ < 1/3.

**Lemma 6.1 (honest supporters agree).** Two honest processes never support
different values.
*Proof.* Their echo quorums (each ≥ 1 − ρ) share an honest echoer (Lemma 2.2),
and an honest echoer echoes one value. One process cannot support two values
either, since 2(1 − ρ) > 1. ∎

**Property 6.2 (honest leader).** If L is honest, every honest process outputs
(v, 2).
*Proof.* All honest processes echo v, weight ≥ 1 − ρ, so all of them support v,
weight ≥ 1 − ρ, so each gets grade 2. Other values get at most ρ < 1 − ρ of
(faulty) support, so m′ = v. ∎

**Property 6.3 (grade ≥ 1 means the same value).** If honest p and q output
grades ≥ 1, they output the same value.
*Proof.* Grade ≥ 1 means support weight > ρ, which includes an honest supporter
(Lemma 2.1). By Lemma 6.1, all honest supporters support one value m, so both p
and q took m′ = m. ∎

**Property 6.4 (grades differ by at most 1).** If an honest p outputs grade 2,
every honest q outputs grade ≥ 1.
*Proof.* p saw support ≥ 1 − ρ for m, of which ≥ 1 − 2ρ > ρ is honest. Honest
supports go to everyone, so q sees more than ρ for m: grade ≥ 1. ∎

## 7. Weighted Gradecast consensus (ρ < 1/3) [ours, following Ben-Or, Dolev & Hoch]

Every iteration, every positive-weight process P_j gradecasts its value, all in
parallel (3 steps). Process i gets (g[j], c[j]) for every leader j and:

1. maj := the value maximising W({j : c[j] ≥ 1, g[j] = value});
2. BAD := BAD ∪ {j : c[j] ≤ 1} (positive-weight leaders only);
3. V := maj;
4. if W({j : c[j] = 2, g[j] = maj}) ≥ 1 − ρ, exit after one more iteration.

Messages from processes in BAD are ignored. The loop runs at most cap_ρ times,
where cap_ρ = 1 + (the largest number of positive-weight processes whose total
weight is ≤ ρ, counted lightest first).

Call a leader j **disputed** in an iteration if some honest process grades it 0
and another grades it ≥ 1.

**Claim 7.1 (BAD is sound).** No honest process is ever in an honest BAD set.
*Proof.* By induction on iterations. If no honest process was in BAD so far, all
honest messages are read, so Property 6.2 applies: an honest leader gets grade 2
everywhere and is not added. ∎
So honest weight ≥ 1 − ρ is always counted, and Section 6 keeps applying.

**Claim 7.2 (persistence).** If every honest process starts an iteration with v,
all of them end it with v and see the exit condition.
*Proof.* Honest leaders get (v, 2) everywhere (Property 6.2), weight ≥ 1 − ρ.
Faulty leaders weigh at most ρ < 1 − ρ, so maj = v and the grade-2 weight for v
is ≥ 1 − ρ. ∎ (This gives validity.)

**Claim 7.3 (undisputed iterations agree).** If no leader is disputed in an
iteration, all honest processes compute the same maj.
*Proof.* For each leader, either all honest processes grade it 0, or all grade it
≥ 1 with the same value (Property 6.3). So every honest process adds up the same
leaders with the same values and the same weights. ∎

**Claim 7.4 (a faulty leader is disputed at most once).** A disputed leader is
faulty and is in every honest BAD set from then on; it is never disputed again.
*Proof.* Honest leaders are never disputed (Property 6.2). If j is disputed, one
honest process grades it 0 and another ≥ 1; by Property 6.4 that grade is exactly
1. No honest process can grade j as 2, because then every honest process would
grade it ≥ 1 (Property 6.4), contradicting the 0. So every honest process grades
j at most 1 and adds it to BAD. Afterwards no honest process reads j's messages,
so no honest process echoes anything for j. The echoes for j then weigh at most
ρ < 1 − ρ (faulty ones only), so no honest process supports anything for j, and
the supports for j weigh at most ρ, so every honest process grades it 0, the same
everywhere. ∎

**Claim 7.5 (safe exit).** If an honest p sees the exit condition for v in
iteration r, every honest process holds v at the end of iteration r.
*Proof.* The leaders p graded 2 for v weigh ≥ 1 − ρ. By Properties 6.3 and 6.4
every honest q grades them ≥ 1 with value v, so q's weight for v is ≥ 1 − ρ and
any other value has at most the remaining ρ < 1 − ρ. So q's maj is v. ∎
Then Claim 7.2 makes every honest process see the exit condition in iteration
r + 1. p's extra iteration r + 1 is exactly what lets them see it (p's messages
count toward their thresholds). A process returns its value before using the
results of its own extra iteration, so the processes that have already left do
not matter.

**Proposition 7.6 [ours].** For ρ < 1/3, weighted Gradecast consensus solves
weighted BA, with at most cap_ρ iterations of 3 steps. (A process that sees the
exit condition in iteration cap_ρ returns at once: the loop never starts iteration
cap_ρ + 1, matching `gradecast.py`.)
*Proof.* Let F be the number of positive-weight faulty processes. Each disputed
iteration uses up a distinct one of them (Claim 7.4), so among any F + 1
iterations one is undisputed, and all honest processes agree after it (Claim 7.3)
and stay agreed (Claim 7.2). The faulty processes weigh ≤ ρ, so F is at most the
number of lightest positive-weight processes that fit in ρ, i.e. F + 1 ≤ cap_ρ.
So agreement holds before the loop's cap. Exits are safe by Claim 7.5, and
validity is Claim 7.2. ∎

**Corollary 7.7 (early stopping) [ours].** With F actual positive-weight faulty
processes, every honest process stops after at most min(F + 3, cap_ρ)
iterations: at most F + 1 to reach an undisputed iteration, 1 more to see the exit
condition, and 1 extra iteration to carry the others along.

*Comparison.* Ben-Or, Dolev & Hoch state min(f + 2, t + 1). Our count is one
higher because it includes the extra iteration and counts the iteration in which
agreement is reached separately from the one in which it is seen. The slides use
the conservative F + 3. The worst case cap_ρ is driven by the *lightest*
processes, while King's α_ρ is driven by the *heaviest* (Section 3).

**Repeated instances.** When BAD sets are carried into the next instance, they
can differ between honest processes. A process that returns in its extra
iteration does not add leaders disputed in that iteration. Corollary 7.7 still
holds if F counts the positive-weight faulty processes that are *not* yet in every
honest BAD set: a leader in every honest BAD set is graded 0 everywhere and can
never be disputed again (proof of Claim 7.4). The code passes that intersection
to the adversary as `expelled_all`.

**Why the cap uses the lightest processes.** Claim 7.4 means each faulty process
buys the adversary at most one disputed iteration, so the adversary wants as many
faulty processes as possible within weight ρ, which means the lightest ones.

## 8. Weight update with BAD sets [ours, built on Garg & Bridgman Fig. 3]

Run after each consensus instance:

1. **Detection.** Each honest process has a set D_i of processes it has proved
   faulty. Every protocol uses Garg & Bridgman's Lemma 8 part 1 (Lemma 8.0
   below). On top of that: Queen adds Lemma 4.4, King adds Lemma 5.4, and
   Gradecast adds its BAD set.
2. **Learning.** Every process broadcasts D_i.
3. **Agreement.** For each candidate j, all processes run one binary weighted BA
   (all N in parallel). Process i's input is 1 iff the processes reporting j weigh
   more than ρ.
4. **Update.** Set w[j] := 0 for every j decided 1. The total shrinks; this is the
   renormalisation.

**Lemma 8.0 (Garg & Bridgman Lemma 8 part 1).** A positive-weight process that,
in a step where it must send, sends nothing or a malformed message is faulty.
*Proof.* Honest processes always send well-formed messages, and channels are
reliable and synchronous. ∎
*Caveat for Gradecast.* Honest processes stop sending after their extra
iteration. Lemma 8.0 is therefore applied only before the detecting process
reaches its own extra iteration. By Claim 7.5, every honest process is still
running then: if some honest q is already silent in iteration k, q returned
during iteration k − 1 (its extra iteration), so it saw the exit condition in
k − 2. By Claim 7.5 every honest process then saw it by k − 1 and is in its extra
iteration at k, where it does not apply Lemma 8.0. "Well-formed" means exactly what the protocol
parses as a value, including the one-value shorthand for an echo vector.

**Theorem 8.1 (no honest process is ever removed).**
*Proof.* Detection is sound (Lemma 8.0, Claim 7.1, Lemma 4.4, Lemma 5.4), so no honest
process reports an honest j. Its reporters are all faulty and weigh ≤ ρ, so every
honest input for j is 0, and validity of the agreement decides 0. ∎

**Theorem 8.2 (all honest processes keep identical weights).** The agreement
step decides the same bit for every j at every honest process. ∎

**Theorem 8.3 (what gets removed).** If the *honest* processes that detected j
weigh more than ρ, j is removed.
*Proof.* Every honest process then sees more than ρ of reporters, so every
honest input is 1, and validity decides 1. ∎
If the honest reporters weigh ≤ ρ, the adversary can make the inputs mixed (by
reporting j to some processes only) and the outcome is not guaranteed. The
experiment assumes the worst case: j is kept.

**Corollary 8.4 (what each method catches).**
- *Silent or malformed* processes are caught by every method (Lemma 8.0). They
  are caught at every honest process, so their honest reporters weigh
  W(H) ≥ 1 − ρ > ρ and they are removed (Theorem 8.3).
- *Inconsistent* processes, which send well-formed but different values to
  different processes:
  - Gradecast catches a leader when it is disputed (Claim 7.4): every honest
    process grades it ≤ 1, so it is removed.
  - A leader that lies to only a few processes can still get grade 2 at most of
    them. It is then reported by honest weight ≤ ρ and may survive. This is the
    partial detection seen for the "equivocate" attack.
  - Queen and King catch an inconsistent process only when it is the round's
    coordinator *and* contradicts a strong majority (Lemmas 4.4, 5.4). Light
    processes are never coordinators.
- *Consistent liars*, which send the same wrong but well-formed value to
  everyone, are caught by no method. As a Gradecast leader such a process gets
  grade 2. Lies inside echo or support messages are never attributed to the
  liar, since BAD grades leaders only.

**Remark 8.5 (limits).**
- *Consistent liars and sleepers* are never detected (Corollary 8.4). Against
  them every update method equals "no update"; the experiment shows this
  ("flip" attack).
- *Sleepers.* A faulty process that behaves perfectly is never detected by any
  method. A patient adversary can accumulate weight while behaving and attack once
  it holds 1/3. Weight updates defend against faults that show symptoms (crashes,
  bugs, attacks), not against strategic sleepers. Against sleepers every method
  equals "no update".
- The guarantee for an instance still needs the *current* faulty weight
  (detected or not) to be below the bound relative to the *current* total. The
  update helps because removed processes no longer count in either.

## 9. The numbers in the results slides

Group A has n_A processes failing with probability f_A, group B has n_B failing
with probability f_B, all independently. Every A process gets weight w_A and every
B process w_B; W = n_A w_A + n_B w_B.

**9.1 Chance the guarantee is lost (exact).**

  P(lost) = Σ_{a=0}^{n_A} Σ_{b=0}^{n_B} Bin(a; n_A, f_A) · Bin(b; n_B, f_B) · 1[a·w_A + b·w_B ≥ τ·W]

with τ = 1/3 (King, Gradecast) or 1/4 (Queen), where Bin(k; n, p) = C(n, k) p^k (1 − p)^(n−k).
We compare with integer arithmetic, so a faulty weight of exactly τW counts as
lost, as Lemma 1.4 requires.

**9.2 Rounds (exact).** King uses α_ρ rounds (Section 3). Gradecast's worst case
is cap_ρ (Section 7). Its "expected bound" is

  E = Σ_{safe (a,b)} P(a, b) · min(F(a, b) + 3, cap_ρ) / P(safe),

with F(a, b) = a·1[w_A > 0] + b·1[w_B > 0]. This averages the proven bound of
Corollary 7.7 over the failure model; real runs can only be at or below it.

**9.3 Failures over time (appendix experiment).** Process i fails at instance
T_i, with P(T_i ≤ k) = 1 − (1 − q_i)^k, and stays faulty. Without updates the
faulty set only grows, so

  P(guarantee holds at instance k) = Σ_{a,b} Bin(a; n_A, p_A(k)) · Bin(b; n_B, p_B(k)) · 1[a·w_A + b·w_B < τ·W],

with p(k) = 1 − (1 − q)^k (exact). With updates the curve is estimated by
simulation, applying Theorems 8.1 and 8.3 at the agreement step.

## 10. The ideal weight ratio

**10.1 Exact optimum (two classes).** With r = w_A / w_B, the event
a·r + b ≥ τ(n_A r + n_B) is a linear test in r for each (a, b). P(lost) is
therefore piecewise constant in r and changes only at the breakpoints
r = (τ n_B − b)/(a − τ n_A). Evaluating P at every breakpoint, at every midpoint
between neighbours, and at r = 0 and r = ∞ ("only A") finds the exact minimum.
This is `optimal_ratio` in the code.

**10.2 Closed form [approx].** Give class i weight w_i and failure probability p_i,
with n_i members. The faulty weight X = Σ w_i · Bin(n_i, p_i) has

  mean μ = Σ n_i w_i p_i,  variance σ² = Σ n_i w_i² p_i (1 − p_i).

With a normal approximation, P(X ≥ τW) ≈ Φ(−z), where

  z = (τW − μ)/σ = Σ n_i w_i (τ − p_i) / sqrt(Σ n_i w_i² p_i (1 − p_i)).

Minimising P(lost) means maximising z. Write a_i = √(n_i p_i (1 − p_i)) · w_i and
b_i = √n_i (τ − p_i) / √(p_i (1 − p_i)). Then z = Σ a_i b_i / ‖a‖, and by
Cauchy–Schwarz this is largest when a ∝ b, i.e.

  **w_i ∝ (τ − p_i) / (p_i (1 − p_i))**, and **w_i = 0 when p_i ≥ τ**.

The cut-off comes from the constraint w_i ≥ 0: a class with p_i ≥ τ only lowers
the numerator and raises the variance, so the KKT conditions give it weight 0.
The formula works for any number of classes and does not depend on the class
sizes n_i.

**10.3 Our setup.** f_A = 0.1, f_B = 0.3:

| bound τ | w_A | w_B | ratio w_A : w_B |
|---|---|---|---|
| 1/3 (King, Gradecast) | 0.2333/0.09 = 2.59 | 0.0333/0.21 = 0.159 | **16.3 : 1** |
| 1/4 (Queen) | 0.15/0.09 = 1.67 | 0 (f_B ≥ 1/4) | **only A** |

The paper's best weighting (inverse failure probability) is 3 : 1. At
|B| = 60 the formula gives P(lost) = 0.0484, the same as the exact optimum
(0.0484), against 0.0906 for inverse weights. At |B| = 399: 0.0188 (formula),
0.0165 (exact), 0.0370 (inverse).

*Not a tie artifact.* With |A| = 6, two faulty A processes weigh exactly 1/3,
which could make "only A" look bad for a reason unrelated to the formula. So we
re-checked with |A| = 7 and |A| = 10, where that tie cannot happen. The formula
still beats the inverse weights by 2–3× at every |B| tested (30, 60, 200, 399):
for example 0.0139 vs 0.0322 at |A| = 7, |B| = 399. It is always within 0.003 of
the exact optimum. At small |B| with |A| ≥ 7, "only A" is slightly better than
the formula (0.0257 vs 0.0280 at |A| = 7, |B| = 60).

*Why it is only approximate.* The normal approximation is poor when classes are
small (|A| = 6) and in the far tail. The exact search of 10.1 is the ground truth;
the formula is a good starting point, and here lands within 0.003 of the
optimum.

## 11. Summary of what is proven

| Claim | Status |
|---|---|
| N > 3t, t + 1 rounds, FLP | cited |
| ρ ≥ 1/3 impossible; α_ρ lower bound | cited (Garg & Bridgman) |
| Weighted Queen (ρ < 1/4), weighted King (ρ < 1/3) | paper; restated in §4–5 |
| Detection lemmas (Queen Lemma 8; King analogue) | paper; King analogue ours (§5.4) |
| Weighted Gradecast properties | ours (§6), adapting Feldman & Micali |
| Weighted Gradecast consensus correct, cap_ρ, F + 3 early stop | ours (§7), adapting Ben-Or et al.; proposition, not peer reviewed |
| Weight update never removes honest processes; identical weights | ours (§8), built on Garg & Bridgman Fig. 3 |
| Probability and round formulas | exact computations (§9) |
| Ideal ratio formula | approximation (§10.2); the exact optimum is computed (§10.1) |
| Simulations | evidence, not proof: they check that the implementation matches §4–8 |
