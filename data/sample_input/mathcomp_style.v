From mathcomp Require Import all_ssreflect.

(* Lemma: Commutativity of addition *)
Lemma example_proof (n m : nat) : n + m = m + n.
Proof.
  move=> n m.
  elim: n => [|n' IHn] /=.
  - by rewrite addnC.
  - by rewrite IHn addnS.
Qed.

(* Definition: Evenness *)
Definition is_even (n : nat) := exists k, n = k + k.

(* Lemma: Evenness is preserved under addition *)
Lemma even_plus (n m : nat) : is_even n -> is_even m -> is_even (n + m).
Proof.
  move=> n m [k1 Hk1] [k2 Hk2].
  exists (k1 + k2).
  by rewrite Hk1 Hk2 addnA [k2+_]addnC addnA.
Qed.

(* Lemma: Evenness is preserved under multiplication *)
Lemma even_mult (n m : nat) : is_even n -> is_even m -> is_even (n * m).
Proof.
  move=> n m [k1 Hk1] [k2 Hk2].
  exists (k1 * k2).
  by rewrite Hk1 Hk2 mulnC mulnA.
Qed.

(* Lemma: Oddness is preserved under addition *)
Lemma odd_plus (n m : nat) : ~is_even n -> ~is_even m -> ~is_even (n + m).
Proof.
  move=> n m Hn Hm.
  intros [k Hk].
  apply: Hn.
  exists k.
  by rewrite -Hk addnC addnK.
Qed.

(* Lemma: Oddness is preserved under multiplication *)
Lemma odd_mult (n m : nat) : ~is_even n -> ~is_even m -> ~is_even (n * m).
Proof.
  move=> n m Hn Hm.
  intros [k Hk].
  apply: Hn.
  exists k.
  by rewrite -Hk mulnC mulnK.
Qed.

(* Lemma: The sum of two odd numbers is even *)
Lemma odd_sum_even (n m : nat) : ~is_even n -> ~is_even m -> is_even (n + m).
Proof.
  move=> n m Hn Hm.
  exists (n / 2 + m / 2).
  by rewrite addn_div2 addn_div2 addnA.
Qed.

(* Lemma: If n is odd, then n^2 is odd *)
Lemma odd_square (n : nat) : ~is_even n -> ~is_even (n * n).
Proof.
  move=> n Hn.
  intros [k Hk].
  apply: Hn.
  exists (k * 2).
  by rewrite -Hk mulnS mulnC.
Qed.
