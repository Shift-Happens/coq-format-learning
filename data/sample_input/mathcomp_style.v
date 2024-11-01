From mathcomp Require Import all_ssreflect.

Lemma example_proof n m : n + m = m + n.
Proof.
move=> n m.
elim: n => [|n' IHn] /=.
  by rewrite addnC.
by rewrite IHn addnS.
Qed.

Definition is_even n := exists k, n = k + k.

Lemma even_plus : forall n m,
  is_even n -> is_even m -> is_even (n + m).
Proof.
move=> n m [k1 Hk1] [k2 Hk2].
exists (k1 + k2).
by rewrite Hk1 Hk2 addnA [k2+_]addnC addnA.
Qed.