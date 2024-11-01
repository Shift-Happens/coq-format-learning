Require Import Arith.
Require Import Bool.

Theorem plus_comm : forall n m : nat,
  n + m = m + n.
Proof.
  intros n m.
  induction n as [| n' IHn'].
  - simpl. rewrite Nat.add_0_r. reflexivity.
  - simpl. rewrite IHn'. rewrite Nat.add_succ_r. reflexivity.
Qed.

Lemma bool_neg_inv : forall b : bool,
  negb (negb b) = b.
Proof.
  intros b. destruct b; simpl; reflexivity.
Qed.

