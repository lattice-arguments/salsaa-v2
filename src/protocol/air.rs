//! AIR support: proving a committed squaring trace (`Π_air`, `sec:piair`) on
//! top of the generic sumcheck machinery.
//!
//! Following the paper, the committed witness is `V := [W, shift(W)]` with
//! `shift(W)_{i,:} = W_{(i+1) mod ℓ,:}`, committed as one ordinary `relLin`
//! instance over all its columns. The verifier treats that commitment exactly
//! like any other — it does not reconstruct, derive or re-check any part of it.
//!
//! The *prover* does exploit one identity to build it faster. With a
//! power-series commitment key (`key[i] = key[0]·gⁱ`, the geometric
//! instantiation of the row-tensor vSIS key — see `gen_air_crs`) the columns of
//! `shift(W)` need no inner products at all:
//!
//!   com(shift(D)) = g⁻¹·( com(D) + key[0]·(g^ℓ − 1)·D[0] )
//!
//! The trace is one register under consecutive squaring: `w_{i+1} = w_i²`,
//! public input `x = w_0`, public claimed output `y = w_{ℓ-1}`, so
//! `f(Y₀, Y₁) = Y₀² − Y₁` and `C = {(0,0,x), (ℓ-1,0,y)}`.
//!
//! `relLin` needs a short witness while trace values are full-size mod q, so
//! each column of `V` is carried as `k = AIR_DIGIT_COLS` balanced
//! base-`2^AIR_BASE_LOG` digit columns (the existing l2 machinery proves their
//! norm): columns `0..k` are the digits of `W`, columns `k..2k` the digits of
//! `shift(W)`. Digit decomposition is element-wise, so it commutes with the
//! shift. Each block is recovered by the gadget vector `g = (1, b, …, b^{k-1})`
//! — `V₀ = Σ_m bᵐ·D_m` and `V₁ = Σ_m bᵐ·D_{k+m}` — and the first-round sumcheck
//! gains the paper's four claims over the row variables `z` (the
//! `μ = log₂ ℓ` least-significant variables):
//!
//!   transition:  Σ_z eq(η,z)·(1 − eq(z,1⃗))·(V₀(z)² − V₁(z)) = 0
//!   shift:       Σ_z θ̃(z)·V₀(z) − θshift(z)·V₁(z)           = 0
//!   boundary:    Σ_z eq(z,0⃗)·V₀(z) = x,   Σ_z eq(z,1⃗)·V₀(z) = y
//!
//! where `θ̃ = (1, θ, …, θ^{ℓ-1})` and `θshift = (θ, θ², …, θ^{ℓ-1}, 1)`, i.e.
//! `θ·θ̃ − (θ^ℓ − 1)·eq(·,1⃗)`, forcing `V₁ = shift(V₀)`.
//!
//! The quadratic transition runs on prover-side auxiliary `LinearSumcheck`s
//! loaded with the composed columns; the
//! verifier reconstructs their final evaluations from the per-column claims by
//! gadget recomposition, so the claims stay bound to the committed columns.

use crate::common::config::*;
use rokoko::common::{
    arithmetic::{ONE, ZERO},
    decomposition::decompose_chunks_into,
    hash::HashWrapper,
    matrix::{VerticallyAlignedMatrix, new_vec_zero_preallocated},
    ring_arithmetic::{Representation, RingElement},
    structured_row::{PreprocessedRow, StructuredRow},
};
use rokoko::protocol::{
    commitment::{BasicCommitment, commit_basic_internal},
    crs::CRS,
};

pub const AIR_DIGIT_COLS: usize = 8;
pub const AIR_BASE_LOG: u64 = 8; // decomp_base_log

pub struct AirCrsAux {
    /// Inverse of the geometric ratio `g_r` of each commitment-key row.
    pub g_inv: Vec<RingElement>,
    /// `key_r[0] · (g_r^ℓ − 1)` for the full witness height `ℓ`.
    pub correction: Vec<RingElement>,
}

pub struct AirOutput {
    pub input: RingElement,
    pub output: RingElement,
    /// Balanced digits of the input, i.e. the first rows of the `W` digit
    /// columns.
    pub input_digits: Vec<RingElement>,
    /// The `2k`-column witness (digit columns of `W`, then of `shift(W)`).
    pub witness: VerticallyAlignedMatrix<RingElement>,
}

pub struct AirChallenges {
    pub eta: Vec<RingElement>,
    pub theta: RingElement,
}

/// Prover-side data for the AIR sumcheck loader: the composed rows plus the
/// challenges.
pub struct AirLoaderData {
    pub v0: Vec<RingElement>,
    pub v1: Vec<RingElement>,
    pub challenges: AirChallenges,
}

pub fn ring_inverse(el: &RingElement) -> RingElement {
    let mut tmp = el.clone();
    tmp.from_incomplete_ntt_to_homogenized_field_extensions();
    let mut inv = tmp.inverse();

    inv.from_homogenized_field_extensions_to_incomplete_ntt();
    inv
}

/// Assembles a `CRS` from ready-made per-rank tensor-layer rows. This mirrors
/// the slicing in `CRS::gen_crs` (which samples its layers itself, so it cannot
/// be reused directly): the key for dimension `2^i` takes the last `i` layers,
/// and layer `max_log - 1 - p` carries index bit `p`.
fn crs_from_module_rows(module_rows: &[Vec<RingElement>], max_log: usize) -> CRS {
    let rank = module_rows.len();
    let (cks, structured_cks): (Vec<_>, Vec<_>) = (1..=max_log)
        .map(|i| {
            let mut ck = Vec::with_capacity(rank);
            let mut sck = Vec::with_capacity(rank);
            for row in module_rows.iter() {
                let structured_row = StructuredRow {
                    tensor_layers: row.iter().skip(max_log - i).cloned().collect(),
                };
                ck.push(PreprocessedRow::from_structured_row(&structured_row));
                sck.push(structured_row);
            }
            (ck, sck)
        })
        .unzip();

    CRS {
        cks,
        structured_cks,
    }
}

/// Generates a CRS whose commitment keys are a power series: for each rank row
/// `r` a random `g_r` is sampled and the tensor layer for bit position `p` is
/// `a_p = g_r^{2^p} · (1 + g_r^{2^p})⁻¹`, so that
/// `eq-tensor(a, i) = Π_p(1 − a_p) · g_r^i = key_r[0] · g_r^i`.
///
pub fn gen_air_crs(max_wit_dim: usize, rank: usize) -> (CRS, AirCrsAux) {
    debug_assert!(max_wit_dim.is_power_of_two());
    let max_log = max_wit_dim.ilog2() as usize;

    let mut g_inv = Vec::with_capacity(rank);
    let mut g_pow_h = Vec::with_capacity(rank);
    let mut module_rows: Vec<Vec<RingElement>> = Vec::with_capacity(rank);

    for _ in 0..rank {
        let g = RingElement::random(Representation::IncompleteNTT);
        let mut layers = new_vec_zero_preallocated(max_log);
        // gp = g^{2^p}; layer order: the *last* tensor layer corresponds to the
        // least-significant bit, so bit position p goes to index max_log-1-p.
        let mut gp = g.clone();
        for p in 0..max_log {
            let one_plus = &*ONE + &gp;
            let inv = ring_inverse(&one_plus);
            layers[max_log - 1 - p] = &gp * &inv;
            gp = &gp * &gp;
        }
        // After the loop gp = g^{2^max_log} = g^ℓ.
        g_pow_h.push(gp);
        module_rows.push(layers);
        g_inv.push(ring_inverse(&g));
    }

    let crs = crs_from_module_rows(&module_rows, max_log);

    // correction_r = key_r[0] · (g_r^ℓ − 1) for the full witness height.
    let full_ck = crs.ck_for_wit_dim(max_wit_dim);
    let mut correction = Vec::with_capacity(rank);
    for r in 0..rank {
        let g_h_minus_one = &g_pow_h[r] - &*ONE;
        correction.push(&full_ck[r].preprocessed_row[0] * &g_h_minus_one);
    }

    (crs, AirCrsAux { g_inv, correction })
}

pub fn air_witness(x: &RingElement, height: usize) -> AirOutput {
    let k = AIR_DIGIT_COLS;

    let mut input = x.clone();
    input.to_representation(Representation::IncompleteNTT);

    let mut trace: Vec<RingElement> = Vec::with_capacity(height);
    trace.push(input.clone());
    for i in 1..height {
        let sq = &trace[i - 1] * &trace[i - 1];
        trace.push(sq);
    }
    let output = trace[height - 1].clone();

    let mut data = new_vec_zero_preallocated(height * 2 * k);

    // Columns 0..k: digit `m` of every trace entry. `decompose_chunks_into`
    // writes digit-major, which is exactly the column-major layout of a
    // `VerticallyAlignedMatrix` (`data[col * height + row]`).
    decompose_chunks_into(&mut data[..k * height], &trace, AIR_BASE_LOG, k);

    // The digits of the (public) input are the first row of each digit column;
    // the prover needs them for the O(1) shifted-column commitments.
    let input_digits: Vec<RingElement> =
        (0..k).map(|m| data[m * height].clone()).collect();

    // Columns k..2k: `shift` of the digit columns, i.e. row `i` takes row
    // `i+1 mod ℓ`. Digit decomposition is element-wise, so it commutes with the
    // shift.
    for m in 0..k {
        let (head, tail) = data.split_at_mut((k + m) * height);
        let src = &head[m * height..(m + 1) * height];
        let dst = &mut tail[..height];
        for i in 0..height - 1 {
            dst[i].set_from(&src[i + 1]);
        }
        dst[height - 1].set_from(&src[0]);
    }

    AirOutput {
        input,
        output,
        input_digits,
        witness: VerticallyAlignedMatrix {
            height,
            width: 2 * k,
            data,
            used_cols: 2 * k,
        },
    }
}

/// Commits the AIR witness. Oly the `k` digit columns of `W` cost inner
/// products: with a power-series key the columns of `shift(W)` follow in `O(1)`
/// ring operations each from `com(shift(D)) = g⁻¹·( com(D) + key[0]·(g^ℓ − 1)·D[0] )`.
pub fn commit_air(
    crs: &CRS,
    aux: &AirCrsAux,
    witness: &mut VerticallyAlignedMatrix<RingElement>,
    rank: usize,
    input_digits: &[RingElement],
) -> BasicCommitment {
    let k = AIR_DIGIT_COLS;
    debug_assert_eq!(witness.width, 2 * k);

    // leave the shifted half zeroed and ready to be filled in below
    let full_cols = witness.used_cols;
    witness.used_cols = k;
    let mut commitment = commit_basic_internal(crs.ck_for_wit_dim(witness.height), witness, rank);
    witness.used_cols = full_cols;

    let mut temp = RingElement::zero(Representation::IncompleteNTT);
    for r in 0..rank {
        for m in 0..k {
            temp *= (&aux.correction[r], &input_digits[m]);
            temp += &commitment[(r, m)];
            commitment[(r, k + m)] = &aux.g_inv[r] * &temp;
        }
    }

    commitment
}

pub fn sample_air_challenges(mu: usize, hash_wrapper: &mut HashWrapper) -> AirChallenges {
    let mut eta = new_vec_zero_preallocated(mu);
    hash_wrapper.sample_ring_element_vec_into(&mut eta);
    let mut theta = RingElement::zero(Representation::IncompleteNTT);
    hash_wrapper.sample_ring_element_ntt_slots_into(&mut theta);
    AirChallenges { eta, theta }
}

/// Entry `m` of the gadget vector `g = (1, b, b², …, b^{k-1})`, reduced mod q.
#[inline]
pub fn air_gadget_scalar(m: usize) -> RingElement {
    RingElement::constant(
        (1u64 << (AIR_BASE_LOG * m as u64)) % MOD_Q,
        Representation::IncompleteNTT,
    )
}

/// The gadget vector placed on one block of `V = [W, shift(W)]`: `g` on columns
/// `0..k` for the `W` block, on columns `k..2k` for the `shift(W)` block, zero
/// elsewhere. These are the two block rows of `G = I₂ ⊗ gᵀ`, and
/// `⟨weights, row i⟩` recomposes that block's digits into its trace value.
///
/// In the paper's `Π_air` these occupy the slot of the block selectors
/// `e_{0,2} ⊗ α` and `e_{1,2} ⊗ α` (and `e_{j_k,2t}` in the boundary claims).
/// With one register `α` is a scalar, so the selectors are just `(1,0)`/`(0,1)`
/// there; here the same position carries `g` instead, because the committed
/// columns are digits rather than trace values.
pub fn air_gadget_weights(cols: usize, shift_block: bool) -> Vec<RingElement> {
    let k = AIR_DIGIT_COLS;
    let mut weights = new_vec_zero_preallocated(cols);
    for m in 0..k {
        let idx = if shift_block { k + m } else { m };
        weights[idx] = air_gadget_scalar(m);
    }
    weights
}

/// The composed rows `V₀ = Σ_m 2^{bm}·col_m` and `V₁ = Σ_m 2^{bm}·col_{k+m}`.
pub fn air_composed_columns(
    witness: &VerticallyAlignedMatrix<RingElement>,
) -> (Vec<RingElement>, Vec<RingElement>) {
    let k = AIR_DIGIT_COLS;
    let h = witness.height;
    let mut v0 = new_vec_zero_preallocated(h);
    let mut v1 = new_vec_zero_preallocated(h);
    let mut temp = RingElement::zero(Representation::IncompleteNTT);
    for m in 0..k {
        let shift = air_gadget_scalar(m);
        for (i, w) in witness.col(m).iter().enumerate() {
            temp *= (w, &shift);
            v0[i] += &temp;
        }
        for (i, w) in witness.col(k + m).iter().enumerate() {
            temp *= (w, &shift);
            v1[i] += &temp;
        }
    }
    (v0, v1)
}

/// Prover-side weight tables over the row variables:
/// `θ̃ = (1, θ, …, θ^{h-1})` and `θshift = (θ, θ², …, θ^{h-1}, 1)`.
pub fn air_theta_tables(theta: &RingElement, h: usize) -> (Vec<RingElement>, Vec<RingElement>) {
    let mut pows: Vec<RingElement> = Vec::with_capacity(h);
    pows.push(ONE.clone());
    for i in 1..h {
        let next = &pows[i - 1] * theta;
        pows.push(next);
    }
    let mut shift: Vec<RingElement> = Vec::with_capacity(h);
    for j in 0..h - 1 {
        shift.push(&pows[j] * theta);
    }
    shift.push(ONE.clone());
    (pows, shift)
}

/// Verifier-side MLE evaluations of the AIR row-weight tables at the sumcheck
/// point. `rows_ms_first` are the row-variable challenges in tree 
/// order, i.e. the last `μ` entries of `evaluation_points_ring_tree`.
pub struct AirRowEvals {
    pub transition_weight: RingElement,
    pub theta_pows: RingElement,
    pub theta_shift: RingElement,
    pub e_first: RingElement,
    pub e_last: RingElement,
}

pub fn air_verifier_row_evals(
    theta: &RingElement,
    eta: &[RingElement],
    rows_ms_first: &[RingElement],
) -> AirRowEvals {
    // MLE[(θ^i)_i](r) = Π_j (1 − r_j + r_j·θ^{2^j}), LS variable first,
    // squaring θ each step. After the loop tp = θ^h.
    let mut theta_pows = ONE.clone();
    let mut tp = theta.clone();
    let mut term = RingElement::zero(Representation::IncompleteNTT);
    for r in rows_ms_first.iter().rev() {
        let mut factor = &*ONE - r;
        term *= (r, &tp);
        factor += &term;
        theta_pows *= &factor;
        tp = &tp * &tp;
    }
    let theta_pow_h = tp;

    // eq(r, 0⃗) = Π (1 − r_j), eq(r, 1⃗) = Π r_j.
    let mut e_first = ONE.clone();
    let mut e_last = ONE.clone();
    for r in rows_ms_first.iter() {
        let one_minus = &*ONE - r;
        e_first *= &one_minus;
        e_last *= r;
    }

    // The transition weight is eq(η,·) with its last entry zeroed (see
    // `air_transition_weight_table`), whose MLE is
    // eq(η, r) − eq(η, 1⃗)·eq(r, 1⃗). Both η and r are listed MS-first here, so
    // η_j pairs with r_j directly.
    let mut eq_eta = ONE.clone();
    let mut eq_eta_at_one = ONE.clone();
    for (e, r) in eta.iter().zip(rows_ms_first.iter()) {
        // (1 − e)(1 − r) + e·r
        let mut factor = &*ONE - e;
        factor *= &(&*ONE - r);
        term *= (e, r);
        factor += &term;
        eq_eta *= &factor;
        eq_eta_at_one *= e;
    }
    term *= (&eq_eta_at_one, &e_last);
    let mut transition_weight = eq_eta;
    transition_weight -= &term;

    // θshift = θ·θ̃ − (θ^h − 1)·eq(·,1⃗) as multilinear functions.
    let mut theta_shift = &theta_pows * theta;
    let wrap = &theta_pow_h - &*ONE;
    term *= (&wrap, &e_last);
    theta_shift -= &term;

    AirRowEvals {
        transition_weight,
        theta_pows,
        theta_shift,
        e_first,
        e_last,
    }
}

/// The transition weight `eq(η, z)·(1 − eq(z, 1⃗))`, excluding the last row
/// where the cyclic shift wraps `W_{ℓ-1}` against `W_0`.
pub fn air_transition_weight_table(eta: &[RingElement]) -> Vec<RingElement> {
    let mut table = PreprocessedRow::from_structured_row(&StructuredRow {
        tensor_layers: eta.to_vec(),
    })
    .preprocessed_row;
    let last = table.len() - 1;
    table[last] = ZERO.clone();
    table
}

pub fn air_e_first_table(h: usize) -> Vec<RingElement> {
    let mut t = new_vec_zero_preallocated(h);
    t[0] = ONE.clone();
    t
}

pub fn air_e_last_table(h: usize) -> Vec<RingElement> {
    let mut t = new_vec_zero_preallocated(h);
    t[h - 1] = ONE.clone();
    t
}

#[cfg(test)]
mod tests {
    use super::*;
    use rokoko::common::init_common;
    use rokoko::protocol::sumcheck_utils::{common::SumcheckBaseData, linear::LinearSumcheck};

    fn init() {
        static INIT: std::sync::Once = std::sync::Once::new();
        INIT.call_once(init_common);
    }

    fn random_ring_vec(n: usize) -> Vec<RingElement> {
        (0..n)
            .map(|_| RingElement::random(Representation::IncompleteNTT))
            .collect()
    }

    /// The four `Π_air` claim values, evaluated straight from the composed
    /// columns with plain loops.
    struct ClaimValues {
        transition: RingElement,
        shift: RingElement,
        boundary_first: RingElement,
        boundary_last: RingElement,
    }

    fn air_claims(
        v0: &[RingElement],
        v1: &[RingElement],
        eta: &[RingElement],
        theta: &RingElement,
    ) -> ClaimValues {
        let h = v0.len();
        let w_trans = air_transition_weight_table(eta);
        let (theta_pows, theta_shift) = air_theta_tables(theta, h);
        let e_first = air_e_first_table(h);
        let e_last = air_e_last_table(h);

        let mut out = ClaimValues {
            transition: ZERO.clone(),
            shift: ZERO.clone(),
            boundary_first: ZERO.clone(),
            boundary_last: ZERO.clone(),
        };
        let mut temp = RingElement::zero(Representation::IncompleteNTT);

        for i in 0..h {
            // eq(η,z)·(1 − eq(z,1⃗))·(V₀² − V₁)
            let mut f = &v0[i] * &v0[i];
            f -= &v1[i];
            temp *= (&w_trans[i], &f);
            out.transition += &temp;

            // θ̃·V₀ − θshift·V₁
            temp *= (&theta_pows[i], &v0[i]);
            out.shift += &temp;
            temp *= (&theta_shift[i], &v1[i]);
            out.shift -= &temp;

            // eq(z,0⃗)·V₀ and eq(z,1⃗)·V₀
            temp *= (&e_first[i], &v0[i]);
            out.boundary_first += &temp;
            temp *= (&e_last[i], &v0[i]);
            out.boundary_last += &temp;
        }
        out
    }

    /// An honest trace must hit all four targets exactly, for every trace
    /// length. `ℓ = 2` is the sharp case: the wrap row sits next to both
    /// boundary rows, so an off-by-one in the row exclusion or in `θshift`'s
    /// last entry shows up there first.
    #[test]
    fn test_air_claims_hold_for_honest_trace() {
        init();
        for mu in 1..=4 {
            let h = 1usize << mu;
            let x = RingElement::random(Representation::IncompleteNTT);
            let out = air_witness(&x, h);
            let (v0, v1) = air_composed_columns(&out.witness);
            let claims = air_claims(
                &v0,
                &v1,
                &random_ring_vec(mu),
                &RingElement::random(Representation::IncompleteNTT),
            );

            assert_eq!(claims.transition, ZERO.clone(), "l={}: transition != 0", h);
            assert_eq!(claims.shift, ZERO.clone(), "l={}: shift != 0", h);
            assert_eq!(claims.boundary_first, out.input, "l={}: first != x", h);
            assert_eq!(claims.boundary_last, out.output, "l={}: last != y", h);
        }
    }

    /// A trace that skips a squaring step must be caught, and caught by the
    /// *transition* claim: the corruption is applied to both `V₀[j]` and
    /// `V₁[j-1]` so that `V₁ = shift(V₀)` still holds and the shift claim stays
    /// satisfied. Only the transition can detect this.
    #[test]
    fn test_broken_squaring_step_breaks_transition_claim() {
        init();
        let mu = 4;
        let h = 1usize << mu;
        let x = RingElement::random(Representation::IncompleteNTT);
        let out = air_witness(&x, h);
        let (mut v0, mut v1) = air_composed_columns(&out.witness);

        let j = h / 2;
        let bogus = RingElement::random(Representation::IncompleteNTT);
        v0[j] = bogus.clone();
        v1[j - 1] = bogus; // keep V₁ = shift(V₀)

        let claims = air_claims(
            &v0,
            &v1,
            &random_ring_vec(mu),
            &RingElement::random(Representation::IncompleteNTT),
        );
        assert_eq!(
            claims.shift,
            ZERO.clone(),
            "shift claim should be unaffected: V1 is still shift(V0)"
        );
        assert_ne!(
            claims.transition,
            ZERO.clone(),
            "a broken squaring step went undetected by the transition claim"
        );
    }

    /// The wrap entry `V₁[ℓ-1] = V₀[0]` is the one the transition claim cannot
    /// see, because its row is excluded by `1 − eq(z,1⃗)`. It is held solely by
    /// the `(θ^ℓ − 1)` correction in `θshift`, so corrupting it must leave the
    /// transition claim at zero and break the shift claim.
    #[test]
    fn test_broken_wrap_row_breaks_shift_claim() {
        init();
        let mu = 4;
        let h = 1usize << mu;
        let x = RingElement::random(Representation::IncompleteNTT);
        let out = air_witness(&x, h);
        let (v0, mut v1) = air_composed_columns(&out.witness);

        v1[h - 1] = RingElement::random(Representation::IncompleteNTT);

        let claims = air_claims(
            &v0,
            &v1,
            &random_ring_vec(mu),
            &RingElement::random(Representation::IncompleteNTT),
        );
        assert_eq!(
            claims.transition,
            ZERO.clone(),
            "transition claim must ignore the wrap row"
        );
        assert_ne!(
            claims.shift,
            ZERO.clone(),
            "a corrupted wrap row went undetected by the shift claim"
        );
    }

    /// The boundary claims must pin the public input and output.
    #[test]
    fn test_wrong_boundary_breaks_boundary_claims() {
        init();
        let mu = 4;
        let h = 1usize << mu;
        let x = RingElement::random(Representation::IncompleteNTT);
        let out = air_witness(&x, h);
        let (v0, v1) = air_composed_columns(&out.witness);
        let eta = random_ring_vec(mu);
        let theta = RingElement::random(Representation::IncompleteNTT);

        let mut wrong_input = v0.clone();
        wrong_input[0] = RingElement::random(Representation::IncompleteNTT);
        assert_ne!(
            air_claims(&wrong_input, &v1, &eta, &theta).boundary_first,
            out.input,
            "a wrong first row went undetected by the input boundary claim"
        );

        let mut wrong_output = v0.clone();
        wrong_output[h - 1] = RingElement::random(Representation::IncompleteNTT);
        assert_ne!(
            air_claims(&wrong_output, &v1, &eta, &theta).boundary_last,
            out.output,
            "a wrong last row went undetected by the output boundary claim"
        );
    }

    /// Every prover-side row table must agree with the verifier's O(μ)
    /// evaluation of its MLE at the sumcheck point.
    #[test]
    fn test_row_tables_match_verifier_evaluations() {
        init();
        let mu = 4;
        let h = 1 << mu;
        let eta: Vec<RingElement> = (0..mu)
            .map(|_| RingElement::random(Representation::IncompleteNTT))
            .collect();
        let theta = RingElement::random(Representation::IncompleteNTT);
        // Challenges in tree (MS-first) order, as the verifier receives them.
        let r: Vec<RingElement> = (0..mu)
            .map(|_| RingElement::random(Representation::IncompleteNTT))
            .collect();

        // Folding consumes the least significant variable first, i.e. the last
        // entry of an MS-first point.
        let fold_at_r = |table: &[RingElement]| {
            let mut sc = LinearSumcheck::new(h);
            sc.load_from(table);
            for x in r.iter().rev() {
                sc.partial_evaluate(x);
            }
            sc.final_evaluations().clone()
        };

        let evals = air_verifier_row_evals(&theta, &eta, &r);
        let (theta_pows, theta_shift) = air_theta_tables(&theta, h);

        assert_eq!(
            fold_at_r(&air_transition_weight_table(&eta)),
            evals.transition_weight,
            "transition weight MLE mismatch"
        );
        assert_eq!(fold_at_r(&theta_pows), evals.theta_pows, "θ̃ MLE mismatch");
        assert_eq!(
            fold_at_r(&theta_shift),
            evals.theta_shift,
            "θshift MLE mismatch"
        );
        assert_eq!(
            fold_at_r(&air_e_first_table(h)),
            evals.e_first,
            "e_first MLE mismatch"
        );
        assert_eq!(
            fold_at_r(&air_e_last_table(h)),
            evals.e_last,
            "e_last MLE mismatch"
        );
    }

    #[test]
    fn test_air_witness_recomposition_and_shift() {
        init();
        let h = 16;
        let x = RingElement::random(Representation::IncompleteNTT);
        let out = air_witness(&x, h);
        let (v0, v1) = air_composed_columns(&out.witness);

        // v0 is the squaring trace
        assert_eq!(v0[0], out.input);
        for i in 1..h {
            assert_eq!(v0[i], &v0[i - 1] * &v0[i - 1], "trace broken at {}", i);
        }
        assert_eq!(v0[h - 1], out.output);
        // v1 is shift(v0): row i takes row i+1, wrapping.
        for i in 0..h {
            assert_eq!(v1[i], v0[(i + 1) % h], "shift broken at {}", i);
        }
        // the transition holds on every row but the last, where the shift wraps
        for i in 0..h - 1 {
            assert_eq!(&v0[i] * &v0[i], v1[i], "transition broken at {}", i);
        }
    }

    #[test]
    fn test_air_crs_geometric() {
        init();
        let h = 16;
        let rank = 2;
        let (crs, aux) = gen_air_crs(h, rank);
        let ck = crs.ck_for_wit_dim(h);
        for r in 0..rank {
            // key[i] = key[0]·gⁱ, checked stepwise through the g⁻¹ the prover
            // actually uses: key[i]·g⁻¹ = key[i-1].
            for i in 1..h {
                let mut expected = ck[r].preprocessed_row[i].clone();
                expected *= &aux.g_inv[r];
                assert_eq!(
                    expected,
                    ck[r].preprocessed_row[i - 1],
                    "commitment key is not a power series"
                );
            }
        }
    }

}
