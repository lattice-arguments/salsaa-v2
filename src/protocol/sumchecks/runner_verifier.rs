use crate::{
    common::config::*,
    protocol::{
        config::{RoundConfig, SalsaaProof},
        sumchecks::context_verifier::VerifierSumcheckContext,
        vdf::{VDFCrs, compute_ip_df_claim},
    },
};
use rokoko::{
    common::{
        arithmetic::{field_to_ring_element_into, precompute_structured_values_fast},
        hash::HashWrapper,
        matrix::{HorizontallyAlignedMatrix, new_vec_zero_preallocated},
        ring_arithmetic::{QuadraticExtension, Representation, RingElement},
        sumcheck_element::SumcheckElement,
    },
    protocol::{
        project_2::BatchedProjectionChallengesSuccinct
    },
};

/// Computes the batched claim from individual sumcheck claims.
/// Type1 sumchecks are product sumchecks with claims = <evaluation_outer, column_claims>,
/// Type3 is a diff sumcheck with claim = 0.
fn batch_claims(
    config: &RoundConfig,
    claims: &HorizontallyAlignedMatrix<RingElement>,
    evaluation_points_outer: &[RingElement],
    ip_l2_claim: Option<&RingElement>,
    ip_linf_claim: Option<&RingElement>,
    ip_df_claim: Option<&RingElement>,
    type31_claims: &[RingElement],
    combination: &[RingElement],
) -> RingElement {
    let mut batched_claim = RingElement::zero(Representation::IncompleteNTT);
    let mut idx = 0;

    // Type1 sumchecks: claim = <evaluation_outer, column_claims[i]>
    for i in 0..config.inner_evaluation_claims {
        let mut type1_claim = RingElement::zero(Representation::IncompleteNTT);
        for (c, r) in claims.row(i).iter().zip(evaluation_points_outer.iter()) {
            type1_claim += &(c * r);
        }
        let mut weighted = type1_claim;
        weighted *= &combination[idx];
        batched_claim += &weighted;
        idx += 1;
    }

    if let RoundConfig::Intermediate { .. } = config {
        // zero claim, nothing to add
        idx += 1;
    }

    // L2: product sumcheck over conjugated witness and selected witness.
    if config.l2 {
        let mut weighted = ip_l2_claim
            .expect("Missing l2 claim in proof while l2 constraint is enabled")
            .clone();
        weighted *= &combination[idx];
        batched_claim += &weighted;
        idx += 1;
    }

    // Linf: exact-binariness sumcheck claim.
    if config.exact_binariness {
        let mut weighted = ip_linf_claim
            .expect("Missing linf claim in proof while exact_binariness is enabled")
            .clone();
        weighted *= &combination[idx];
        batched_claim += &weighted;
        idx += 1;
    }

    // VDF: product sumcheck claim = -y_0 + c^{2K} · y_t
    if config.vdf {
        let mut weighted = ip_df_claim
            .expect("Missing vdf claim in proof while vdf is enabled")
            .clone();
        weighted *= &combination[idx];
        batched_claim += &weighted;
        idx += 1;
    }

    // Type31: projection sumcheck claims for IntermediateUnstructured
    for type31_claim in type31_claims {
        let mut weighted = type31_claim.clone();
        weighted *= &combination[idx];
        batched_claim += &weighted;
        idx += 1;
    }

    assert_eq!(
        idx,
        combination.len(),
        "batch_claims: index mismatch with combination length"
    );
    batched_claim
}

pub fn sumcheck_verifier(
    round_config: &RoundConfig,
    proof: &SalsaaProof,
    verifier_sumcheck_context: &mut VerifierSumcheckContext,
    evaluation_points_outer: &[RingElement],
    vdf_challenge: Option<&RingElement>,
    vdf_outputs: Option<(
        &[RingElement; VDF_MATRIX_HEIGHT],
        &[RingElement; VDF_MATRIX_HEIGHT],
    )>,
    projection_challenges_unstructured: &Option<[BatchedProjectionChallengesSuccinct; NOF_BATCHES]>,
    vdf_crs_param: Option<&VDFCrs>,
    hash_wrapper: &mut HashWrapper,
    claims: &HorizontallyAlignedMatrix<RingElement>,
) -> (
    Vec<RingElement>,
    QuadraticExtension,
    Vec<RingElement>,
    [QuadraticExtension; HALF_DEGREE],
) {
    // Sample random batching coefficients (same Fiat-Shamir as prover)
    let num_sumchecks = verifier_sumcheck_context
        .combiner_evaluation
        .borrow()
        .sumchecks_count();
    let mut combination = new_vec_zero_preallocated(num_sumchecks);
    hash_wrapper.sample_ring_element_vec_into(&mut combination);

    let mut combination_to_field = RingElement::zero(Representation::IncompleteNTT);
    hash_wrapper.sample_ring_element_into(&mut combination_to_field);
    combination_to_field.from_incomplete_ntt_to_homogenized_field_extensions();
    let qe = combination_to_field.split_into_quadratic_extensions();

    // Compute type31 claims for rounds with unstructured projection
    let type31_claims: Vec<RingElement> = match proof {
        SalsaaProof::IntermediateUnstructured {
            projection_image_batched,
            ..
        }
        | SalsaaProof::Last {
            projection_image_batched,
            ..
        } => {
            let challenges = projection_challenges_unstructured
                .as_ref()
                .expect("Missing projection challenges for type31 claims");
            let mut claims_vec = Vec::with_capacity(NOF_BATCHES);
            for batch_idx in 0..NOF_BATCHES {
                let c_2_values =
                    precompute_structured_values_fast(&challenges[batch_idx].c_2_layers);
                let mut claim = RingElement::zero(Representation::IncompleteNTT);
                let mut temp = RingElement::zero(Representation::IncompleteNTT);
                for k in 0..projection_image_batched.width {
                    temp *= (
                        &projection_image_batched[(batch_idx, k)],
                        &RingElement::constant(c_2_values[k], Representation::IncompleteNTT),
                    );
                    claim += &temp;
                }
                claims_vec.push(claim);
            }
            claims_vec
        }
        _ => vec![],
    };

    // Compute expected batched claim over field
    let batched_claim = batch_claims(
        round_config,
        claims,
        &evaluation_points_outer,
        proof.ip_l2_claim.as_ref(),
        proof.ip_linf_claim.as_ref(),
        compute_ip_df_claim(
            round_config,
            vdf_challenge,
            vdf_outputs.map(|(y_0, y_t)| (y_0, y_t, vdf_crs_param.unwrap())),
        )
        .as_ref(),
        &type31_claims,
        &combination,
    );

    let mut batched_claim_over_field = {
        let batched_claim_field = {
            let mut temp = batched_claim.clone();
            temp.from_incomplete_ntt_to_homogenized_field_extensions();
            temp
        };
        let mut temp = batched_claim_field.split_into_quadratic_extensions();
        let mut result = QuadraticExtension::zero();
        for i in 0..HALF_DEGREE {
            temp[i] *= &qe[i];
            result += &temp[i];
        }
        result
    };

    // Verify each sumcheck round: poly(0) + poly(1) == running_claim
    let mut num_vars = proof.sumcheck_transcript.len();

    let mut evaluation_points_field = Vec::new();
    let mut evaluation_points_ring = Vec::new();

    let mut round_idx = 0;
    while num_vars > 0 {
        num_vars -= 1;

        let poly = &proof.sumcheck_transcript[round_idx];

        hash_wrapper.update_with_quadratic_extension_slice(&poly.coefficients);

        assert_eq!(
            poly.at_zero() + poly.at_one(),
            batched_claim_over_field,
            "Sumcheck round {}: poly(0) + poly(1) != running claim",
            round_idx
        );

        let mut f = QuadraticExtension::zero();
        hash_wrapper.sample_field_element_into(&mut f);

        batched_claim_over_field = poly.at(&f);

        evaluation_points_field.push(f);

        let mut r = RingElement::zero(Representation::IncompleteNTT);

        field_to_ring_element_into(&mut r, &f);

        r.from_homogenized_field_extensions_to_incomplete_ntt();

        evaluation_points_ring.push(r);

        round_idx += 1;
    }

    (
        //evaluation_points_field,
        evaluation_points_ring,
        batched_claim_over_field,
        combination,
        qe,
    )
}
