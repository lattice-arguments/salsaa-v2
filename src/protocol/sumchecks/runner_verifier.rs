use rokoko::{
    common::{
        arithmetic::field_to_ring_element_into,
        hash::HashWrapper,
        matrix::{new_vec_zero_preallocated, HorizontallyAlignedMatrix, VerticallyAlignedMatrix},
        ring_arithmetic::{QuadraticExtension, Representation, RingElement},
        structured_row::PreprocessedRow,
        sumcheck_element::SumcheckElement,
    }
};
use crate::{
    protocol::{
        config::RoundConfig,
        open::evaluation_point_to_structured_row,
        sumcheck_utils::{common::HighOrderSumcheckData, polynomial::Polynomial},
        sumchecks::context::VerifierSumcheckContext,
    },
};

fn sumcheck_verifier(
    config: &SumcheckConfig,
    verifier_sumcheck_context: &mut VerifierSumcheckContext,
    transcript: &[SumcheckElement],
    hash_wrapper: &mut HashWrapper,
    mut running_claim: QuadraticExtension,
) -> (
    Vec<QuadraticExtension>,
    Vec<RingElement>,
    QuadraticExtension,
) {

    // Sample random batching coefficients (same Fiat-Shamir as prover)
    let num_sumchecks = verifier_context
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
        config,
        claims,
        &evaluation_points_outer,
        proof.ip_l2_claim.as_ref(),
        proof.ip_linf_claim.as_ref(),
        compute_ip_vdf_claim(
            config,
            vdf_challenge.as_ref(),
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
    let mut evaluation_points_field = Vec::new();
    let mut evaluation_points_ring = Vec::new();

    let mut num_vars = transcript.len();
    let mut round_idx = 0;

    while num_vars > 0 {
        num_vars -= 1;

        let poly = &transcript[round_idx];

        hash_wrapper.update_with_quadratic_extension_slice(
            &poly.coefficients
        );

        assert_eq!(
            poly.at_zero() + poly.at_one(),
            running_claim,
            "Sumcheck round {} failed",
            round_idx
        );

        let mut f = QuadraticExtension::zero();
        hash_wrapper.sample_field_element_into(&mut f);

        running_claim = poly.at(&f);

        evaluation_points_field.push(f);

        let mut r = RingElement::zero(
            Representation::IncompleteNTT
        );

        field_to_ring_element_into(&mut r, &f);

        r.from_homogenized_field_extensions_to_incomplete_ntt();

        evaluation_points_ring.push(r);

        round_idx += 1;
    }

    (
        evaluation_points_field,
        evaluation_points_ring,
        running_claim,
    )
}
