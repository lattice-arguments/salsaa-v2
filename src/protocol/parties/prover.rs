use rokoko::{
    common::{
        arithmetic::{ALL_ONE_COEFFS, ONE, ZERO},
        decomposition::{compose_from_decomposed, decompose_chunks_into},
        estimator::{RSISParameters, estimate_rsis_security},
        hash::HashWrapper,
        matrix::{HorizontallyAlignedMatrix, VerticallyAlignedMatrix, new_vec_zero_preallocated},
        norms,
        projection_matrix::ProjectionMatrix,
        ring_arithmetic::{Representation, RingElement},
        structured_row::{PreprocessedRow, StructuredRow},
    },
    protocol::{
        commitment::{commit_basic, commit_basic_internal},
        config::ConfigBase,
        crs::CRS,
        fold::fold,
        open::evaluation_point_to_structured_row,
        project::{prepare_i16_witness, project},
        project_2::{batch_projection_n_times, project_coefficients},
        sumcheck_utils::common::HighOrderSumcheckData,
    },
};

use crate::{
    common::config::*,
    protocol::{
        config::{RoundConfig, SalsaaProof, SalsaaProofCommon, paste_by_prefix},
        project::BatchingChallenges,
        sumchecks::{context_prover::ProverSumcheckContext, runner_prover::sumcheck},
        vdf::{VDFCrs, compute_ip_df_claim},
    },
};

fn structured_round(
    crs: &CRS,
    witness: &VerticallyAlignedMatrix<RingElement>,
    config: &RoundConfig,
    sumcheck_context: &mut ProverSumcheckContext,
    evaluation_points_inner: &Vec<StructuredRow>,
    claims: &HorizontallyAlignedMatrix<RingElement>,
    hash_wrapper: &mut HashWrapper,
    vdf_params: Option<(
        &[RingElement; VDF_MATRIX_HEIGHT],
        &[RingElement; VDF_MATRIX_HEIGHT],
        &VDFCrs,
    )>,
) -> SalsaaProof {
    let RoundConfig::Intermediate {
        decomposition_base_log,
        projection_prefix,
        next,
        split_factor,
        ..
    } = config
    else {
        unreachable!("Expected RoundConfig::Intermediate, got mismatch")
    };
    assert!(
        *split_factor == 1 || *split_factor == 2,
        "split_factor must be 1 or 2"
    );

    let witness_16 = prepare_i16_witness(witness);
    let mut projection_matrix = ProjectionMatrix::new(witness.width, 256);
    projection_matrix.sample(hash_wrapper);
    let mut projected_witness = project(&witness_16, &projection_matrix);
    projected_witness.width = 1;
    projected_witness.used_cols = 1;
    projected_witness.height = witness.height;
    let projection_commitment = commit_basic(crs, &projected_witness, rank());
    let batching_challenges = BatchingChallenges::sample(config, hash_wrapper);

    let vdf_challenge = if config.vdf {
        let mut challenge = RingElement::zero(Representation::IncompleteNTT);
        hash_wrapper.sample_ring_element_ntt_slots_into(&mut challenge);
        Some(challenge)
    } else {
        None
    };

    if DEBUG {
        println!("witness.data.len {:?}", witness.data.len());
    }
    let mut extended_witness =
        new_vec_zero_preallocated(witness.data.len() << config.main_witness_prefix.length);
    let mut witness_conjugated = new_vec_zero_preallocated(witness.data.len());
    for (i, w) in witness.data.iter().enumerate() {
        w.conjugate_into(&mut witness_conjugated[i]);
    }

    let ip_l2_claim = compute_ip_l2_claim(config, witness, &witness_conjugated);
    let ip_linf_claim = compute_ip_linf_claim(config, witness, &witness_conjugated);

    paste_by_prefix(
        &mut extended_witness,
        &witness.data,
        &config.main_witness_prefix,
    );
    paste_by_prefix(
        &mut extended_witness,
        &projected_witness.data,
        projection_prefix,
    );

    let mut evaluation_points_outer = new_vec_zero_preallocated(config.main_witness_columns);
    hash_wrapper.sample_ring_element_vec_into(&mut evaluation_points_outer);

    sumcheck_context.load_data(
        &extended_witness,
        &witness_conjugated,
        evaluation_points_inner,
        &evaluation_points_outer,
        Some(&projection_matrix),
        Some(&batching_challenges),
        None,
        vdf_challenge.as_ref(),
        vdf_params.map(|(_, _, crs)| crs),
    );

    load_combination_challenge(sumcheck_context, hash_wrapper);

    if DEBUG {
        let ip_df_claim = compute_ip_df_claim(config, vdf_challenge.as_ref(), vdf_params);
        if !sumcheck_context.type1sumcheck.is_empty() {
            let claim = sumcheck_context.type1sumcheck[0].output.borrow().claim();
            let mut expected_claim = ZERO.clone();
            for (c, r) in claims.row(0).iter().zip(evaluation_points_outer.iter()) {
                expected_claim += &(c * r);
            }
            assert_eq!(
                claim, expected_claim,
                "Claim from the sumcheck does not match the expected claim computed from the committed witness and the evaluation points"
            );
        }
        let projection_claim = sumcheck_context
            .type3sumcheck
            .as_ref()
            .unwrap()
            .output
            .borrow()
            .claim();
        assert_eq!(
            projection_claim,
            ZERO.clone(),
            "Projection claim does not match"
        );
        debug_assertions_common(
            sumcheck_context,
            config,
            ip_df_claim.as_ref(),
            ip_l2_claim.as_ref(),
            ip_linf_claim.as_ref(),
        );
    }

    let (claims_out, claim_over_projection, polys, evaluation_points) = sumcheck(
        sumcheck_context,
        hash_wrapper,
        witness,
        Some(&projected_witness),
        config,
    );

    let mut folding_challenges = new_vec_zero_preallocated(config.main_witness_columns);
    hash_wrapper.sample_biased_ternary_ring_element_vec_into(&mut folding_challenges);
    let folded_witness = fold(witness, &folding_challenges);

    if DEBUG && *split_factor == 2 {
        let commitment_to_folded_witness = commit_basic(crs, &folded_witness, rank());
        let split_ref = VerticallyAlignedMatrix {
            height: folded_witness.height / 2,
            width: 2,
            data: folded_witness.data.clone(),
            used_cols: 2,
        };
        let commitment_to_split_witness = commit_basic(crs, &split_ref, rank());
        let old_ck = crs.structured_ck_for_wit_dim(split_ref.height * 2);
        let composed = &(&(&*ONE - &old_ck[0].tensor_layers[0])
            * &commitment_to_split_witness[(0, 0)])
            + &(&old_ck[0].tensor_layers[0] * &commitment_to_split_witness[(0, 1)]);
        assert_eq!(
            composed,
            commitment_to_folded_witness[(0, 0)],
            "Composed commitment from the split witness does not match the commitment to the folded witness"
        );
    }

    let split_witness = VerticallyAlignedMatrix {
        height: folded_witness.height / *split_factor,
        width: *split_factor,
        data: folded_witness.data,
        used_cols: *split_factor,
    };
    // width = split_factor (witness cols) * 2 (decomp) + split_factor (proj cols) * 2 (decomp)
    let decomposed_width = split_factor * 4;
    let mut decomposed_split_witness = VerticallyAlignedMatrix {
        height: split_witness.height,
        width: decomposed_width,
        data: new_vec_zero_preallocated(split_witness.height * decomposed_width),
        used_cols: decomposed_width,
    };

    if DEBUG_HARDNESS {
        println!("Estimating hardness against RSIS attack for the structured round...");
        let norm = norms::l2_norm(&split_witness.data);
        let norm_composed = norm * (1f64 + 2f64.powi(*decomposition_base_log as i32));

        let norm_unfolded = norm_composed * 8f64 * DEGREE as f64; // 2 for sis break, 2 for challenges in numerator, 2 for denominator

        let sec: Result<rokoko::common::estimator::EstimatorResult, std::io::Error> =
            estimate_rsis_security(&RSISParameters {
                m: config.witness_height() as u64,
                n: rank() as u64,
                length_bound: norm_unfolded as u64,
            });

        let sec_par = sec.unwrap().secpar;

        assert!(
            sec_par >= EXPECTED_SEC_PARAM as f64,
            "Estimated security against RSIS attack for the structured round is below the expected security parameter: {} bits, expected at least {} bits, increase RANK by setting --rank=<new_rank> in CLI (current rank: {})",
            sec_par,
            EXPECTED_SEC_PARAM,
            rank()
        );

        println!(
            "Estimated security against RSIS attack for structured round: {} bits",
            sec_par
        );
    }

    let h = split_witness.height;
    for i in 0..*split_factor {
        decompose_chunks_into(
            &mut decomposed_split_witness.data[i * 2 * h..(i + 1) * 2 * h],
            &split_witness.data[i * h..(i + 1) * h],
            *decomposition_base_log,
            2,
        );
    }
    let proj_offset = *split_factor * 2 * h;
    for i in 0..*split_factor {
        decompose_chunks_into(
            &mut decomposed_split_witness.data
                [proj_offset + i * 2 * h..proj_offset + (i + 1) * 2 * h],
            &projected_witness.data[i * h..(i + 1) * h],
            *decomposition_base_log,
            2,
        );
    }
    let decomposed_split_commitment = commit_basic(crs, &decomposed_split_witness, rank());

    if DEBUG && *split_factor == 2 {
        debug_check_decomposed(
            crs,
            &split_witness,
            &decomposed_split_commitment,
            &projection_commitment,
            *decomposition_base_log as usize,
        );
    }
    let outer_points_len =
        config.main_witness_columns.ilog2() as usize + config.main_witness_prefix.length;
    let split_bits = split_factor.ilog2() as usize;

    let new_evaluation_points_inner: Vec<_> = evaluation_points
        .iter()
        .skip(outer_points_len + split_bits)
        .cloned()
        .collect();

    let new_evaluation_points_inner_expanded = PreprocessedRow::from_structured_row(
        &evaluation_point_to_structured_row(&new_evaluation_points_inner),
    );
    let new_evaluation_points_inner_conjugated: Vec<_> = new_evaluation_points_inner
        .iter()
        .map(RingElement::conjugate)
        .collect();
    let new_evaluation_points_inner_conjugated_expanded = PreprocessedRow::from_structured_row(
        &evaluation_point_to_structured_row(&new_evaluation_points_inner_conjugated),
    );
    let new_claims = commit_basic_internal(
        &vec![
            new_evaluation_points_inner_expanded,
            new_evaluation_points_inner_conjugated_expanded,
        ],
        &decomposed_split_witness,
        2,
    );
    let next_level_eval_points = vec![
        evaluation_point_to_structured_row(&new_evaluation_points_inner),
        evaluation_point_to_structured_row(&new_evaluation_points_inner_conjugated),
    ];
    let next_level_proof = match next.as_ref() {
        Some(next_config) => Some(Box::new(prover_round(
            crs,
            &decomposed_split_witness,
            next_config,
            sumcheck_context.next.as_mut().unwrap(),
            &next_level_eval_points,
            &new_claims,
            hash_wrapper,
            None,
        ))),
        None => None,
    };

    if DEBUG && next_level_proof.is_none() {
        println!(
            "No recursion to next round since this is the last round of the folding scheme config. Skipping checks for the next round's evaluation points and claims."
        );
        println!(
            "Next round witness height and width (after folding, splitting, and decomposition) would have been: height={}, width={}",
            split_witness.height, decomposed_split_witness.width
        );
    }

    let common = SalsaaProofCommon {
        ip_l2_claim,
        ip_linf_claim,
        sumcheck_transcript: polys,
        claims: claims_out,
    };
    SalsaaProof::Intermediate {
        common,
        new_claims,
        decomposed_split_commitment,
        projection_commitment,
        claim_over_projection: claim_over_projection.unwrap(),
        next: next_level_proof,
    }
}

fn unstructured_round(
    crs: &CRS,
    witness: &VerticallyAlignedMatrix<RingElement>,
    config: &RoundConfig,
    sumcheck_context: &mut ProverSumcheckContext,
    evaluation_points_inner: &Vec<StructuredRow>,
    claims: &HorizontallyAlignedMatrix<RingElement>,
    hash_wrapper: &mut HashWrapper,
) -> SalsaaProof {
    let RoundConfig::IntermediateUnstructured {
        decomposition_base_log,
        projection_ratio,
        next,
        split_factor,
        ..
    } = config
    else {
        unreachable!("Expected RoundConfig::IntermediateUnstructured, got mismatch")
    };
    assert!(
        *split_factor == 1 || *split_factor == 2,
        "split_factor must be 1 or 2"
    );

    let mut projection_matrix = ProjectionMatrix::new(*projection_ratio, PROJECTION_HEIGHT);
    projection_matrix.sample(hash_wrapper);
    let projection_ct = project_coefficients(witness, &projection_matrix);
    let (batched_image, unstructured_batching_challenges) = batch_projection_n_times(
        witness,
        &projection_matrix,
        hash_wrapper,
        NOF_BATCHES,
        false,
    );

    let vdf_challenge = if config.vdf {
        let mut challenge = RingElement::zero(Representation::IncompleteNTT);
        hash_wrapper.sample_ring_element_ntt_slots_into(&mut challenge);
        Some(challenge)
    } else {
        None
    };

    if DEBUG {
        println!("witness.data.len {:?}", witness.data.len());
    }

    let mut extended_witness =
        new_vec_zero_preallocated(witness.data.len() << config.main_witness_prefix.length);
    let mut witness_conjugated = new_vec_zero_preallocated(witness.data.len());
    for (i, w) in witness.data.iter().enumerate() {
        w.conjugate_into(&mut witness_conjugated[i]);
    }

    let ip_l2_claim = compute_ip_l2_claim(config, witness, &witness_conjugated);
    let ip_linf_claim = compute_ip_linf_claim(config, witness, &witness_conjugated);

    paste_by_prefix(
        &mut extended_witness,
        &witness.data,
        &config.main_witness_prefix,
    );

    let mut evaluation_points_outer = new_vec_zero_preallocated(config.main_witness_columns);
    hash_wrapper.sample_ring_element_vec_into(&mut evaluation_points_outer);

    sumcheck_context.load_data(
        &extended_witness,
        &witness_conjugated,
        evaluation_points_inner,
        &evaluation_points_outer,
        None,
        None,
        Some(&unstructured_batching_challenges),
        vdf_challenge.as_ref(),
        None,
    );

    load_combination_challenge(sumcheck_context, hash_wrapper);

    if DEBUG {
        if !sumcheck_context.type1sumcheck.is_empty() {
            let claim = sumcheck_context.type1sumcheck[0].output.borrow().claim();
            let mut expected_claim = ZERO.clone();
            for (c, r) in claims.row(0).iter().zip(evaluation_points_outer.iter()) {
                expected_claim += &(c * r);
            }
            assert_eq!(
                claim, expected_claim,
                "Claim from the sumcheck does not match the expected claim computed from the committed witness and the evaluation points"
            );
        }
        for (batch_idx, type31) in sumcheck_context
            .type31sumchecks
            .as_ref()
            .unwrap()
            .iter()
            .enumerate()
        {
            let projection_claim = type31.output.borrow().claim();
            let batch_image = &batched_image.row(batch_idx);
            let challenges = &unstructured_batching_challenges[batch_idx].c_2_values;
            let mut expected_projection_claim = RingElement::zero(Representation::IncompleteNTT);
            let mut temp = RingElement::zero(Representation::IncompleteNTT);
            for (c, r) in batch_image.iter().zip(challenges.iter()) {
                temp *= (c, &RingElement::constant(*r, Representation::IncompleteNTT));
                expected_projection_claim += &temp;
            }
            assert_eq!(
                projection_claim, expected_projection_claim,
                "Projection claim from the sumcheck does not match the expected projection claim"
            );
        }
        println!(
            "Unstructured projection claims from the sumcheck match the expected projection claims"
        );
        debug_assertions_common(
            sumcheck_context,
            config,
            None,
            ip_l2_claim.as_ref(),
            ip_linf_claim.as_ref(),
        );
    }

    let (claims_out, _, polys, evaluation_points) =
        sumcheck(sumcheck_context, hash_wrapper, witness, None, config);

    let mut folding_challenges = new_vec_zero_preallocated(config.main_witness_columns);
    hash_wrapper.sample_biased_ternary_ring_element_vec_into(&mut folding_challenges);
    let folded_witness = fold(witness, &folding_challenges);

    let split_witness = VerticallyAlignedMatrix {
        height: folded_witness.height / *split_factor,
        width: *split_factor,
        data: folded_witness.data,
        used_cols: *split_factor,
    };
    let decomposed_width = split_factor * 2;
    let mut decomposed_split_witness = VerticallyAlignedMatrix {
        height: split_witness.height,
        width: decomposed_width,
        data: new_vec_zero_preallocated(split_witness.height * decomposed_width),
        used_cols: decomposed_width,
    };
    let h = split_witness.height;
    for i in 0..*split_factor {
        decompose_chunks_into(
            &mut decomposed_split_witness.data[i * 2 * h..(i + 1) * 2 * h],
            &split_witness.data[i * h..(i + 1) * h],
            *decomposition_base_log,
            2,
        );
    }
    let decomposed_split_commitment = commit_basic(crs, &decomposed_split_witness, rank());

    if DEBUG_HARDNESS {
        let norm = norms::l2_norm(&decomposed_split_witness.data);
        let norm_composed = norm * (1f64 + 2f64.powi(*decomposition_base_log as i32));

        let norm_unfolded = norm_composed * 8f64 * DEGREE as f64; // 2 for sis break, 2 for challenges in numerator, 2 for denominator

        let sec = estimate_rsis_security(&RSISParameters {
            m: config.witness_height() as u64,
            n: RANK as u64,
            length_bound: norm_unfolded as u64,
        });
        let sec_par = sec.unwrap().secpar;

        assert!(
            sec_par >= EXPECTED_SEC_PARAM as f64,
            "Estimated security against RSIS attack for the structured round is below the expected security parameter: {} bits, expected at least {} bits, increase RANK by setting --rank=<new_rank> in CLI (current rank: {})",
            sec_par,
            EXPECTED_SEC_PARAM,
            rank()
        );

        println!(
            "Estimated security against RSIS attack for structured round: {} bits",
            sec_par
        );
    }

    let outer_points_len =
        config.main_witness_columns.ilog2() as usize + config.main_witness_prefix.length;
    let split_bits = split_factor.ilog2() as usize;

    let new_evaluation_points_inner: Vec<_> = evaluation_points
        .iter()
        .skip(outer_points_len + split_bits)
        .cloned()
        .collect();
    let new_evaluation_points_inner_expanded = PreprocessedRow::from_structured_row(
        &evaluation_point_to_structured_row(&new_evaluation_points_inner),
    );
    let new_evaluation_points_inner_conjugated: Vec<_> = new_evaluation_points_inner
        .iter()
        .map(RingElement::conjugate)
        .collect();
    let new_evaluation_points_inner_conjugated_expanded = PreprocessedRow::from_structured_row(
        &evaluation_point_to_structured_row(&new_evaluation_points_inner_conjugated),
    );
    let new_claims = commit_basic_internal(
        &vec![
            new_evaluation_points_inner_expanded,
            new_evaluation_points_inner_conjugated_expanded,
        ],
        &decomposed_split_witness,
        2,
    );
    let next_level_eval_points = vec![
        evaluation_point_to_structured_row(&new_evaluation_points_inner),
        evaluation_point_to_structured_row(&new_evaluation_points_inner_conjugated),
    ];
    let next_level_proof = prover_round(
        crs,
        &decomposed_split_witness,
        next,
        sumcheck_context.next.as_mut().unwrap(),
        &next_level_eval_points,
        &new_claims,
        hash_wrapper,
        None,
    );

    let common = SalsaaProofCommon {
        ip_l2_claim,
        ip_linf_claim,
        sumcheck_transcript: polys,
        claims: claims_out,
    };
    SalsaaProof::IntermediateUnstructured {
        common,
        new_claims: new_claims.data,
        decomposed_split_commitment,
        next: Box::new(next_level_proof),
        projection_image_ct: projection_ct,
        projection_image_batched: batched_image,
    }
}

fn last_round(
    witness: &VerticallyAlignedMatrix<RingElement>,
    config: &RoundConfig, // must be RoundConfig::Last
    sumcheck_context: &mut ProverSumcheckContext,
    evaluation_points_inner: &Vec<StructuredRow>,
    claims: &HorizontallyAlignedMatrix<RingElement>,
    hash_wrapper: &mut HashWrapper,
) -> SalsaaProof {
    let RoundConfig::Last {
        projection_ratio, ..
    } = config
    else {
        unreachable!("Expected RoundConfig::Last, got mismatch")
    };

    println!(
        "Using unstructured projection with ratio {}",
        projection_ratio
    );
    println!(
        "Sampling projection matrix for unstructured projection with ratio {}",
        projection_ratio
    );
    let mut projection_matrix = ProjectionMatrix::new(*projection_ratio, PROJECTION_HEIGHT);
    projection_matrix.sample(hash_wrapper);
    let projection_ct = project_coefficients(witness, &projection_matrix);
    let (batched_image, unstructured_batching_challenges) = batch_projection_n_times(
        witness,
        &projection_matrix,
        hash_wrapper,
        NOF_BATCHES,
        false,
    );

    let vdf_challenge = if config.vdf {
        let mut challenge = RingElement::zero(Representation::IncompleteNTT);
        hash_wrapper.sample_ring_element_ntt_slots_into(&mut challenge);
        Some(challenge)
    } else {
        None
    };

    if DEBUG {
        println!("witness.data.len {:?}", witness.data.len());
    }
    let mut extended_witness =
        new_vec_zero_preallocated(witness.data.len() << config.main_witness_prefix.length);
    let mut witness_conjugated = new_vec_zero_preallocated(witness.data.len());
    for (i, w) in witness.data.iter().enumerate() {
        w.conjugate_into(&mut witness_conjugated[i]);
    }

    let ip_l2_claim = compute_ip_l2_claim(config, witness, &witness_conjugated);
    let ip_linf_claim = compute_ip_linf_claim(config, witness, &witness_conjugated);

    paste_by_prefix(
        &mut extended_witness,
        &witness.data,
        &config.main_witness_prefix,
    );

    let mut evaluation_points_outer = new_vec_zero_preallocated(config.main_witness_columns);
    hash_wrapper.sample_ring_element_vec_into(&mut evaluation_points_outer);

    sumcheck_context.load_data(
        &extended_witness,
        &witness_conjugated,
        evaluation_points_inner,
        &evaluation_points_outer,
        None,
        None,
        Some(&unstructured_batching_challenges),
        vdf_challenge.as_ref(),
        None,
    );

    load_combination_challenge(sumcheck_context, hash_wrapper);

    if DEBUG {
        if !sumcheck_context.type1sumcheck.is_empty() {
            let claim = sumcheck_context.type1sumcheck[0].output.borrow().claim();
            let mut expected_claim = ZERO.clone();
            for (c, r) in claims.row(0).iter().zip(evaluation_points_outer.iter()) {
                expected_claim += &(c * r);
            }
            assert_eq!(
                claim, expected_claim,
                "Claim from the sumcheck does not match the expected claim computed from the committed witness and the evaluation points"
            );
        }
        debug_assertions_common(
            sumcheck_context,
            config,
            None,
            ip_l2_claim.as_ref(),
            ip_linf_claim.as_ref(),
        );
    }

    let (claims_out, _, polys, _) = sumcheck(sumcheck_context, hash_wrapper, witness, None, config);

    let mut folding_challenges = new_vec_zero_preallocated(config.main_witness_columns);
    hash_wrapper.sample_biased_ternary_ring_element_vec_into(&mut folding_challenges);
    let folded_witness = fold(witness, &folding_challenges);

    let common = SalsaaProofCommon {
        ip_l2_claim,
        ip_linf_claim,
        sumcheck_transcript: polys,
        claims: claims_out,
    };
    SalsaaProof::Last {
        common,
        folded_witness: folded_witness.data,
        projection_image_ct: projection_ct,
        projection_image_batched: batched_image,
    }
}

pub fn prover_round(
    crs: &CRS,
    witness: &VerticallyAlignedMatrix<RingElement>,
    config: &RoundConfig,
    sumcheck_context: &mut ProverSumcheckContext,
    evaluation_points_inner: &Vec<StructuredRow>,
    claims: &HorizontallyAlignedMatrix<RingElement>,
    hash_wrapper: &mut HashWrapper,
    vdf_params: Option<(
        &[RingElement; VDF_MATRIX_HEIGHT],
        &[RingElement; VDF_MATRIX_HEIGHT],
        &VDFCrs,
    )>,
) -> SalsaaProof {
    match config {
        RoundConfig::Intermediate { .. } => structured_round(
            crs,
            witness,
            config,
            sumcheck_context,
            evaluation_points_inner,
            claims,
            hash_wrapper,
            vdf_params,
        ),
        RoundConfig::IntermediateUnstructured { .. } => unstructured_round(
            crs,
            witness,
            config,
            sumcheck_context,
            evaluation_points_inner,
            claims,
            hash_wrapper,
        ),
        RoundConfig::Last { .. } => last_round(
            witness,
            config,
            sumcheck_context,
            evaluation_points_inner,
            claims,
            hash_wrapper,
        ),
    }
}

fn compute_ip_l2_claim(
    config: &RoundConfig,
    witness: &VerticallyAlignedMatrix<RingElement>,
    witness_conjugated: &[RingElement],
) -> Option<RingElement> {
    if !config.l2 {
        return None;
    }
    let mut temp = RingElement::zero(Representation::IncompleteNTT);
    let mut claim = RingElement::zero(Representation::IncompleteNTT);
    for (w, wc) in witness.data.iter().zip(witness_conjugated.iter()) {
        temp *= (w, wc);
        claim += &temp;
    }
    Some(claim)
}

fn compute_ip_linf_claim(
    config: &RoundConfig,
    witness: &VerticallyAlignedMatrix<RingElement>,
    witness_conjugated: &[RingElement],
) -> Option<RingElement> {
    if !config.exact_binariness {
        return None;
    }
    let mut temp = RingElement::zero(Representation::IncompleteNTT);
    let mut claim = RingElement::zero(Representation::IncompleteNTT);
    for (w, wc) in witness.data.iter().zip(witness_conjugated.iter()) {
        temp -= (&*ALL_ONE_COEFFS, w);
        temp *= wc;
        claim += &temp;
    }
    Some(claim)
}

fn load_combination_challenge(
    sumcheck_context: &mut ProverSumcheckContext,
    hash_wrapper: &mut HashWrapper,
) {
    let num_sumchecks = sumcheck_context.combiner.borrow().sumchecks_count();
    let mut combination = new_vec_zero_preallocated(num_sumchecks);
    hash_wrapper.sample_ring_element_vec_into(&mut combination);
    sumcheck_context
        .combiner
        .borrow_mut()
        .load_challenges_from(&combination);

    let mut combination_to_field = RingElement::zero(Representation::IncompleteNTT);
    hash_wrapper.sample_ring_element_into(&mut combination_to_field);
    combination_to_field.from_incomplete_ntt_to_homogenized_field_extensions();
    let qe = combination_to_field.split_into_quadratic_extensions();
    sumcheck_context
        .field_combiner
        .borrow_mut()
        .load_challenges_from(qe);
}

fn debug_assertions_common(
    sumcheck_context: &ProverSumcheckContext,
    config: &RoundConfig,
    ip_df_claim: Option<&RingElement>,
    ip_l2_claim: Option<&RingElement>,
    ip_linf_claim: Option<&RingElement>,
) {
    if config.l2 {
        let l2_claim = sumcheck_context
            .l2sumcheck
            .as_ref()
            .unwrap()
            .output
            .borrow()
            .claim();

        let ip_l2_claim = ip_l2_claim.expect("exepcted ip_l2_claim in debug function, got None");
        assert_eq!(
            l2_claim, *ip_l2_claim,
            "L2 claim from the projection sumcheck does not match the expected l2 claim computed from the witness"
        );
    }
    if config.exact_binariness {
        let linf_claim = sumcheck_context
            .linfsumcheck
            .as_ref()
            .unwrap()
            .output
            .borrow()
            .claim();
        let ct = linf_claim.constant_term_from_incomplete_ntt();
        assert_eq!(
            ct, 0,
            "Linf claim from the projection sumcheck is not zero, which means that the witness is not exactly binary as expected"
        );

        let ip_linf_claim =
            ip_linf_claim.expect("exepcted ip_linf_claim in debug function, got None");
        assert_eq!(
            linf_claim, *ip_linf_claim,
            "Linf claim from the projection sumcheck does not match the expected linf claim computed from the witness"
        );
    }
    if config.vdf {
        let vdf_claim = sumcheck_context
            .vdfsumcheck
            .as_ref()
            .unwrap()
            .output
            .borrow()
            .claim();
        assert_eq!(
            vdf_claim,
            *ip_df_claim.unwrap(),
            "VDF claim from the sumcheck does not match the expected VDF claim"
        );
    }
}

fn debug_check_decomposed(
    crs: &CRS,
    split_witness: &VerticallyAlignedMatrix<RingElement>,
    decomposed_split_commitment: &HorizontallyAlignedMatrix<RingElement>,
    projection_commitment: &HorizontallyAlignedMatrix<RingElement>,
    decomposition_base_log: usize,
) {
    let commitment_to_split_witness = commit_basic(crs, split_witness, rank());
    let old_ck = crs.structured_ck_for_wit_dim(split_witness.height * 2);
    let composed = compose_from_decomposed(
        &[
            decomposed_split_commitment[(0, 0)].clone(),
            decomposed_split_commitment[(0, 1)].clone(),
            decomposed_split_commitment[(0, 2)].clone(),
            decomposed_split_commitment[(0, 3)].clone(),
        ],
        decomposition_base_log as u64,
        2,
    );
    assert_eq!(
        composed[0],
        commitment_to_split_witness[(0, 0)],
        "Composed commitment from the decomposed split witness does not match the commitment to the split witness"
    );
    assert_eq!(
        composed[1],
        commitment_to_split_witness[(0, 1)],
        "Composed commitment from the decomposed split projected witness does not match the commitment to the projected witness"
    );

    let composed_projection = compose_from_decomposed(
        &[
            decomposed_split_commitment[(0, 4)].clone(),
            decomposed_split_commitment[(0, 5)].clone(),
            decomposed_split_commitment[(0, 6)].clone(),
            decomposed_split_commitment[(0, 7)].clone(),
        ],
        decomposition_base_log as u64,
        2,
    );
    let unsplit_projection = &(&(&*ONE - &old_ck[0].tensor_layers[0]) * &composed_projection[0])
        + &(&old_ck[0].tensor_layers[0] * &composed_projection[1]);
    assert_eq!(
        unsplit_projection,
        projection_commitment[(0, 0)],
        "Composed commitment from the decomposed split projected witness does not match the commitment to the projected witness"
    );
}
