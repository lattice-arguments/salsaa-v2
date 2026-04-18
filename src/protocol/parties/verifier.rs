use crate::{
    common::config::*,
    protocol::{
        config::{RoundConfig, SalsaaProof},
        project::BatchingChallenges,
        sumchecks::{
            context_verifier::VerifierSumcheckContext, runner_verifier::sumcheck_verifier,
        },
        vdf::VDFCrs,
    },
};

use rokoko::{
    common::{
        arithmetic::ONE,
        decomposition::compose_from_decomposed,
        hash::HashWrapper,
        matrix::{HorizontallyAlignedMatrix, VerticallyAlignedMatrix, new_vec_zero_preallocated},
        projection_matrix::ProjectionMatrix,
        ring_arithmetic::{QuadraticExtension, Representation, RingElement},
        structured_row::{PreprocessedRow, StructuredRow},
    },
    protocol::{
        commitment::{BasicCommitment, commit_basic},
        crs::CRS,
        open::evaluation_point_to_structured_row,
        project_2::{BatchedProjectionChallengesSuccinct, verifier_sample_projection_challenges},
    },
};

pub struct VerifierRoundState<'a> {
    pub config: &'a RoundConfig,
    pub crs: &'a CRS,
    pub verifier_context: &'a mut VerifierSumcheckContext,
    pub commitment: &'a BasicCommitment,
    pub proof: &'a SalsaaProof,
    pub evaluation_points_inner: &'a [StructuredRow],
    pub claims: &'a HorizontallyAlignedMatrix<RingElement>,
    pub hash_wrapper: &'a mut HashWrapper,
    pub vdf_crs_param: Option<&'a VDFCrs>,
    pub vdf_outputs: Option<(
        &'a [RingElement; VDF_MATRIX_HEIGHT],
        &'a [RingElement; VDF_MATRIX_HEIGHT],
    )>,
    pub round_index: usize,

    pub round_start: std::time::Instant,

    pub projection_matrix: Option<ProjectionMatrix>,
    pub batching_challenges: Option<BatchingChallenges>,
    pub projection_challenges_unstructured:
        Option<[BatchedProjectionChallengesSuccinct; NOF_BATCHES]>,

    pub vdf_challenge: Option<RingElement>,

    pub evaluation_points_outer: Vec<RingElement>,
    pub evaluation_points_ring_tree: Vec<RingElement>,

    pub folding_challenges: Vec<RingElement>,
    pub outer_points_len: usize,
    pub layer: RingElement,
    pub conj_layer: RingElement,

    pub evaluation_points_ring: Vec<RingElement>,
    pub batched_claim_over_field: QuadraticExtension,
    pub combination: Vec<RingElement>,
    pub qe: [QuadraticExtension; HALF_DEGREE],

    pub folded_claim: RingElement,
    pub folded_conj_claim: RingElement,
}

impl<'a> VerifierRoundState<'a> {
    pub fn new(
        config: &'a RoundConfig,
        crs: &'a CRS,
        verifier_context: &'a mut VerifierSumcheckContext,
        commitment: &'a BasicCommitment,
        proof: &'a SalsaaProof,
        evaluation_points_inner: &'a [StructuredRow],
        claims: &'a HorizontallyAlignedMatrix<RingElement>,
        hash_wrapper: &'a mut HashWrapper,
        vdf_crs_param: Option<&'a VDFCrs>,
        vdf_outputs: Option<(
            &'a [RingElement; VDF_MATRIX_HEIGHT],
            &'a [RingElement; VDF_MATRIX_HEIGHT],
        )>,
        round_index: usize,
    ) -> Self {
        let round_start = std::time::Instant::now();

        // projection matrix
        let projection_matrix = match config {
            RoundConfig::Intermediate { .. } => {
                let mut pm = ProjectionMatrix::new(config.main_witness_columns, PROJECTION_HEIGHT);
                pm.sample(hash_wrapper);
                Some(pm)
            }
            _ => None,
        };

        // batching
        let batching_challenges = match config {
            RoundConfig::Intermediate { .. } => {
                Some(BatchingChallenges::sample(config, hash_wrapper))
            }
            _ => None,
        };

        // unstructured projection challenges
        let projection_challenges_unstructured = match config {
            RoundConfig::IntermediateUnstructured {
                projection_ratio, ..
            }
            | RoundConfig::Last {
                projection_ratio, ..
            } => {
                let mut projection_matrix =
                    ProjectionMatrix::new(*projection_ratio, PROJECTION_HEIGHT);
                projection_matrix.sample(hash_wrapper);

                Some(std::array::from_fn(|_| {
                    verifier_sample_projection_challenges(&projection_matrix, config, hash_wrapper)
                }))
            }
            _ => None,
        };

        // VDF
        let vdf_challenge = if config.vdf {
            let mut challenge = RingElement::zero(Representation::IncompleteNTT);
            hash_wrapper.sample_ring_element_ntt_slots_into(&mut challenge);
            Some(challenge)
        } else {
            None
        };

        // outer evaluation points
        let mut evaluation_points_outer = new_vec_zero_preallocated(config.main_witness_columns);
        hash_wrapper.sample_ring_element_vec_into(&mut evaluation_points_outer);

        // sumcheck
        let (evaluation_points_ring, batched_claim_over_field, combination, qe) = sumcheck_verifier(
            config,
            proof,
            verifier_context,
            &evaluation_points_outer,
            vdf_challenge.as_ref(),
            vdf_outputs,
            &projection_challenges_unstructured,
            vdf_crs_param,
            hash_wrapper,
            claims,
        );

        // folding challenges
        let mut folding_challenges = new_vec_zero_preallocated(config.main_witness_columns);
        hash_wrapper.sample_biased_ternary_ring_element_vec_into(&mut folding_challenges);

        let outer_points_len =
            config.main_witness_columns.ilog2() as usize + config.main_witness_prefix.length;

        let evaluation_points_ring_tree = evaluation_points_ring
            .iter()
            .rev()
            .cloned()
            .collect::<Vec<_>>();

        let layer = evaluation_points_ring_tree[outer_points_len].clone();
        let conj_layer = layer.conjugate();

        // folded claims
        let mut folded_claim = RingElement::zero(Representation::IncompleteNTT);
        for i in 0..config.main_witness_columns {
            let mut term = folding_challenges[i].clone();
            term *= &proof.claims[(0, i)];
            folded_claim += &term;
        }

        let mut folded_conj_claim = RingElement::zero(Representation::IncompleteNTT);
        for i in 0..config.main_witness_columns {
            let mut term = folding_challenges[i].clone();
            term *= &proof.claims[(1, i)];
            folded_conj_claim += &term;
        }

        Self {
            config,
            crs,
            verifier_context,
            commitment,
            proof,
            evaluation_points_inner,
            claims,
            hash_wrapper,
            vdf_crs_param,
            vdf_outputs,
            round_index,
            round_start,
            projection_matrix,
            batching_challenges,
            projection_challenges_unstructured,
            vdf_challenge,
            evaluation_points_outer,
            evaluation_points_ring,
            evaluation_points_ring_tree,
            folding_challenges,
            outer_points_len,
            layer,
            conj_layer,
            batched_claim_over_field,
            combination,
            qe,
            folded_claim,
            folded_conj_claim,
        }
    }
}

pub fn verifier_round(
    config: &RoundConfig,
    crs: &CRS,
    verifier_context: &mut VerifierSumcheckContext,
    commitment: &BasicCommitment,
    proof: &SalsaaProof,
    evaluation_points_inner: &[StructuredRow],
    claims: &HorizontallyAlignedMatrix<RingElement>,
    hash_wrapper: &mut HashWrapper,
    vdf_crs_param: Option<&VDFCrs>,
    vdf_outputs: Option<(
        &[RingElement; VDF_MATRIX_HEIGHT],
        &[RingElement; VDF_MATRIX_HEIGHT],
    )>,
    round_index: usize,
) {
    let state = VerifierRoundState::new(
        config,
        crs,
        verifier_context,
        commitment,
        proof,
        evaluation_points_inner,
        claims,
        hash_wrapper,
        vdf_crs_param,
        vdf_outputs,
        round_index,
    );

    match (config, proof) {
        (
            RoundConfig::Intermediate {
                decomposition_base_log,
                next,
                ..
            },
            SalsaaProof::Intermediate {
                new_claims,
                decomposed_split_commitment,
                claim_over_projection,
                projection_commitment,
                next: next_proof,
                ..
            },
        ) => structured_round(
            state,
            *decomposition_base_log,
            next.as_ref().unwrap(),
            new_claims,
            decomposed_split_commitment,
            claim_over_projection,
            projection_commitment,
            next_proof,
        ),

        (
            RoundConfig::IntermediateUnstructured {
                decomposition_base_log,
                next,
                ..
            },
            SalsaaProof::IntermediateUnstructured {
                new_claims,
                decomposed_split_commitment,
                next: next_proof,
                ..
            },
        ) => unstructured_round(
            state,
            *decomposition_base_log,
            next,
            new_claims,
            decomposed_split_commitment,
            next_proof,
        ),

        (RoundConfig::Last { .. }, SalsaaProof::Last { folded_witness, .. }) => {
            last_round(state, folded_witness.clone())
        }

        _ => panic!("Config and proof mismatch"),
    }
}

fn structured_round(
    state: VerifierRoundState,
    decomposition_base_log: u64,
    next: &RoundConfig,
    new_claims: &HorizontallyAlignedMatrix<RingElement>,
    decomposed_split_commitment: &BasicCommitment,
    claim_over_projection: &Vec<RingElement>,
    projection_commitment: &BasicCommitment,
    next_proof: &SalsaaProof,
) {
    let recomposed_claims = HorizontallyAlignedMatrix {
        height: 2,
        width: 4,
        data: compose_from_decomposed(&new_claims.data, decomposition_base_log, 2),
    };

    assert_eq!(
        state.folded_claim,
        &(&(&*ONE - &state.layer) * &recomposed_claims[(0, 0)])
            + &(&state.layer * &recomposed_claims[(0, 1)]),
        "Recomposed claim for the witness does not match the original claim"
    );

    assert_eq!(
        state.folded_conj_claim,
        &(&(&*ONE - &state.conj_layer) * &recomposed_claims[(1, 0)])
            + &(&state.conj_layer * &recomposed_claims[(1, 1)]),
        "Recomposed conjugate claim for the witness does not match the original claim"
    );

    // Check claims over the projection
    assert_eq!(
        claim_over_projection[0],
        &(&(&*ONE - &state.layer) * &recomposed_claims[(0, 2)])
            + &(&state.layer * &recomposed_claims[(0, 3)]),
        "Recomposed claim for the projection does not match the original claim"
    );

    assert_eq!(
        claim_over_projection[1],
        &(&(&*ONE - &state.conj_layer) * &recomposed_claims[(1, 2)])
            + &(&state.conj_layer * &recomposed_claims[(1, 3)]),
        "Recomposed conjugate claim for the projection does not match the original claim"
    );

    let recomposed_commitments = HorizontallyAlignedMatrix {
        height: rank(),
        width: 4,
        data: compose_from_decomposed(
            &decomposed_split_commitment.data,
            decomposition_base_log as u64,
            2,
        ),
    };

    let mut temp = RingElement::zero(Representation::IncompleteNTT);
    for r in 0..rank() {
        let layer = state.crs.structured_ck_for_wit_dim(
            state.config.extended_witness_length / 2 / state.config.main_witness_columns,
        )[r]
            .tensor_layers
            .first()
            .unwrap();

        let mut folded_commitment_r = RingElement::zero(Representation::IncompleteNTT);
        for i in 0..state.config.main_witness_columns {
            temp *= (&state.folding_challenges[i], &state.commitment[(r, i)]);
            folded_commitment_r += &temp;
        }

        assert_eq!(
            folded_commitment_r,
            &(&(&*ONE - layer) * &recomposed_commitments[(r, 0)])
                + &(layer * &recomposed_commitments[(r, 1)]),
            "Recomposed commitment for the witness does not match the folded commitment"
        );

        assert_eq!(
            projection_commitment[(r, 0)],
            &(&(&*ONE - layer) * &recomposed_commitments[(r, 2)])
                + &(layer * &recomposed_commitments[(r, 3)]),
            "Recomposed commitment for the projection does not match"
        );
    }

    state.verifier_context.load_data(
        state.config,
        state.proof,
        &state.evaluation_points_ring_tree,
        state.evaluation_points_inner,
        &state.evaluation_points_outer,
        &state.batching_challenges,
        &state.projection_matrix,
        &state.projection_challenges_unstructured,
        &state.combination,
        state.qe,
        state.vdf_challenge.as_ref(),
        state.vdf_crs_param,
    );

    let verifier_eval = *state
        .verifier_context
        .field_combiner_evaluation
        .borrow_mut()
        .evaluate_at_ring_point(&state.evaluation_points_ring);

    assert_eq!(
        verifier_eval, state.batched_claim_over_field,
        "Verifier final check failed: tree evaluation does not match sumcheck claim"
    );

    // Recurse into the next round
    let new_evaluation_points_inner = state
        .evaluation_points_ring_tree
        .iter()
        .skip(state.outer_points_len + 1)
        .cloned()
        .collect::<Vec<_>>();

    let new_evaluation_points_inner_conjugated = new_evaluation_points_inner
        .iter()
        .map(RingElement::conjugate)
        .collect::<Vec<_>>();

    let next_level_eval_points = vec![
        evaluation_point_to_structured_row(&new_evaluation_points_inner),
        evaluation_point_to_structured_row(&new_evaluation_points_inner_conjugated),
    ];

    println!(
        "Verifier round {} took {:?}",
        state.round_index,
        state.round_start.elapsed()
    );

    verifier_round(
        next,
        state.crs,
        state.verifier_context.next.as_mut().unwrap(),
        decomposed_split_commitment,
        next_proof,
        &next_level_eval_points,
        new_claims,
        state.hash_wrapper,
        state.vdf_crs_param,
        state.vdf_outputs,
        state.round_index + 1,
    );
}

fn last_round(state: VerifierRoundState, folded_witness: Vec<RingElement>) {
    // Last round: verify claims directly from the witness data

    // Reconstruct the folded witness as a VerticallyAlignedMatrix (1 column)
    let folded_witness_matrix = VerticallyAlignedMatrix {
        height: folded_witness.len(),
        width: 1,
        data: folded_witness.clone(),
        used_cols: 1,
    };

    // Reconstruct projected witness (1 column, same height as folded)
    // let projected_witness_matrix = VerticallyAlignedMatrix {
    //     height: projected_witness.len(),
    //     width: 1,
    //     data: projected_witness.clone(),
    //     used_cols: 1,
    // };

    // Use the current round's sumcheck evaluation points, including the "layer" variable
    // (no +1 skip since there's no split at the last round).
    // The prover computes claims using evaluation_points[outer_points_len..] from THIS round's sumcheck.
    let current_inner_points: Vec<_> = state
        .evaluation_points_ring_tree
        .iter()
        .skip(state.outer_points_len)
        .cloned()
        .collect();

    let eval_points_inner_expanded = PreprocessedRow::from_structured_row(
        &evaluation_point_to_structured_row(&current_inner_points),
    );

    let current_inner_points_conjugated: Vec<_> = current_inner_points
        .iter()
        .map(RingElement::conjugate)
        .collect();

    let eval_points_inner_conj_expanded = PreprocessedRow::from_structured_row(
        &evaluation_point_to_structured_row(&current_inner_points_conjugated),
    );

    // Compute expected claim over folded witness: <eval_points, folded_witness>
    let mut temp = RingElement::zero(Representation::IncompleteNTT);
    let mut expected_folded_claim = RingElement::zero(Representation::IncompleteNTT);
    for (w, r) in folded_witness
        .iter()
        .zip(eval_points_inner_expanded.preprocessed_row.iter())
    {
        temp *= (w, r);
        expected_folded_claim += &temp;
    }

    assert_eq!(
        state.folded_claim, expected_folded_claim,
        "Last round: folded claim does not match evaluation of the folded witness"
    );

    // Compute expected conjugate claim over folded witness
    let mut expected_folded_conj_claim = RingElement::zero(Representation::IncompleteNTT);
    for (w, r) in folded_witness
        .iter()
        .zip(eval_points_inner_conj_expanded.preprocessed_row.iter())
    {
        temp *= (w, r);
        expected_folded_conj_claim += &temp;
    }

    assert_eq!(
        state.folded_conj_claim, expected_folded_conj_claim,
        "Last round: folded conjugate claim does not match evaluation of the folded witness"
    );

    // // Compute expected claim over projected witness
    // let mut expected_projection_claim = RingElement::zero(Representation::IncompleteNTT);
    // for (w, r) in projected_witness.iter().zip(eval_points_inner_expanded.preprocessed_row.iter()) {
    //     temp *= (w, r);
    //     expected_projection_claim += &temp;
    // }

    // assert_eq!(
    //     proof.claim_over_projection[0], expected_projection_claim,
    //     "Last round: projection claim does not match evaluation of the projected witness"
    // );

    // // Compute expected conjugate claim over projected witness
    // let mut expected_projection_conj_claim = RingElement::zero(Representation::IncompleteNTT);
    // for (w, r) in projected_witness.iter().zip(eval_points_inner_conj_expanded.preprocessed_row.iter()) {
    //     temp *= (w, r);
    //     expected_projection_conj_claim += &temp;
    // }

    // assert_eq!(
    //     proof.claim_over_projection[1], expected_projection_conj_claim,
    //     "Last round: conjugate projection claim does not match evaluation of the projected witness"
    // );

    let comm_time = std::time::Instant::now();

    // Verify commitment: commit(folded_witness) should match folded commitments
    let folded_witness_commitment = commit_basic(state.crs, &folded_witness_matrix, rank());
    // let projected_witness_commitment = commit_basic(crs, &projected_witness_matrix, RANK);

    let elapsed = comm_time.elapsed();
    println!(
        "Verifier commitment recomputation took {} µs",
        elapsed.as_micros()
    );

    for r in 0..rank() {
        let mut folded_commitment_r = RingElement::zero(Representation::IncompleteNTT);
        for i in 0..state.config.main_witness_columns {
            temp *= (&state.folding_challenges[i], &state.commitment[(r, i)]);
            folded_commitment_r += &temp;
        }

        assert_eq!(
            folded_commitment_r,
            folded_witness_commitment[(r, 0)],
            "Last round: folded witness commitment does not match"
        );

        // assert_eq!(
        //     proof.projection_commitment[(r, 0)], projected_witness_commitment[(r, 0)],
        //     "Last round: projected witness commitment does not match"
        // );
    }

    state.verifier_context.load_data(
        state.config,
        state.proof,
        &state.evaluation_points_ring_tree,
        state.evaluation_points_inner,
        &state.evaluation_points_outer,
        &state.batching_challenges,
        &state.projection_matrix,
        &state.projection_challenges_unstructured,
        &state.combination,
        state.qe,
        state.vdf_challenge.as_ref(),
        state.vdf_crs_param,
    );

    let verifier_eval = *state
        .verifier_context
        .field_combiner_evaluation
        .borrow_mut()
        .evaluate_at_ring_point(&state.evaluation_points_ring);

    assert_eq!(
        verifier_eval, state.batched_claim_over_field,
        "Verifier final check failed: tree evaluation does not match sumcheck claim"
    );

    println!(
        "Verifier round {} (last) took {:?}",
        state.round_index,
        state.round_start.elapsed()
    );
    // No recursion at the last round
}

fn unstructured_round(
    state: VerifierRoundState,
    decomposition_base_log: u64,
    next: &RoundConfig,
    new_claims: &Vec<RingElement>,
    decomposed_split_commitment: &BasicCommitment,
    next_proof: &SalsaaProof,
) {
    // Recompose claims: width=2 (no projection columns)
    let recomposed_claims = HorizontallyAlignedMatrix {
        height: 2,
        width: 2,
        data: compose_from_decomposed(new_claims, decomposition_base_log, 2),
    };

    assert_eq!(
        state.folded_claim,
        &(&(&*ONE - &state.layer) * &recomposed_claims[(0, 0)])
            + &(&state.layer * &recomposed_claims[(0, 1)]),
        "IntermediateUnstructured: recomposed claim does not match the folded claim"
    );

    assert_eq!(
        state.folded_conj_claim,
        &(&(&*ONE - &state.conj_layer) * &recomposed_claims[(1, 0)])
            + &(&state.conj_layer * &recomposed_claims[(1, 1)]),
        "IntermediateUnstructured: recomposed conjugate claim does not match"
    );

    // Recompose commitments: width=2 (no projection)
    let recomposed_commitments = HorizontallyAlignedMatrix {
        height: rank(),
        width: 2,
        data: compose_from_decomposed(&decomposed_split_commitment.data, decomposition_base_log, 2),
    };

    let mut temp = RingElement::zero(Representation::IncompleteNTT);
    for r in 0..rank() {
        let layer = state.crs.structured_ck_for_wit_dim(
            (state.config.extended_witness_length >> state.config.main_witness_prefix.length)
                / state.config.main_witness_columns,
        )[r]
            .tensor_layers
            .first()
            .unwrap();

        let mut folded_commitment_r = RingElement::zero(Representation::IncompleteNTT);
        for i in 0..state.config.main_witness_columns {
            temp *= (&state.folding_challenges[i], &state.commitment[(r, i)]);
            folded_commitment_r += &temp;
        }

        assert_eq!(
            folded_commitment_r,
            &(&(&*ONE - layer) * &recomposed_commitments[(r, 0)])
                + &(layer * &recomposed_commitments[(r, 1)]),
            "IntermediateUnstructured: recomposed commitment does not match"
        );
    }

    state.verifier_context.load_data(
        state.config,
        state.proof,
        &state.evaluation_points_ring_tree,
        state.evaluation_points_inner,
        &state.evaluation_points_outer,
        &state.batching_challenges,
        &state.projection_matrix,
        &state.projection_challenges_unstructured,
        &state.combination,
        state.qe,
        state.vdf_challenge.as_ref(),
        state.vdf_crs_param,
    );

    let verifier_eval = *state
        .verifier_context
        .field_combiner_evaluation
        .borrow_mut()
        .evaluate_at_ring_point(&state.evaluation_points_ring);

    assert_eq!(
        verifier_eval, state.batched_claim_over_field,
        "IntermediateUnstructured: tree evaluation does not match sumcheck claim"
    );

    // Recurse into the next round
    let new_evaluation_points_inner = state
        .evaluation_points_ring_tree
        .iter()
        .skip(state.outer_points_len + 1)
        .cloned()
        .collect::<Vec<_>>();

    let new_evaluation_points_inner_conjugated = new_evaluation_points_inner
        .iter()
        .map(RingElement::conjugate)
        .collect::<Vec<_>>();

    let next_level_eval_points = vec![
        evaluation_point_to_structured_row(&new_evaluation_points_inner),
        evaluation_point_to_structured_row(&new_evaluation_points_inner_conjugated),
    ];

    let recomposed_new_claims = HorizontallyAlignedMatrix {
        height: 2,
        width: next.main_witness_columns,
        data: new_claims.clone(),
    };

    println!(
        "Verifier round {} (unstructured) took {:?}",
        state.round_index,
        state.round_start.elapsed()
    );

    verifier_round(
        next,
        state.crs,
        state.verifier_context.next.as_mut().unwrap(),
        decomposed_split_commitment,
        next_proof,
        &next_level_eval_points,
        &recomposed_new_claims,
        state.hash_wrapper,
        state.vdf_crs_param,
        state.vdf_outputs,
        state.round_index + 1,
    );
}
