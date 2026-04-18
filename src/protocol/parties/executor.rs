use crate::{
    common::config::*,
    protocol::{
        config::{CONFIG, to_kb},
        parties::{prover::prover_round, verifier::verifier_round},
        sumchecks::{
            builder_prover::init_prover_sumcheck, builder_verifier::init_verifier_sumcheck,
        },
        vdf::{delay_function, vdf_init},
    },
};

use rokoko::{
    common::{
        hash::HashWrapper,
        matrix::{HorizontallyAlignedMatrix, VerticallyAlignedMatrix},
        ring_arithmetic::{Representation, RingElement},
    },
    protocol::{commitment::commit_basic, config::SizeableProof, crs::CRS},
};

pub struct VDFOutput {
    y_int: [RingElement; VDF_MATRIX_HEIGHT], // TODO: this y_int is not needed but let's keep it for now
    y_t: [RingElement; VDF_MATRIX_HEIGHT],
    trace_witness: VerticallyAlignedMatrix<RingElement>,
}
fn sample_random_binary_vector(len: usize) -> Vec<RingElement> {
    (0..len)
        .map(|_| RingElement::random_bounded_unsigned(Representation::IncompleteNTT, 2))
        .collect()
}

pub fn binary_witness_sampler() -> VerticallyAlignedMatrix<RingElement> {
    VerticallyAlignedMatrix {
        height: WITNESS_DIM,
        width: WITNESS_WIDTH,
        data: sample_random_binary_vector(WITNESS_DIM * WITNESS_WIDTH),
        // data: vec![RingElement::all(0, Representation::IncompleteNTT); WITNESS_DIM * WITNESS_WIDTH],
        used_cols: WITNESS_WIDTH,
    }
}

pub fn execute() {
    println!("Generating CRS...");

    let active_rank = rank();
    println!("Using rank={active_rank}");
    let crs = CRS::gen_crs(WITNESS_DIM, active_rank);
    let vdf_crs = vdf_init();

    println!("CRS generated. Starting execution...");
    let y_0: [RingElement; VDF_MATRIX_HEIGHT] =
        std::array::from_fn(|_| RingElement::random(Representation::IncompleteNTT)); // TODO: from hash
    let vdf_output = delay_function(&y_0, WITNESS_DIM, &vdf_crs);

    let mut sumcheck_context = init_prover_sumcheck(&crs, &CONFIG);

    println!("===== COMMITTING WITNESS =====");
    let start = std::time::Instant::now();

    let commitment = commit_basic(&crs, &vdf_output.trace_witness, active_rank);

    let commit_duration = start.elapsed().as_nanos();
    println!("TOTAL Commit time: {:?} ns", commit_duration);

    let no_claims = HorizontallyAlignedMatrix {
        height: 0,
        width: 2,
        data: vec![],
    };

    println!("===== STARTING PROVER =====");
    let start = std::time::Instant::now();
    let proof = prover_round(
        &crs,
        &vdf_output.trace_witness,
        &CONFIG,
        &mut sumcheck_context,
        &vec![], // no evaluation points for first round
        &no_claims,
        &mut HashWrapper::new(),
        Some((&y_0, &vdf_output.y_t, &vdf_crs)),
    );
    let prove_duration = start.elapsed().as_millis();
    println!("TOTAL Prove time: {:?} ms", prove_duration);

    println!("===== PROOF SIZE =====");
    let proof_size_bits = proof.size_in_bits();
    println!("Total proof size: {:.2} KB", to_kb(proof_size_bits));

    println!("===== STARTING VERIFIER =====");
    let start = std::time::Instant::now();
    let mut verifier_context = init_verifier_sumcheck(&CONFIG);
    verifier_round(
        &CONFIG,
        &crs,
        &mut verifier_context,
        &commitment,
        &proof,
        &[],        // no evaluation points for first round
        &no_claims, // no claims for first round
        &mut HashWrapper::new(),
        Some(&vdf_crs),
        Some((&y_0, &vdf_output.y_t)),
        0,
    );
    let verify_duration = start.elapsed().as_nanos();
    println!("TOTAL Verify time: {:?} ns", verify_duration);
    println!("===== VERIFICATION PASSED =====");
}
