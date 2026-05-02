use crate::{
    common::config::*,
    protocol::{
        config::{CONFIG, to_kb},
        parties::{prover::prover_round, verifier::verifier_round},
        sumchecks::{
            builder_prover::init_prover_sumcheck, builder_verifier::init_verifier_sumcheck,
        },
        df::{delay_function, df_init},
    },
};

use rokoko::{
    common::{
        hash::HashWrapper,
        matrix::{HorizontallyAlignedMatrix, VerticallyAlignedMatrix},
        ring_arithmetic::{Representation, RingElement},
        sampling::sample_random_short_vector,
    },
    protocol::{commitment::commit_basic, config::SizeableProof, crs::CRS},
};

pub fn witness_sampler(width: usize, height: usize) -> VerticallyAlignedMatrix<RingElement> {
    VerticallyAlignedMatrix {
        height: height,
        width: width,
        data: sample_random_short_vector(
            height * width,
            2u64.pow(9 as u32 - 1), // balanced representation
            Representation::IncompleteNTT,
        ),
        used_cols: width,
    }
}

pub fn execute() {
    println!("===== CONFIG =====");
    println!("Mode: {:?}", mode());
    let active_rank = rank();
    let active_witness_dim = witness_dim();

    let (witness, vdf_params) = match mode() {
        Mode::SNARK | Mode::FOLDING_SCHEME => {
            let witness_height = active_witness_dim / witness_width();

            println!("Witness height: {witness_height}");
            println!("Witness width: {}", witness_width());
            println!("norm_inf: 2^10");
            println!("Using rank={active_rank}");
            println!("=================");

            let witness_sampled = witness_sampler(witness_width(), witness_height);

            // let decomposed_witness = decompose_witness(&witness_sampled);

            (witness_sampled, None)
        }
        Mode::VDF => {
            let witness_height = active_witness_dim / witness_width();
            println!("Witness height: {witness_height}");
            println!("Witness width: {}", witness_width());
            println!("norm: binary");
            println!("Using rank={active_rank}");
            println!(
                "DF steps: {}",
                active_witness_dim / (DF_MATRIX_HEIGHT * DF_BITS)
            );
            println!("=================");

            let df_crs = df_init();

            println!("DF execution...");

            let y_0: [RingElement; DF_MATRIX_HEIGHT] =
                std::array::from_fn(|_| RingElement::random(Representation::IncompleteNTT));
            let df_output = delay_function(&y_0, witness_height, &df_crs);
            let witness_trace = df_output.trace_witness;
            (witness_trace, Some((y_0, df_output.y_t, df_crs)))
        }
    };

    println!("Generating CRS...");

    let crs = CRS::gen_crs(witness.height, active_rank);

    println!("CRS generated. Starting protocol execution...");

    let mut sumcheck_context = init_prover_sumcheck(&crs, &CONFIG);

    println!("===== COMMITTING WITNESS =====");
    let start = std::time::Instant::now();

    let commitment = commit_basic(&crs, &witness, active_rank);

    let commit_duration = start.elapsed().as_nanos();
    println!("TOTAL Commit time: {:?} ns", commit_duration);
    let mut commitment_size_bits = 0;
    for comm in &commitment.data {
        commitment_size_bits += comm.size_in_bits();
    }
    println!("Commitment size: {:.2} KB", to_kb(commitment_size_bits));

    let no_claims = HorizontallyAlignedMatrix {
        height: 0,
        width: 2,
        data: vec![],
    };

    println!("===== STARTING PROVER =====");
    let start = std::time::Instant::now();
    let proof = prover_round(
        &crs,
        &witness,
        &CONFIG,
        &mut sumcheck_context,
        &vec![], // no evaluation points for first round
        &no_claims,
        &mut HashWrapper::new(),
        vdf_params
            .as_ref()
            .map(|(y_0, y_t, vdf_crs)| (y_0, y_t, vdf_crs)),
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
        vdf_params.as_ref().map(|(_, _, vdf_crs)| vdf_crs),
        vdf_params.as_ref().map(|(y_0, y_t, _)| (y_0, y_t)),
        0,
    );
    let verify_duration = start.elapsed().as_nanos();
    println!("TOTAL Verify time: {:?} ns", verify_duration);
    println!("===== VERIFICATION PASSED =====");
}
