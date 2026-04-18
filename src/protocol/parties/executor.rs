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

pub fn execute() {
    println!("===== CONFIG =====");
    println!("Mode: {:?}", mode());
    // if mode() == Mode::VDF {
    //     // println("For VDF we have a witness of
    //     let witness_height = witness_dim() * WITNESS_WIDTH;
    //     println!("VDF witness height: {witness_height}");
    //     println!("VDF steps: {}", witness_height / (VDF_MATRIX_HEIGHT * 64));
    // }

    let (witness, vdf_params) = match mode() {
        Mode::SNARK => {
            panic!("SNARK mode is not implemented yet");
        }
        Mode::VDF => {
            let active_witness_dim = witness_dim();

            let vdf_crs = vdf_init();

            let witness_height = active_witness_dim / WITNESS_WIDTH;

            println!("DF execution...");

            let y_0: [RingElement; VDF_MATRIX_HEIGHT] =
                std::array::from_fn(|_| RingElement::random(Representation::IncompleteNTT));
            let vdf_output = delay_function(&y_0, witness_height, &vdf_crs);
            let witness_trace = vdf_output.trace_witness;
            (witness_trace, Some((y_0, vdf_output.y_t, vdf_crs)))
        }
        Mode::FOLDING_SCHEME => {
            panic!("Folding scheme mode is not implemented yet");
        }
    };

    println!("Generating CRS...");

    let active_rank = rank();
    println!("Using rank={active_rank}");
    let crs = CRS::gen_crs(witness.height, active_rank);

    println!("CRS generated. Starting protocol execution...");

    let mut sumcheck_context = init_prover_sumcheck(&crs, &CONFIG);

    println!("===== COMMITTING WITNESS =====");
    let start = std::time::Instant::now();

    let commitment = commit_basic(&crs, &witness, active_rank);

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
