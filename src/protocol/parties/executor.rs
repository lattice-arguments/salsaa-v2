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
        arithmetic::ZERO, decomposition::decompose, hash::HashWrapper, matrix::{HorizontallyAlignedMatrix, VerticallyAlignedMatrix}, ring_arithmetic::{Representation, RingElement}, sampling::sample_random_short_vector
    },
    protocol::{commitment::commit_basic, config::SizeableProof, crs::CRS, open::evaluation_point_to_structured_row},
};

pub struct VDFOutput {
    y_int: [RingElement; VDF_MATRIX_HEIGHT], // TODO: this y_int is not needed but let's keep it for now
    y_t: [RingElement; VDF_MATRIX_HEIGHT],
    trace_witness: VerticallyAlignedMatrix<RingElement>,
}
pub fn witness_sampler(
    width: usize,
    height: usize,
) -> VerticallyAlignedMatrix<RingElement> {
    VerticallyAlignedMatrix {
        height: height,
        width: width,
        data: sample_random_short_vector(
            height * width,
            2u64.pow(10 as u32 - 1), // balanced representation
            Representation::IncompleteNTT,
        ),
        used_cols: width,
    }
}

// pub fn decompose_witness(
//     witness: &VerticallyAlignedMatrix<RingElement>,
// ) -> VerticallyAlignedMatrix<RingElement> {
//     let decomposed_data = decompose(
//         &witness.data,
//         4,
//         2,
//     );
//     VerticallyAlignedMatrix {
//         height: witness.height * 2,
//         width: witness.width,
//         data: decomposed_data,
//         used_cols: witness.width,
//     }
// }


pub fn execute() {
    println!("===== CONFIG =====");
    println!("Mode: {:?}", mode());
    let active_rank = rank();
    let active_witness_dim = witness_dim();

    let (witness, vdf_params) = match mode() {
        Mode::SNARK => {
            let witness_height = active_witness_dim / WITNESS_WIDTH;
            
            println!("Witness height: {witness_height}");
            println!("Witness width: {}", WITNESS_WIDTH);
            println!("norm_inf: 2^10");
            println!("Using rank={active_rank}");
            println!("=================");

            let witness_sampled = witness_sampler(WITNESS_WIDTH, witness_height);

            // let decomposed_witness = decompose_witness(&witness_sampled);

            (witness_sampled, None)
        }
        Mode::VDF => {

            let witness_height = active_witness_dim / WITNESS_WIDTH;
            println!("Witness height: {witness_height}");
            println!("Witness width: {}", WITNESS_WIDTH);
            println!("norm: binary");
            println!("Using rank={active_rank}");
            println!(
                "VDF steps: {}",
                active_witness_dim / (VDF_MATRIX_HEIGHT * VDF_BITS)
            );
            println!("=================");

            let vdf_crs = vdf_init();

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
