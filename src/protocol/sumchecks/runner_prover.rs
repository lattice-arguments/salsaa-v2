use rokoko::{
    common::{
        arithmetic::field_to_ring_element_into,
        hash::HashWrapper,
        matrix::{HorizontallyAlignedMatrix, VerticallyAlignedMatrix, new_vec_zero_preallocated},
        ring_arithmetic::{QuadraticExtension, Representation, RingElement},
        structured_row::PreprocessedRow,
        sumcheck_element::SumcheckElement,
    },
    protocol::{
        open::evaluation_point_to_structured_row,
        sumcheck_utils::{common::HighOrderSumcheckData, polynomial::Polynomial},
    },
};

use crate::{
    common::config::DEBUG,
    protocol::{config::RoundConfig, sumchecks::context_prover::ProverSumcheckContext},
};

pub fn sumcheck(
    sumcheck_context: &mut ProverSumcheckContext,
    hash_wrapper: &mut HashWrapper,
    witness: &VerticallyAlignedMatrix<RingElement>,
    projected_witness: Option<&VerticallyAlignedMatrix<RingElement>>,
    config: &RoundConfig,
) -> (
    HorizontallyAlignedMatrix<RingElement>,
    Option<Vec<RingElement>>,
    Vec<Polynomial<QuadraticExtension>>,
    Vec<RingElement>,
) {
    let mut num_vars = sumcheck_context.combiner.borrow().variable_count();
    let mut time_poly = std::time::Duration::ZERO;
    let mut time_eval = std::time::Duration::ZERO;
    let mut evaluation_points = Vec::new();
    let mut polys = Vec::new();

    while num_vars > 0 {
        num_vars -= 1;
        let t1 = std::time::Instant::now();
        let mut poly_over_field = Polynomial::<QuadraticExtension>::new(0);
        sumcheck_context
            .field_combiner
            .borrow_mut()
            .univariate_polynomial_into(&mut poly_over_field);
        time_poly += t1.elapsed();

        hash_wrapper.update_with_quadratic_extension_slice(&poly_over_field.coefficients);
        let mut r = RingElement::zero(Representation::IncompleteNTT);
        let mut f = QuadraticExtension::zero();
        hash_wrapper.sample_field_element_into(&mut f);
        field_to_ring_element_into(&mut r, &f);
        r.from_homogenized_field_extensions_to_incomplete_ntt();
        evaluation_points.push(r.clone());

        let t2 = std::time::Instant::now();
        sumcheck_context.partial_evaluate_all(&r);
        time_eval += t2.elapsed();

        polys.push(poly_over_field);
    }

    // Sumcheck rounds produce LS-first challenges; the rest of this prover flow
    // still expects the legacy outer-first (MS-style) view for slicing/splitting.
    evaluation_points.reverse();

    if DEBUG {
        println!(
            "Polynomial time: {:?}, Evaluation time: {:?}",
            time_poly, time_eval
        );
    }

    let outer_points_len =
        config.main_witness_columns.ilog2() as usize + config.main_witness_prefix.length;
    let evaluation_points_inner: Vec<_> = evaluation_points
        .iter()
        .skip(outer_points_len)
        .cloned()
        .collect();

    // Compute per-column claims (and optionally projection claims) over the inner
    // evaluation point, both for the witness and its conjugate.

    let mut preprocessed = PreprocessedRow::from_structured_row(
        &evaluation_point_to_structured_row(&evaluation_points_inner),
    );
    let mut temp = RingElement::zero(Representation::IncompleteNTT);
    let mut claims =
        HorizontallyAlignedMatrix::new_zero_preallocated(2, config.main_witness_columns);
    let mut claim_over_projection = match config {
        RoundConfig::Intermediate { .. } => Some(new_vec_zero_preallocated(2)),
        _ => None,
    };

    for i in 0..config.main_witness_columns {
        for (w, r) in witness
            .col(i)
            .iter()
            .zip(preprocessed.preprocessed_row.iter())
        {
            temp *= (w, r);
            claims[(0, i)] += &temp;
        }
    }

    if let (Some(pw), Some(cop)) = (projected_witness, claim_over_projection.as_mut()) {
        for (c, r) in pw.data.iter().zip(preprocessed.preprocessed_row.iter()) {
            temp *= (c, r);
            cop[0] += &temp;
        }
    }

    // Conjugate eval point in place and repeat the logic to get the claims for the conjugated witness, which will be used in the l2 and linf sumchecks
    for r in preprocessed.preprocessed_row.iter_mut() {
        r.conjugate_in_place();
    }

    for i in 0..witness.width {
        for (w, r) in witness
            .col(i)
            .iter()
            .zip(preprocessed.preprocessed_row.iter())
        {
            temp *= (w, r);
            claims[(1, i)] += &temp;
        }
    }

    if let (Some(pw), Some(cop)) = (projected_witness, claim_over_projection.as_mut()) {
        for (c, r) in pw.data.iter().zip(preprocessed.preprocessed_row.iter()) {
            temp *= (c, r);
            cop[1] += &temp;
        }
    }

    (claims, claim_over_projection, polys, evaluation_points)
}
