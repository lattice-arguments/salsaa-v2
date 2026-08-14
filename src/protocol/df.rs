use crate::{common::config::*, protocol::config::RoundConfig};
use rokoko::common::{
    matrix::{HorizontallyAlignedMatrix, VerticallyAlignedMatrix},
    ring_arithmetic::{Representation, RingElement},
};

pub struct DFCrs {
    pub data: HorizontallyAlignedMatrix<RingElement>,
}
pub struct DFOutput {
    pub y_int: [RingElement; DF_MATRIX_HEIGHT],
    pub y_t: [RingElement; DF_MATRIX_HEIGHT],
    pub trace_witness: VerticallyAlignedMatrix<RingElement>,
}
pub fn delay_function(
    y_0: &[RingElement; DF_MATRIX_HEIGHT],
    dim: usize,
    df_crs: &DFCrs,
) -> DFOutput {
    // Delay Function with G = I_{HEIGHT} ⊗ g^T (gadget) and A (HEIGHT × WIDTH CRS matrix).
    //
    // Per step:
    //   w_step = G^{-1}(-y_step)   — decompose each component of y_step into DF_BITS binary planes
    //   y_{step+1} = A · w_step    — full matrix-vector product giving HEIGHT outputs
    //
    // The witness is split into two columns (matching vertical memory alignment).
    // y_int is the intermediate value at the column boundary.

    let mut trace_witness = VerticallyAlignedMatrix {
        height: dim,
        width: 2,
        data: vec![RingElement::zero(Representation::IncompleteNTT); dim * 2],
        used_cols: 2,
    };

    let steps_per_col = dim / DF_MATRIX_WIDTH;
    let total_steps = steps_per_col * 2;

    let mut neg_y: [RingElement; DF_MATRIX_HEIGHT] = std::array::from_fn(|r| y_0[r].negate());
    let mut y_int: [RingElement; DF_MATRIX_HEIGHT] =
        std::array::from_fn(|_| RingElement::zero(Representation::IncompleteNTT));
    let mut temp = RingElement::zero(Representation::IncompleteNTT);

    println!("Executing delay function with {} steps", total_steps);
    // y_{step+1} = A · w_step: full matrix-vector product
    let mut y_next: [RingElement; DF_MATRIX_HEIGHT] =
        std::array::from_fn(|_| RingElement::zero(Representation::IncompleteNTT));
    let df_start = std::time::Instant::now();
    for step in 0..total_steps {
        let col = step >> steps_per_col.trailing_zeros();
        let row_in_col = step & (steps_per_col - 1); // equivalent to % if divisor is power of 2
        let base_row = row_in_col * DF_MATRIX_WIDTH;

        // w_step = G^{-1}(-y_step): decompose each component into DF_BITS binary planes
        let data_offset = col * dim + base_row;
        for r in 0..DF_MATRIX_HEIGHT {
            decompose_binary_into(
                &neg_y[r],
                &mut trace_witness.data
                    [data_offset + r * DF_BITS..data_offset + (r + 1) * DF_BITS],
            );
        }

        for r in 0..DF_MATRIX_HEIGHT {
            for j in 0..DF_MATRIX_WIDTH {
                temp *= (&df_crs.data[(r, j)], &trace_witness.data[data_offset + j]);
                if j == 0 {
                    y_next[r].set_from(&temp);
                } else {
                    y_next[r] += &temp;
                }
            }
        }

        if step == steps_per_col - 1 {
            y_int = y_next.clone();
        }

        neg_y = std::array::from_fn(|r| y_next[r].negate());
    }
    let df_duration = df_start.elapsed().as_micros();
    println!("Delay function executed in {:?} µs", df_duration);
    println!(
        "Avg step time: {:?} µs",
        df_duration as f64 / (total_steps as f64)
    );

    let y_t: [RingElement; DF_MATRIX_HEIGHT] = std::array::from_fn(|r| neg_y[r].negate());

    DFOutput {
        y_int,
        y_t,
        trace_witness,
    }
}

/// Computes ip_df_claim = Σ_r c^r·(-y_0[r]) + c^{DF_STRIDE·2K+r}·y_t[r] from the DF challenge and outputs.
pub fn compute_ip_df_claim(
    config: &RoundConfig,
    df_challenge: Option<&RingElement>,
    df_params: Option<(
        &[RingElement; DF_MATRIX_HEIGHT],
        &[RingElement; DF_MATRIX_HEIGHT],
        &DFCrs,
    )>,
) -> Option<RingElement> {
    if !config.vdf {
        return None;
    }
    let c = df_challenge.expect("DF enabled but no challenge");
    let (y_0, y_t, _) = df_params.expect("DF enabled but no params");
    let two_k = config.extended_witness_length / 2 / DF_MATRIX_WIDTH;

    // Compute c^{DF_STRIDE * 2K}
    let mut c_stride = RingElement::constant(1, Representation::IncompleteNTT);
    for _ in 0..DF_STRIDE {
        c_stride *= c;
    }
    let mut c_stride_2k = RingElement::constant(1, Representation::IncompleteNTT);
    for _ in 0..two_k {
        c_stride_2k *= &c_stride;
    }

    // claim = Σ_r c^r · (-y_0[r]) + Σ_r c^{DF_STRIDE·2K + r} · y_t[r]
    let mut claim = RingElement::zero(Representation::IncompleteNTT);
    let mut c_power = RingElement::constant(1, Representation::IncompleteNTT); // c^r
    let mut temp = RingElement::zero(Representation::IncompleteNTT);
    for r in 0..DF_MATRIX_HEIGHT {
        // -c^r · y_0[r]
        temp *= (&c_power, &y_0[r]);
        claim -= &temp;
        // c^{DF_STRIDE·2K + r} · y_t[r]
        temp *= (&c_stride_2k, &c_power); // temp = c^{DF_STRIDE*2K + r}
        temp *= &y_t[r];
        claim += &temp;
        c_power *= c;
    }
    Some(claim)
}

pub fn df_init() -> DFCrs {
    println!("Initializing DF CRS...");
    let data = HorizontallyAlignedMatrix {
        height: DF_MATRIX_HEIGHT,
        width: DF_MATRIX_WIDTH,
        data: (0..DF_MATRIX_HEIGHT * DF_MATRIX_WIDTH)
            .map(|_| RingElement::random(Representation::IncompleteNTT))
            .collect(),
    };
    DFCrs { data }
}

/// Decomposes a RingElement into 64 bit-plane RingElements, writing into `target`.
/// target\[b\].v\[j\] = (element.v\[j\] >> b) & 1 for each coefficient j and bit b.
/// The input is assumed to be in IncompleteNTT; we convert to EvenOddCoefficients
/// to access raw coefficients, decompose, then convert each result back.

pub fn decompose_binary_into(element: &RingElement, target: &mut [RingElement]) {
    assert!(
        target.len() >= 64,
        "target slice must have at least 64 elements"
    );

    let mut tmp = element.clone();
    tmp.from_incomplete_ntt_to_even_odd_coefficients();

    for bit_elem in target[..64].iter_mut() {
        *bit_elem = RingElement::zero(Representation::EvenOddCoefficients);
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
    {
        use std::arch::x86_64::*;
        unsafe {
            let one = _mm512_set1_epi64(1);
            // Process 8 coefficients at a time
            for chunk_start in (0..DEGREE).step_by(8) {
                let coeffs = _mm512_loadu_epi64(tmp.v[chunk_start..].as_ptr() as *const i64);
                for b in 0..64u64 {
                    let shift_amt = _mm512_set1_epi64(b as i64);
                    let shifted = _mm512_srlv_epi64(coeffs, shift_amt);
                    let masked = _mm512_and_epi64(shifted, one);
                    _mm512_storeu_epi64(
                        target[b as usize].v[chunk_start..].as_mut_ptr() as *mut i64,
                        masked,
                    );
                }
            }
        }
    }

    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
    {
        for j in 0..DEGREE {
            let val = tmp.v[j];
            for b in 0..64usize {
                target[b].v[j] = (val >> b) & 1;
            }
        }
    }

    for bit_elem in target[..64].iter_mut() {
        bit_elem.from_even_odd_coefficients_to_incomplete_ntt_representation();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::config::MOD_Q;

    #[test]
    fn test_decompose_binary_roundtrip() {
        let elem = RingElement::random(Representation::IncompleteNTT);
        let mut bits: Vec<RingElement> = (0..64)
            .map(|_| RingElement::zero(Representation::IncompleteNTT))
            .collect();
        decompose_binary_into(&elem, &mut bits);

        assert_eq!(bits.len(), 64);

        // Recompose: sum_b bits[b] * 2^b  (in EvenOdd space, then convert back)
        let mut recomposed = RingElement::zero(Representation::IncompleteNTT);
        recomposed.from_incomplete_ntt_to_even_odd_coefficients();
        for (b, bit_elem) in bits.iter().enumerate() {
            let mut bit_copy = bit_elem.clone();
            bit_copy.from_incomplete_ntt_to_even_odd_coefficients();
            let shift = 1u64 << b;
            for j in 0..DEGREE {
                recomposed.v[j] = (recomposed.v[j] + bit_copy.v[j] * shift) % MOD_Q;
            }
        }
        recomposed.from_even_odd_coefficients_to_incomplete_ntt_representation();

        assert_eq!(recomposed, elem, "Binary decomposition roundtrip failed");
    }

    #[test]
    fn test_decompose_binary_bits_are_binary() {
        let elem = RingElement::random(Representation::IncompleteNTT);
        let mut bits: Vec<RingElement> = (0..64)
            .map(|_| RingElement::zero(Representation::IncompleteNTT))
            .collect();
        decompose_binary_into(&elem, &mut bits);

        for (b, bit_elem) in bits.iter().enumerate() {
            let mut bit_copy = bit_elem.clone();
            bit_copy.from_incomplete_ntt_to_even_odd_coefficients();
            for j in 0..DEGREE {
                assert!(
                    bit_copy.v[j] == 0 || bit_copy.v[j] == 1,
                    "Bit plane {} coeff {} is {}, expected 0 or 1",
                    b,
                    j,
                    bit_copy.v[j]
                );
            }
        }
    }

    #[test]
    fn test_decompose_binary_high_bits_zero() {
        // MOD_Q < 2^51, so bits 51..63 should be all zero
        let elem = RingElement::random(Representation::IncompleteNTT);
        let mut bits: Vec<RingElement> = (0..64)
            .map(|_| RingElement::zero(Representation::IncompleteNTT))
            .collect();
        decompose_binary_into(&elem, &mut bits);

        for b in 51..64 {
            let mut bit_copy = bits[b].clone();
            bit_copy.from_incomplete_ntt_to_even_odd_coefficients();
            for j in 0..DEGREE {
                assert_eq!(
                    bit_copy.v[j], 0,
                    "Bit plane {} coeff {} should be 0 (above modulus bit-width)",
                    b, j
                );
            }
        }
    }

    /// Verify the matrix equation from execute_df:
    ///
    /// | G       |    | w_0 w_K |     | -y_0   -y_int |
    /// | A G     |    | w_1 ... |     |   0      0    |
    /// |   A G   |  * | ...     |  =  |   0      0    |
    /// |     A G |    | ...     |     |   0      0    |
    /// |       A |    |---------|     |  y_int  y_t   |
    ///
    /// where K = steps_per_col and G = I_{HEIGHT} ⊗ g^T recomposes
    /// each component independently via sum_j 2^j * bit_j.
    #[test]
    fn test_df_matrix_equation() {
        let test_dim: usize = 1 << 12; // 4096, giving steps_per_col = 4096 / 128 = 32
        let y_0: [RingElement; DF_MATRIX_HEIGHT] =
            std::array::from_fn(|_| RingElement::random(Representation::IncompleteNTT));
        let df_crs = df_init();
        let df_output = delay_function(&y_0, test_dim, &df_crs);

        let steps_per_col = test_dim / DF_MATRIX_WIDTH;
        let w = &df_output.trace_witness;

        // Helper: compute G * w_block where G = I_{HEIGHT} ⊗ g^T.
        // Component r recomposes DF_BITS bits starting at offset r * DF_BITS.
        let recompose = |base_row: usize, col: usize| -> [RingElement; DF_MATRIX_HEIGHT] {
            std::array::from_fn(|r| {
                let mut result = RingElement::zero(Representation::IncompleteNTT);
                result.from_incomplete_ntt_to_even_odd_coefficients();
                for j in 0..DF_BITS {
                    let mut bit_copy = w[(base_row + r * DF_BITS + j, col)].clone();
                    bit_copy.from_incomplete_ntt_to_even_odd_coefficients();
                    let shift = 1u64 << j;
                    for k in 0..DEGREE {
                        result.v[k] = (result.v[k] + bit_copy.v[k] * shift) % MOD_Q;
                    }
                }
                result.from_even_odd_coefficients_to_incomplete_ntt_representation();
                result
            })
        };

        // Helper: compute A * w_block where A is HEIGHT × WIDTH.
        // Returns one ring element per row of A.
        let inner_product_a = |base_row: usize, col: usize| -> [RingElement; DF_MATRIX_HEIGHT] {
            std::array::from_fn(|r| {
                let mut result = RingElement::zero(Representation::IncompleteNTT);
                let mut temp = RingElement::zero(Representation::IncompleteNTT);
                for j in 0..DF_MATRIX_WIDTH {
                    temp *= (&df_crs.data[(r, j)], &w[(base_row + j, col)]);
                    result += &temp;
                }
                result
            })
        };

        let zero = RingElement::zero(Representation::IncompleteNTT);

        // Check both columns
        let y_starts: [&[RingElement; DF_MATRIX_HEIGHT]; 2] = [&y_0, &df_output.y_int];
        let y_ends: [&[RingElement; DF_MATRIX_HEIGHT]; 2] = [&df_output.y_int, &df_output.y_t];

        for col in 0..2 {
            // First row: G * w_0 = -y_start
            let gw0 = recompose(0, col);
            for r in 0..DF_MATRIX_HEIGHT {
                assert_eq!(
                    gw0[r],
                    y_starts[col][r].negate(),
                    "Column {}, component {}: G * w_0 != -y_start",
                    col,
                    r
                );
            }

            // Middle rows: A * w_i + G * w_{i+1} = 0
            for i in 0..steps_per_col - 1 {
                let aw_i = inner_product_a(i * DF_MATRIX_WIDTH, col);
                let gw_next = recompose((i + 1) * DF_MATRIX_WIDTH, col);
                for r in 0..DF_MATRIX_HEIGHT {
                    let sum = &aw_i[r] + &gw_next[r];
                    assert_eq!(
                        sum,
                        zero,
                        "Column {}, step {}, component {}: A*w_{} + G*w_{} != 0",
                        col,
                        i + 1,
                        r,
                        i,
                        i + 1
                    );
                }
            }

            // Last row: A * w_last = y_end
            let aw_last = inner_product_a((steps_per_col - 1) * DF_MATRIX_WIDTH, col);
            for r in 0..DF_MATRIX_HEIGHT {
                assert_eq!(
                    aw_last[r], y_ends[col][r],
                    "Column {}, component {}: A * w_last != y_end",
                    col, r
                );
            }
        }
    }
}
