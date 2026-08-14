use rokoko::common::{
    hash::HashWrapper,
    ring_arithmetic::{Representation, RingElement},
    structured_row::StructuredRow,
};

use crate::{common::config::PROJECTION_HEIGHT, protocol::config::RoundConfig};

pub struct BatchingChallenges {
    // in succinct form
    pub c0: StructuredRow<RingElement>,
    pub c1: StructuredRow<RingElement>,
    pub c2: StructuredRow<RingElement>,
}

impl BatchingChallenges {
    pub fn sample(config: &RoundConfig, hash_wrapper: &mut HashWrapper) -> Self {
        match config {
            RoundConfig::Intermediate {
                projection_ratio, ..
            } => {
                let c2_len = config.main_witness_columns;
                let c1_len = PROJECTION_HEIGHT;
                let single_col_height =
                    config.extended_witness_length / 2 / config.main_witness_columns;
                let c0_len: usize = single_col_height / (PROJECTION_HEIGHT * projection_ratio);
                assert!(c0_len > 0, "c0_len must be greater than 0");
                let mut result = Self {
                    c0: StructuredRow {
                        tensor_layers: vec![
                            RingElement::zero(Representation::IncompleteNTT);
                            c0_len.ilog2() as usize
                        ],
                    },
                    c1: StructuredRow {
                        tensor_layers: vec![
                            RingElement::zero(Representation::IncompleteNTT);
                            c1_len.ilog2() as usize
                        ],
                    },
                    c2: StructuredRow {
                        tensor_layers: vec![
                            RingElement::zero(Representation::IncompleteNTT);
                            c2_len.ilog2() as usize
                        ],
                    },
                };

                hash_wrapper
                    .sample_ring_element_ntt_slots_same_vec_into(&mut result.c0.tensor_layers);
                hash_wrapper
                    .sample_ring_element_ntt_slots_same_vec_into(&mut result.c1.tensor_layers);
                hash_wrapper
                    .sample_ring_element_ntt_slots_same_vec_into(&mut result.c2.tensor_layers);

                result
            }
            _ => panic!("Batching challenges should only be sampled for rounds with projection"),
        }
    }
}
