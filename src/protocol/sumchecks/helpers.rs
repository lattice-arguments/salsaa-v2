use rokoko::{
    common::ring_arithmetic::RingElement,
    protocol::{
        commitment::Prefix,
        sumcheck_utils::{
            elephant_cell::ElephantCell, selector_eq::SelectorEq,
        },
    },
};

// TODO: make the function in Rokoko public and delete from here
pub fn sumcheck_from_prefix(
    prefix: &Prefix,
    total_vars: usize,
) -> ElephantCell<SelectorEq<RingElement>> {
    ElephantCell::new(SelectorEq::<RingElement>::new(
        prefix.prefix,
        prefix.length,
        total_vars,
    ))
}
