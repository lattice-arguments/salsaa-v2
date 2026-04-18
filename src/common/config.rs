use std::sync::OnceLock;

pub static DEGREE: usize = 128;
pub static HALF_DEGREE: usize = 64;
pub static MOD_Q: u64 = 1125899906839937;

pub static NOF_BATCHES: usize = 2;
// these should be re-exports from rokoko

pub const DEBUG: bool = true;

pub const DEBUG_HARDNESS: bool = true;

#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    SNARK,
    VDF,
    FOLDING_SCHEME,
}

pub const MODE: Mode = Mode::VDF;
// const WITNESS_DIM: usize = 2usize.pow(14);
pub const WITNESS_DIM: usize = 2usize.pow(20); // most can fit on 64 GB
pub const WITNESS_WIDTH: usize = 2usize;
pub const RANK: usize = 8;
pub const EXPECTED_SEC_PARAM: usize = 128;
static RANK_OVERRIDE: OnceLock<usize> = OnceLock::new();
static WITNESS_DIM_OVERRIDE: OnceLock<usize> = OnceLock::new();
static MODE_OVERRIDE: OnceLock<Mode> = OnceLock::new();

pub fn set_rank(rank: usize) -> Result<(), String> {
    if rank == 0 {
        return Err("rank must be greater than 0".to_string());
    }

    RANK_OVERRIDE
        .set(rank)
        .map_err(|_| "rank has already been set".to_string())
}

#[inline(always)]
pub fn rank() -> usize {
    *RANK_OVERRIDE.get().unwrap_or(&RANK)
}

pub fn set_witness_dim(witness_dim: usize) -> Result<(), String> {
    if witness_dim == 0 {
        return Err("witness_dim must be greater than 0".to_string());
    }

    WITNESS_DIM_OVERRIDE
        .set(witness_dim)
        .map_err(|_| "witness_dim has already been set".to_string())
}

#[inline(always)]
pub fn witness_dim() -> usize {
    *WITNESS_DIM_OVERRIDE.get().unwrap_or(&WITNESS_DIM)
}

pub fn set_mode(mode: Mode) -> Result<(), String> {
    MODE_OVERRIDE
        .set(mode)
        .map_err(|_| "mode has already been set".to_string())
}

#[inline(always)]
pub fn mode() -> Mode {
    *MODE_OVERRIDE.get().unwrap_or(&MODE)
}

pub fn witness_height() -> usize {
    // TODO divide by 2 if SNARK/folding scheme since we docomp
    witness_dim() / WITNESS_WIDTH
}

pub const VDF_MATRIX_HEIGHT: usize = 4;
pub const VDF_BITS: usize = 64;
pub const VDF_MATRIX_WIDTH: usize = VDF_BITS * VDF_MATRIX_HEIGHT;

/// Step stride for c-powers: consecutive steps are spaced VDF_MATRIX_HEIGHT apart.
/// Within each step, G uses c^{0..HEIGHT-1} and A uses c^{HEIGHT..2*HEIGHT-1}.
/// A-powers for step i overlap with G-powers for step i+1, giving telescoping.
pub const VDF_STRIDE: usize = VDF_MATRIX_HEIGHT;

pub const NUM_COLUMNS_INITIAL: usize = 2;

pub const PROJECTION_HEIGHT: usize = 256;

pub const MAX_UNSTRUCT_PROJ_RATIO: usize = 8; // to prevent degenerate projection configs with very large proj matrx

pub const LAST_ROUND_THRESHOLD: usize = 256;
