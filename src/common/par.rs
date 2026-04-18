//! Feature-gated rayon wrappers.
//!
//! With the `parallel` feature on, the `maybe_par_*!` macros expand to
//! rayon's parallel iterators; without it they fall back to std iterators.
//! Call sites use the same macro names either way, so the `#[cfg]` lives
//! only in this module.
//!
//! Macros (not functions) because the two branches return different types
//! — expanding inline lets each call site use whichever methods exist on
//! the active iterator type, rather than being limited to the intersection.
//!
//! When the `parallel` feature is on, `rayon::prelude::*` is re-exported
//! from this module for convenience (`use crate::common::par::*;`).

#[cfg(feature = "parallel")]
pub use rayon::prelude::*;

/// `maybe_par_iter!(expr)` → `expr.par_iter()` with `parallel`, else `expr.iter()`.
#[macro_export]
macro_rules! maybe_par_iter {
    ($e:expr) => {{
        #[cfg(feature = "parallel")]
        {
            rayon::prelude::IntoParallelRefIterator::par_iter(&$e)
        }
        #[cfg(not(feature = "parallel"))]
        {
            ($e).iter()
        }
    }};
}

/// `maybe_par_iter_mut!(expr)` → `expr.par_iter_mut()` or `expr.iter_mut()`.
#[macro_export]
macro_rules! maybe_par_iter_mut {
    ($e:expr) => {{
        #[cfg(feature = "parallel")]
        {
            rayon::prelude::IntoParallelRefMutIterator::par_iter_mut(&mut $e)
        }
        #[cfg(not(feature = "parallel"))]
        {
            ($e).iter_mut()
        }
    }};
}

/// `maybe_into_par_iter!(expr)` → `expr.into_par_iter()` or `expr.into_iter()`.
#[macro_export]
macro_rules! maybe_into_par_iter {
    ($e:expr) => {{
        #[cfg(feature = "parallel")]
        {
            rayon::prelude::IntoParallelIterator::into_par_iter($e)
        }
        #[cfg(not(feature = "parallel"))]
        {
            IntoIterator::into_iter($e)
        }
    }};
}

/// `maybe_par_chunks!(slice, n)` → `.par_chunks(n)` or `.chunks(n)`.
#[macro_export]
macro_rules! maybe_par_chunks {
    ($e:expr, $n:expr) => {{
        #[cfg(feature = "parallel")]
        {
            rayon::slice::ParallelSlice::par_chunks(&$e[..], $n)
        }
        #[cfg(not(feature = "parallel"))]
        {
            ($e).chunks($n)
        }
    }};
}

/// `maybe_par_join!(a, b)` → `rayon::join(a, b)` with `parallel`, else `(a(), b())`.
#[macro_export]
macro_rules! maybe_par_join {
    ($a:expr, $b:expr) => {{
        #[cfg(feature = "parallel")]
        {
            rayon::join($a, $b)
        }
        #[cfg(not(feature = "parallel"))]
        {
            (($a)(), ($b)())
        }
    }};
}

/// `maybe_par_chunks_mut!(slice, n)` → `.par_chunks_mut(n)` or `.chunks_mut(n)`.
#[macro_export]
macro_rules! maybe_par_chunks_mut {
    ($e:expr, $n:expr) => {{
        #[cfg(feature = "parallel")]
        {
            rayon::slice::ParallelSliceMut::par_chunks_mut(&mut $e[..], $n)
        }
        #[cfg(not(feature = "parallel"))]
        {
            ($e).chunks_mut($n)
        }
    }};
}
