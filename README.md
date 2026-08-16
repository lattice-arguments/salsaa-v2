# Salsaa

A Rust implementation of Salsaa, a framework for constructing efficient and versatile lattice-based succinct arguments. The repository further implements three applications of SALSAA: a SNARK/PCS, a folding scheme and a VDF.

The codebase is an auxiliary material for *SALSAA: Sumcheck-Aided Lattice-based Succinct Arguments and Applications*.

## Build and Run instructions

The protocol can be compiled and run directly with the command, for example executing the SNARK protocol with rank 11 and witness dimension of 2^26 (in Z_q elements).
```
cargo +nightly run --release --features debug-hardness -- -r 11 -w 26 -m snark
```
Where the command line option are as follows
| Option | Argument   | Description                              |
|--------|---------- |------------------------------------------|
| -r     | numeric | Sets the rank (height of commitment matrix, e.g 9,10,11..)                  |
| -w     | numeric | Sets the power of the witness dimension (e.g `26`,`28`,`30` for 2^26, 2^28, 2^30 respectively) |
| -m     | `vdf`,`folding-scheme`,`snark`   | Selects the execution mode (one of `vdf`, `folding-scheme`, `snark`) |

Using the debug-hardness flag estimates the security level (in bits) based on the SIS hardness, verifying that it is at least 128 bits for the given rank.
For optimal performance, run with `--features unsafe-sumcheck`, which uses further code optimisations in the sumcheck subprotocols.

### Modular ring arithmetic backend
The codebase uses the [incomplete-rexl](https://crates.io/crates/incomplete-rexl) backend for modular arithmetic and NTT operations.
It is required to compile and run the project on an AVX-512-enabled processor to achieve optimal performance.
Different processors may support different AVX-512 instruction subsets, [as listed here](https://en.wikipedia.org/wiki/AVX-512#CPUs_with_AVX-512).
Performance will be slower when the required instructions are not present, falling back to scalar code otherwise.

## Features

* `incomplete-rexl`: enables the pure-Rust ring arithmetic back-end
* `unsafe-sumcheck`: enables zero-cost borrow checking by using `UnsafeCell` instead of `RefCell` in sumcheck subprotocols
* `debug-hardness`: verifies the hardness of underlying SIS instances
* `debug`: additional checks for sumcheck claims, decomposition, and other operations

## License

Salsaa is licensed under the Apache 2.0 License.
