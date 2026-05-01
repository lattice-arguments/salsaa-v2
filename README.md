# Salsaa

A Rust implementation of Salsaa, a framework for constructing efficient and versatile lattice-based succinct arguments. The repository further implements three applications of SALSAA: a SNARK/PCS, a folding scheme and a VDF.

The codebase is an auxiliary material for *SALSAA: Sumcheck-Aided Lattice-based Succinct Arguments and Applications*, and has been built on top of the library provided by [RoKoKo](https://github.com/lattice-arguments/rokoko).

## Experiments
The codebase has been benchmarked on a Dell PowerEdge XE9680,  with a 2x32 core Xeon Platinum 8562Y+ 2.8GHz processor. Results of the benchmarks for each class of parameters can be found on the `/experiments` folder, together with benchmarks of RoKoKo and Greyhound.

## Build and Run instructions

The protocol can be compiled and run directly with the command, for example executing the SNARK protocol with rank 10 and witness dimension of 2^26 (in Z_q elements).
```
cargo +nightly run --release --features debug-hardness -- -r 10 -w 26 -m snark
```
Where the command line option are as follows
| Option | Argument   | Description                              |
|--------|---------- |------------------------------------------|
| -r     | numeric | Sets the rank parameter (e.g 9,10,11..)                  |
| -w     | numeric | Sets the power of the witness dimension (e.g `26`,`28`,`30` for 2^26, 2^28, 2^30 respectively) |
| -m     | `vdf`,`folding-scheme`,`snark`   | Selects the execution mode (one of `vdf`, `folding-scheme`, `snark`) |

Using the debug-hardness flag estimates the security bits against the RSIS attack, checking that they are at least 128bits for the given rank.
For optimal performance, run with `--features unsafe-sumcheck`, which uses further code optimisations in the sumcheck subprotocols.

### Using the HEXL backend
Note that by default, the codebase uses the Rokoko's `incomplete-rexl` backend for modular arithmetic and NTT operations. In order to run the Intel HEXL library backend instead, follow the [build instructions](https://github.com/lattice-arguments/rokoko#using-hexl-c-bindings) listed on their codebase and add the command flag `--no-default-features` when building Salsaa's codebase.

## Cached allocations
For the best performance, it is advisable to run the protocol twice. During the first run, the protocol collects the allocation descriptions (and stores them as a file, while printing a number of warnings about an unpopulated cache). On the next run, those allocations will be done in advance, which impact especially the commitment and verifier performance.

## Features

* `incomplete-rexl`: enables the pure-Rust ring arithmetic back-end
* `unsafe-sumcheck`: enables zero-cost borrow checking by using `UnsafeCell` instead of `RefCell` in sumcheck subprotocols
* `debug-hardness`: verifies the hardness of underlying SIS instances

## License

Salsaa is licensed under the Apache 2.0 License.
