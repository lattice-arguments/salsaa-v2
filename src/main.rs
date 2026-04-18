use rokoko::common::init_common;
use rokoko::common::pool::{load_and_preallocate, save_access_stats};
use salsaa::common::config::{rank, set_rank};
use salsaa::protocol::parties::executor::execute;

fn parse_rank_from_cli() -> Result<Option<usize>, String> {
    let mut rank_arg: Option<usize> = None;
    let mut args = std::env::args().skip(1);

    while let Some(arg) = args.next() {
        if arg == "--rank" || arg == "-r" {
            let value = args
                .next()
                .ok_or_else(|| "missing value for --rank".to_string())?;
            let parsed = value
                .parse::<usize>()
                .map_err(|_| format!("invalid rank value: {value}"))?;
            rank_arg = Some(parsed);
        } else if let Some(value) = arg.strip_prefix("--rank=") {
            let parsed = value
                .parse::<usize>()
                .map_err(|_| format!("invalid rank value: {value}"))?;
            rank_arg = Some(parsed);
        } else if arg == "--help" || arg == "-h" {
            println!("Usage: salsaa [--rank <usize>] [-r <usize>]");
            std::process::exit(0);
        } else {
            return Err(format!("unknown argument: {arg}"));
        }
    }

    Ok(rank_arg)
}

fn main() {
    if let Some(cli_rank) = parse_rank_from_cli().unwrap_or_else(|err| {
        eprintln!("CLI error: {err}");
        std::process::exit(2);
    }) {
        set_rank(cli_rank).unwrap_or_else(|err| {
            eprintln!("Configuration error: {err}");
            std::process::exit(2);
        });
    }
    println!("Using rank={}", rank());

    #[cfg(feature = "unsafe-sumcheck")]
    {
        println!("Sumcheck unsafe...");
    }

    // Check AVX-512F support
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            println!("✓ AVX-512F is enabled in runtime detection and available on this CPU");
            #[cfg(all(target_feature = "avx512f"))]
            {
                println!("✓✓ AVX-512F is enabled at compile time");
            }
            #[cfg(not(target_feature = "avx512f"))]
            {
                println!("✗ AVX-512F is NOT enabled at compile time");
            }
        } else {
            println!("✗ AVX-512F is NOT available on this CPU");
        }

        if is_x86_feature_detected!("avx512dq") {
            println!("✓ AVX-512DQ is enabled in runtime detection and available on this CPU");
            #[cfg(all(target_feature = "avx512dq"))]
            {
                println!("✓✓ AVX-512DQ is enabled at compile time");
            }
            #[cfg(not(target_feature = "avx512dq"))]
            {
                println!("✗ AVX-512DQ is NOT enabled at compile time");
            }
        } else {
            println!("✗ AVX-512DQ is NOT available on this CPU");
        }
        if is_x86_feature_detected!("avx512vbmi2") {
            println!("✓ AVX-512VBMI2 is enabled in runtime detection and available on this CPU");
            #[cfg(all(target_feature = "avx512vbmi2"))]
            {
                println!("✓✓ AVX-512VBMI2 is enabled at compile time");
            }
            #[cfg(not(target_feature = "avx512vbmi2"))]
            {
                println!("✗ AVX-512VBMI2 is NOT enabled at compile time");
            }
        } else {
            println!("✗ AVX-512VBMI2 is NOT available on this CPU");
        }
    }

    #[cfg(feature = "incomplete-rexl")]
    {
        // Trigger CPU feature detection and print features if enabled
        //incomplete_rexl::cpu_features::print_features();
    }

    #[cfg(not(target_arch = "x86_64"))]
    {
        println!("✗ AVX-512 is only available on x86_64 architecture");
    }

    load_and_preallocate("pool_stats.txt").expect("Failed to load stats");
    init_common();
    println!("Running executor...");
    execute();
    save_access_stats("pool_stats.txt").expect("Failed to save stats");
}
