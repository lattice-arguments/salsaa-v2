use rokoko::common::init_common;
use rokoko::common::pool::{load_and_preallocate, save_access_stats};
use salsaa::common::config::{Mode, mode, rank, set_mode, set_rank, set_witness_dim, witness_dim};
use salsaa::protocol::parties::executor::execute;

struct CliConfig {
    rank: Option<usize>,
    witness_dim: Option<usize>,
    mode: Option<Mode>,
}

fn parse_mode(value: &str) -> Result<Mode, String> {
    match value.to_ascii_lowercase().as_str() {
        "snark" => Ok(Mode::SNARK),
        "vdf" => Ok(Mode::VDF),
        "folding" | "folding_scheme" | "folding-scheme" => Ok(Mode::FOLDING_SCHEME),
        _ => Err(format!(
            "invalid mode value: {value} (expected one of: snark, vdf, folding-scheme)"
        )),
    }
}

fn parse_witness_dim_log(value: &str) -> Result<usize, String> {
    let log_dim = value
        .parse::<usize>()
        .map_err(|_| format!("invalid witness dim log value: {value}"))?;

    if log_dim >= usize::BITS as usize {
        return Err(format!(
            "witness dim log is too large for this platform: {log_dim}"
        ));
    }

    Ok(1usize << log_dim)
}

fn parse_cli_config() -> Result<CliConfig, String> {
    let mut cfg = CliConfig {
        rank: None,
        witness_dim: None,
        mode: None,
    };
    let mut args = std::env::args().skip(1);

    while let Some(arg) = args.next() {
        if arg == "--rank" || arg == "-r" {
            let value = args
                .next()
                .ok_or_else(|| "missing value for --rank".to_string())?;
            let parsed = value
                .parse::<usize>()
                .map_err(|_| format!("invalid rank value: {value}"))?;
            cfg.rank = Some(parsed);
        } else if let Some(value) = arg.strip_prefix("--rank=") {
            let parsed = value
                .parse::<usize>()
                .map_err(|_| format!("invalid rank value: {value}"))?;
            cfg.rank = Some(parsed);
        } else if arg == "--wit-dim-log" {
            let value = args
                .next()
                .ok_or_else(|| "missing value for --wit-dim-log".to_string())?;
            cfg.witness_dim = Some(parse_witness_dim_log(&value)?);
        } else if let Some(value) = arg.strip_prefix("--wit-dim-log=") {
            cfg.witness_dim = Some(parse_witness_dim_log(value)?);
        } else if arg == "--mode" || arg == "-m" {
            let value = args
                .next()
                .ok_or_else(|| "missing value for --mode".to_string())?;
            cfg.mode = Some(parse_mode(&value)?);
        } else if let Some(value) = arg.strip_prefix("--mode=") {
            cfg.mode = Some(parse_mode(value)?);
        } else if arg == "--help" || arg == "-h" {
            println!(
                "Usage: salsaa [--rank <usize>] [--wit-dim-log <log2>] [--mode <snark|vdf|folding-scheme>]"
            );
            println!("Short flags: -r <usize> -m <mode>");
            std::process::exit(0);
        } else {
            return Err(format!("unknown argument: {arg}"));
        }
    }

    Ok(cfg)
}

fn main() {
    let cli_cfg = parse_cli_config().unwrap_or_else(|err| {
        eprintln!("CLI error: {err}");
        std::process::exit(2);
    });

    if let Some(cli_rank) = cli_cfg.rank {
        set_rank(cli_rank).unwrap_or_else(|err| {
            eprintln!("Configuration error: {err}");
            std::process::exit(2);
        });
    }

    if let Some(cli_witness_dim) = cli_cfg.witness_dim {
        set_witness_dim(cli_witness_dim).unwrap_or_else(|err| {
            eprintln!("Configuration error: {err}");
            std::process::exit(2);
        });
    }

    if let Some(cli_mode) = cli_cfg.mode {
        set_mode(cli_mode).unwrap_or_else(|err| {
            eprintln!("Configuration error: {err}");
            std::process::exit(2);
        });
    }

    let witness_dim_log = witness_dim().ilog2();
    println!(
        "Using rank={}, witness_dim=2^{witness_dim_log} ({}), mode={:?}",
        rank(),
        witness_dim(),
        mode()
    );

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
