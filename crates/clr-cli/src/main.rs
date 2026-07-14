use clr_core::runtime_info;
use clr_qwen3_moe::{GenerationSession, frozen_tiny_model};

fn main() {
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    match execute(&arguments) {
        Ok(output) => println!("{output}"),
        Err(error) => {
            eprintln!("error: {error}");
            std::process::exit(2);
        }
    }
}

fn execute(arguments: &[String]) -> Result<String, String> {
    if arguments.is_empty() {
        let info = runtime_info();
        return Ok(format!(
            "{} {}\nhost: {}-{}\nstatus: bootstrap ready",
            info.name, info.version, info.architecture, info.operating_system
        ));
    }
    if arguments.first().map(String::as_str) != Some("generate") {
        return Err(format!("unknown command '{}'", arguments[0]));
    }
    let options = GenerateOptions::parse(&arguments[1..])?;
    generate(&options)
}

#[derive(Debug, PartialEq)]
struct GenerateOptions {
    tokens: Vec<usize>,
    max_new_tokens: usize,
    temperature: Option<f32>,
    seed: u64,
}

impl GenerateOptions {
    fn parse(arguments: &[String]) -> Result<Self, String> {
        let mut tokens = None;
        let mut max_new_tokens = None;
        let mut temperature = None;
        let mut seed = None;
        let mut index = 0;
        while index < arguments.len() {
            let flag = arguments[index].as_str();
            let value = arguments
                .get(index + 1)
                .ok_or_else(|| format!("missing value for {flag}"))?;
            match flag {
                "--tokens" if tokens.is_none() => tokens = Some(parse_tokens(value)?),
                "--max-new-tokens" if max_new_tokens.is_none() => {
                    max_new_tokens = Some(parse_value(value, flag)?);
                }
                "--temperature" if temperature.is_none() => {
                    temperature = Some(parse_value(value, flag)?);
                }
                "--seed" if seed.is_none() => seed = Some(parse_value(value, flag)?),
                "--tokens" | "--max-new-tokens" | "--temperature" | "--seed" => {
                    return Err(format!("duplicate option {flag}"));
                }
                _ => return Err(format!("unknown option {flag}")),
            }
            index += 2;
        }
        Ok(Self {
            tokens: tokens.ok_or_else(|| "missing required --tokens".to_string())?,
            max_new_tokens: max_new_tokens
                .ok_or_else(|| "missing required --max-new-tokens".to_string())?,
            temperature,
            seed: seed.unwrap_or(0),
        })
    }
}

fn parse_tokens(value: &str) -> Result<Vec<usize>, String> {
    if value.is_empty() {
        return Err("--tokens must contain at least one token ID".to_string());
    }
    value
        .split(',')
        .map(|token| {
            token
                .parse::<usize>()
                .map_err(|_| format!("invalid token ID '{token}'"))
        })
        .collect()
}

fn parse_value<T>(value: &str, flag: &str) -> Result<T, String>
where
    T: std::str::FromStr,
{
    value
        .parse()
        .map_err(|_| format!("invalid value '{value}' for {flag}"))
}

fn generate(options: &GenerateOptions) -> Result<String, String> {
    let capacity = options
        .tokens
        .len()
        .checked_add(options.max_new_tokens)
        .ok_or_else(|| "requested context length overflowed".to_string())?;
    let model = frozen_tiny_model().map_err(|error| error.to_string())?;
    let mut session = GenerationSession::resident(&model, capacity, options.seed)
        .map_err(|error| error.to_string())?;
    session
        .prefill(&options.tokens)
        .map_err(|error| error.to_string())?;
    let mut generated = Vec::with_capacity(options.max_new_tokens);
    for _ in 0..options.max_new_tokens {
        let token = match options.temperature {
            Some(temperature) => session
                .decode_temperature(temperature)
                .map_err(|error| error.to_string())?,
            None => session.decode_greedy().map_err(|error| error.to_string())?,
        };
        generated.push(token);
    }
    Ok(format!(
        "generated: {}\nsequence: {}\nkv-cache: {} bytes, {}/{} tokens",
        join_tokens(&generated),
        join_tokens(session.sequence()),
        session.cache().byte_size(),
        session.cache().len(),
        session.cache().capacity()
    ))
}

fn join_tokens(tokens: &[usize]) -> String {
    tokens
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(",")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arguments(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| (*value).to_string()).collect()
    }

    #[test]
    fn no_arguments_preserves_bootstrap_smoke_output() {
        let output = execute(&[]).expect("bootstrap output");
        assert!(output.contains("colibri-lite-rs"));
        assert!(output.contains("status: bootstrap ready"));
    }

    #[test]
    fn generate_accepts_token_ids_and_emits_cache_accounting() {
        let output = execute(&arguments(&[
            "generate",
            "--tokens",
            "1,7,3,12",
            "--max-new-tokens",
            "2",
        ]))
        .expect("greedy generation");

        assert!(output.contains("generated: 10"));
        assert!(output.contains("sequence: 1,7,3,12"));
        assert!(output.contains("6/6 tokens"));
    }

    #[test]
    fn generate_rejects_missing_invalid_and_duplicate_options() {
        assert_eq!(
            execute(&arguments(&["generate", "--max-new-tokens", "1"])),
            Err("missing required --tokens".to_string())
        );
        assert_eq!(
            execute(&arguments(&[
                "generate",
                "--tokens",
                "1,nope",
                "--max-new-tokens",
                "1",
            ])),
            Err("invalid token ID 'nope'".to_string())
        );
        assert_eq!(
            execute(&arguments(&[
                "generate",
                "--tokens",
                "1",
                "--tokens",
                "2",
                "--max-new-tokens",
                "1",
            ])),
            Err("duplicate option --tokens".to_string())
        );
    }
}
