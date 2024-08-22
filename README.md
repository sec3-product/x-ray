# X-Ray: Solana Programs Static Analysis Tool

## Introduction

The X-Ray Toolchain is an open-source, cross-platform command-line interface
(CLI) tool designed for static analysis of Solana programs and smart contracts
written in Rust. The tool parses Rust programs, generates an Abstract Syntax
Tree (AST), converts the AST to an LLVM Intermediate Representation (LLVM-IR)
format, and applies static analysis over the LLVM-IR to capture potential
issues.

One of the primary goals of this project is to provide a general and extensible
platform for Solana smart contracts code parsing and analysis. During the
analysis process, it tracks and monitors relevant instructions and accounts
information, facilitating detailed analysis and verification of smart
contracts.

In this open-source version, we have compiled and provided several built-in
common rules as examples. These rules are ready to use in analyzing your
programs and serve as references for extending your own custom rules. You can
define additional bug/security-related rules to suit your specific needs.

Additionally, we offer a production version of the tool, which includes a more
comprehensive set of rules for scanning and analysis. We invite you to try it
out for even deeper insights and broader coverage.

### Built-in Rules for Solana Smart Contracts

The X-Ray toolchain provides several built-in rules designed to detect common
vulnerabilities and issues in Solana smart contracts. Below is a summary of
these rules, including their descriptions and related references.

| **Rule ID** | **Rule Name**              | **Description**                                                                                  | **Reference**                                                                                                      |
|-------------|----------------------------|--------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------|
| 1004        | **IntegerUnderflow**       | Detects potential underflows in operations that could lead to erroneous behavior.                | [Understanding Arithmetic Overflow/Underflows in Rust and Solana](https://www.sec3.dev/blog/understanding-arithmetic-overflow-underflows-in-rust-and-solana-smart-contracts) |
| 1003-1006   | **IntegerOverflow**        | Identifies add, mul and div operations that may result in overflows, causing unexpected outcomes or incorrect calculations. | [Understanding Arithmetic Overflow/Underflows in Rust and Solana](https://www.sec3.dev/blog/understanding-arithmetic-overflow-underflows-in-rust-and-solana-smart-contracts) |
| 1007        | **UnverifiedParsedAccount**| Detects cases where account data is parsed without prior validation, which could lead to misuse. | [Account Data Matching in Solana](https://github.com/project-serum/sealevel-attacks/tree/master/programs/1-account-data-matching) |
| 1010        | **TypeFullCosplay**        | Detects type confusion vulnerabilities where fully compatible data types can be exploited.       | [Type Confusion Attacks in Solana](https://github.com/project-serum/sealevel-attacks/tree/master/programs/3-type-cosplay) |
| 1011        | **TypePartialCosplay**     | Identifies partial type confusion vulnerabilities that might be exploited in attacks.            | [Type Confusion Attacks in Solana](https://github.com/project-serum/sealevel-attacks/tree/master/programs/3-type-cosplay) |
| 1014        | **BumpSeedNotValidated**   | Identifies situations where the bump seed of a program address is not validated, risking misuse. | [Bump Seed Canonicalization Attacks](https://github.com/project-serum/sealevel-attacks/tree/master/programs/7-bump-seed-canonicalization) |
| 1015        | **InsecurePDASharing**     | Warns against insecure sharing of program-derived addresses (PDA) which might be exploited.      | [PDA Sharing Vulnerabilities](https://github.com/project-serum/sealevel-attacks/tree/master/programs/8-pda-sharing) |
| 1017        | **MaliciousSimulation**    | Detects potentially malicious code that simulates transactions, posing security risks.           | [Detecting Transaction Simulation](https://opcodes.fr/en/publications/2022-01/detecting-transaction-simulation/)    |
| 2001        | **IncorrectLoopBreakLogic**| Detects incorrect loop break logic, which may cause unintended control flow.                     | [Jet Protocol Loop Break Issue](https://twitter.com/JetProtocol/status/1476244740601524234)                         |
| 2002        | **IncorrectConditionCheck**| Identifies condition checks that use incorrect operators, leading to faulty logic.               | [Incorrect Liquidation Condition](https://www.sec3.dev/blog/how-to-audit-solana-smart-contracts-part-1-a-systematic-approach) |
| 2003        | **ExponentialCalculation** | Flags calculations with exponential complexity that could degrade performance.                   | [Exponential Complexity in Calculations](https://www.sec3.dev/blog/how-to-audit-solana-smart-contracts-part-1-a-systematic-approach) |
| 2004        | **IncorrectDivisionLogic** | Detects division logic errors, such as incorrect use of `checked_div` instead of `checked_ceil_div`. | [SPL Token Swap Division Issue](https://github.com/solana-labs/solana-program-library/pull/2942) |
| 2005        | **IncorrectTokenCalculation** | Identifies incorrect token amount calculations, which could result in financial inaccuracies.    | [Incorrect Token Calculation in SushiSwap](https://medium.com/certora/exploiting-an-invariant-break-how-we-found-a-pool-draining-bug-in-sushiswaps-trident-585bd98a4d4f) |

A full list of the supported bug patterns for Solana contracts is saved in
[xray.json](package/conf/xray.json).

## Installation

### Using Prebuilt Binaries

TBA

### Using Prebuilt Container Images

TBA

### Building From Source

For developers who wish to build X-Ray from source, please refer to
[Building from Source](docs/developer.md#building-from-source) for detailed
instructions.

## Usage

### Prepare a Target Repository

To scan a target repository, simply clone it and run the container or binary
CLI, providing the path to the project's root directory -- there's no need to
locate the specific source code files.

For example, we will use
[Solana Labs' Helloworld](https://github.com/solana-labs/example-helloworld.git)
repository. First, clone the repository to your local workspace.

```sh
mkdir -p workspace
git clone https://github.com/solana-labs/example-helloworld.git workspace/example-helloworld
```

### Start a scan

```sh
docker run --rm -v $(pwd)/workspace:/workspace /workspace/example-helloworld
```

Alternatively you can use the installed native binary:

```sh
/path/to/extracted/xray workspace/example-helloworld
```

### Example Output

X-Ray will report each detected potential issues along with its code snippet.
At the end of the output, X-Ray will provide a summary.

```
Analyzing /home/sec3/x-ray-toolchain/workspace/program-rust/workspace_program-rust.ll ...
Detecting Vulnerabilities
==============VULNERABLE: IntegerAddOverflow!============
Found a potential vulnerability at line 43, column 29 in workspace/program-rust/src/lib.rs
The add operation may result in overflows:
 37|        msg!("Greeted account does not have the correct program id");
 38|        return Err(ProgramError::IncorrectProgramId);
 39|    }
 40|
 41|    // Increment and store the number of times the account has been greeted
 42|    let mut greeting_account = GreetingAccount::try_from_slice(&account.data.borrow())?;
>43|    greeting_account.counter += 1;
 44|    greeting_account.serialize(&mut &mut account.data.borrow_mut()[..])?;
 45|
 46|    msg!("Greeted {} time(s)!", greeting_account.counter);
 47|
 48|    Ok(())
 49|}
>>>Stack Trace:
>>>sol.process_instruction [workspace/program-rust/src/lib.rs:19]

For more info, see https://www.sec3.dev/blog/understanding-arithmetic-overflow-underflows-in-rust-and-solana-smart-contracts

--------The summary of potential vulnerabilities in workspace_program-rust.ll--------

         1 unsafe operation issues
```

You can also find the JSON report in the `.xray` directory (or
`workspace/.xray` if using container version).

## Developer Guide

Please refer to [Developer Guide](docs/developer.md) for detailed instructions.
