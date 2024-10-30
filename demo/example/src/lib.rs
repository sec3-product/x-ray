pub mod error;
pub mod instruction;
pub mod processor;
pub mod state;

#[cfg(not(feature = "no-entrypoint"))]
pub mod entrypoint;

pub const STAKING_SEED: &str = "staking";
pub const STAKING_INFO_SEED: &str = "stakinginfo" ; 


solana_program::declare_id!("2s8x7Jn2KuFJtHFouBpxeMLrvaZtwyZHuTF1X5PmTg93");