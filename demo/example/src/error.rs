use solana_program::program_error::ProgramError;
use thiserror::Error;

#[derive(Error, Debug, Copy, Clone)]
pub enum StakeError {
    #[error("You have nothing to claim")]
    Underfunded,

    #[error("The time for claim is not yet ripe")]
    ClaimTimeError,

    #[error("The time for claim is not yet ripe")]
    StakeTimeError,

    #[error("Wrong staking PDA")]
    WrongStakingPDA,

    #[error("Admin signature is required")]
    AdminRequired,
}

impl From<StakeError> for ProgramError {
    fn from(e: StakeError) -> Self {
        ProgramError::Custom(e as u32)
    }
}
