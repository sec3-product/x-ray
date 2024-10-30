use borsh::{BorshDeserialize, BorshSerialize};
use solana_program::account_info::{next_account_info, AccountInfo};
use solana_program::entrypoint::ProgramResult;
//use solana_program::msg;
use solana_program::program_error::ProgramError;
use solana_program::pubkey::Pubkey;

use crate::error::StakeError;
use crate::instruction::StakeInstruction;
use crate::state::StakingInfo;

pub struct Processor;

impl Processor {
    pub fn process(_program_id: &Pubkey, accounts: &[AccountInfo], input: &[u8]) -> ProgramResult {
        //msg!("counter: {:?}", input);
        let instruction = StakeInstruction::try_from_slice(input)?;
        match instruction {
            StakeInstruction::UpdateAdmin { admin } => Self::update_admin(accounts, admin),
        }
    }

    fn update_admin(accounts: &[AccountInfo], admin: [u8; 32]) -> ProgramResult {
        let acc_iter = &mut accounts.iter();
        let admin_info = next_account_info(acc_iter)?;
        let staking_info = next_account_info(acc_iter)?;

        // if !admin_info.is_signer {
        //     return Err(ProgramError::MissingRequiredSignature);
        // }

        let mut staking = StakingInfo::try_from_slice(&staking_info.data.borrow())?;

        if staking.admin == [0; 32] {
            staking.admin = admin;
        } else if staking.admin == admin_info.key.to_bytes() {
            staking.admin = admin;
        } else {
            return Err(StakeError::AdminRequired.into());
        }

        let _ = staking.serialize(&mut &mut staking_info.data.borrow_mut()[..]);

        Ok(())
    }
}
