use borsh::{BorshDeserialize, BorshSerialize};
use solana_program::pubkey::Pubkey;

use crate::{id, STAKING_SEED , STAKING_INFO_SEED };


#[derive(BorshSerialize, BorshDeserialize, Debug)]
pub struct Staking {
    
    pub amount: u128,

    pub reward_allowed: u128,

    pub reward_debt: u128,
    
    pub distributed: u128,

    pub reward: u64,
}


impl Staking {
    pub fn get_staking_pubkey(user: &Pubkey) -> Pubkey {
        Pubkey::create_with_seed(user, STAKING_SEED, &id()).unwrap()
    }

    pub fn is_ok_counter_pubkey(user: &Pubkey, counter: &Pubkey) -> bool {
        counter.to_bytes() == Self::get_staking_pubkey(user).to_bytes()
    }
}


#[derive(BorshSerialize, BorshDeserialize, Debug)]
pub struct StakingInfo { 
    pub admin: [u8; 32],
    pub start_time: i64 , 
    pub distribution_time: i64, 
    pub unlock_claim_time: i64, 
    pub reward_total: u128 ,
    pub total_staked: u128 , 
    pub total_distributed: u128 ,
    pub tokens_per_stake: u128 , 
    pub reward_produced: u128 , 
    pub all_produced: u128 , 
    pub produced_time: i64 , 
}

      

impl StakingInfo {
    pub fn get_settings_pubkey_with_bump() -> (Pubkey, u8) {
        Pubkey::find_program_address(&[STAKING_INFO_SEED.as_bytes()], &id())
    }

    pub fn get_settings_pubkey() -> Pubkey {
        let (pubkey, _) = Self::get_settings_pubkey_with_bump();
        pubkey
    }

    pub fn is_ok_settings_pubkey(settings_pubkey: &Pubkey) -> bool {
        let (pubkey, _) = Self::get_settings_pubkey_with_bump();
        pubkey.to_bytes() == settings_pubkey.to_bytes()
    }
}


