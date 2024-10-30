use borsh::{BorshDeserialize, BorshSerialize};



#[derive(BorshSerialize, BorshDeserialize, Debug, Clone, PartialEq)]
pub enum StakeInstruction {
    UpdateAdmin {
        admin: [u8; 32],
    },
}
