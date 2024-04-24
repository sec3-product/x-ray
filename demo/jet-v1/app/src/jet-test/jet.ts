import {
  Keypair,
  PublicKey,
  TransactionSignature,
} from "@solana/web3.js";
import type { Connection } from "@solana/web3.js";
import type { BN, Program, Wallet } from "@project-serum/anchor";
import { DataManager } from "./data";
import { TestToken, TestUtils, toPublicKeys } from ".";
import { TOKEN_PROGRAM_ID } from "@solana/spl-token";
import * as anchor from "@project-serum/anchor";


export interface ReserveAccounts {
  accounts: {
    reserve: Keypair;

    vault: PublicKey;
    tokenSource: PublicKey;
    tokenMint: TestToken;

    loanNoteMint: PublicKey;
    depositNoteMint: PublicKey;
    depositNoteDest: PublicKey;
  };
  bump: {
    vault: number;
    loanNoteMint: number;
    depositNoteMint: number;
    depositNoteDest: number;
  };
}


export interface ReserveConfig {
  utilizationRate1: BN,
  utilizationRate2: BN,
  borrowRate0: BN,
  borrowRate1: BN,
  borrowRate2: BN,
  borrowRate3: BN,
  minCollateralRatio: BN,
  liquidationPremium: BN
}

export class JetUtils {
  static readonly programId = DataManager.programId;

  conn: Connection;
  wallet: Wallet;
  config: DataManager;
  utils: TestUtils;
  program: Program;

  constructor(conn: Connection, wallet: Wallet, program: Program) {
    this.conn = conn;
    this.wallet = wallet;
    this.config = new DataManager(conn, wallet);
    this.utils = new TestUtils(conn, wallet);
    this.program = program;
  }

  public async createReserveAccount(
    tokenMint: TestToken,
    initial_amount: BN,
    market: Keypair,
    marketOwner: PublicKey
  ): Promise<ReserveAccounts> {
    const reserve = Keypair.generate();
    const [depositNoteMint, depositNoteMintBump] =
      await this.utils.findProgramAddress(this.program.programId, [
        "deposits",
        reserve,
        tokenMint,
      ]);
    const [loanNoteMint, loanNoteMintBump] = await this.utils.findProgramAddress(
      this.program.programId,
      ["loans", reserve, tokenMint]
    );
    const [depositNoteDest, depositNoteDestBump] =
      await this.utils.findProgramAddress(this.program.programId, [
        reserve,
        this.wallet,
      ]);
    const [vault, vaultBump] = await this.utils.findProgramAddress(
      this.program.programId,
      [market, reserve]
    );
    const tokenSource = await this.utils.createTokenAccount(
      tokenMint,
      marketOwner,
      initial_amount
    );

    return {
      accounts: {
        reserve,
        tokenSource,
        vault,
        tokenMint,

        depositNoteMint,
        depositNoteDest,

        loanNoteMint,
      },

      bump: {
        vault: vaultBump,
        depositNoteMint: depositNoteMintBump,
        depositNoteDest: depositNoteDestBump,
        loanNoteMint: loanNoteMintBump,
      },
    };
  }

  public async initReserve(
    reserve: ReserveAccounts,
    reserveConfig: ReserveConfig,
    liquidityAmount: BN,
    market: PublicKey,
    marketOwner: PublicKey,
    pythProduct: PublicKey,
    pythPrice: PublicKey,
  ): Promise<TransactionSignature> {
    let [marketAuthority] = await this.findMarketAuthorityAddress(market);

    return await this.program.rpc.initReserve(reserve.bump, reserveConfig, liquidityAmount, {
      accounts: toPublicKeys({
        market,
        marketAuthority,
        owner: marketOwner,

        oracleProduct: pythProduct,
        oraclePrice: pythPrice,

        tokenProgram: TOKEN_PROGRAM_ID,
        clock: anchor.web3.SYSVAR_CLOCK_PUBKEY,
        rent: anchor.web3.SYSVAR_RENT_PUBKEY,
        systemProgram: anchor.web3.SystemProgram.programId,

        ...reserve.accounts,
      }),
      signers: [reserve.accounts.reserve],
      instructions: [
        await this.program.account.reserve.createInstruction(
          reserve.accounts.reserve
        ),
      ],
    });
  }

  public async findMarketAuthorityAddress(market: PublicKey) {
    return PublicKey.findProgramAddress(
      [market.toBuffer()],
      this.program.programId
    );
  }
}